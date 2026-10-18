"""
Regression suite for stage 7 — the HTTP contract.

Runs against a stub pipeline injected into module state, so no index, no model
download and no network. What is being checked is the contract: does a
malformed request get rejected before any work happens, is an abstention a
200 rather than an error, does every citation carry the fields a caller needs
to verify the answer.

Run:  pytest test_api.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient          # noqa: E402

import api                                          # noqa: E402
from rag import RagPipeline                         # noqa: E402
from retrieval import BM25, tokenize                # noqa: E402

PASSAGES = [
    ("SUPP", "Our manufacturing operations rely on a small number of vendors "
             "for critical components. If any vendor fails to deliver on "
             "schedule we could experience material delays in production."),
    ("CYBR", "We rely on information systems to process transactions and store "
             "confidential customer information. Our systems may be vulnerable "
             "to attacks by hackers or breaches caused by employee error."),
    ("RATE", "A substantial portion of our indebtedness bears interest at "
             "variable rates. If market interest rates increase our debt "
             "service obligations would rise materially."),
]


@pytest.fixture
def client(monkeypatch):
    df = pd.DataFrame([
        {"chunk_id": tag, "cik": 100 + i, "ticker": tag, "filing_year": 2023,
         "text": text, "heading": "", "n_chars": len(text)}
        for i, (tag, text) in enumerate(PASSAGES)])

    vocab = [w for w in sorted({w for _, t in PASSAGES for w in tokenize(t)})
             if len(w) > 3]
    index = {w: i for i, w in enumerate(vocab)}

    def encode(texts):
        out = np.zeros((len(texts), len(index)), dtype="float32")
        for r, t in enumerate(texts):
            for w in tokenize(t):
                if w in index:
                    out[r, index[w]] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    emb = encode(df["text"].tolist())
    bm25 = BM25([tokenize(t) for t in df["text"]])

    api.state.pipeline = RagPipeline(df, emb, bm25, encode, abstain_at=0.15)
    from text_features import LM_CATEGORIES
    api.state.lm = {c: {"RISK", "LOSS", "ADVERSE"} for c in LM_CATEGORIES}
    api.state.meta = {"n_chunks": len(df), "n_firms": 3, "device": "cpu"}
    api.state.latencies = {}

    with TestClient(api.app) as c:
        yield c

    api.state.pipeline = None


# ────────────────────────────────────────────────────────────────── health
def test_health_reports_what_is_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["n_chunks"] == 3


# ────────────────────────────────────────────────────────────────── search
def test_search_returns_ranked_passages(client):
    r = client.post("/search", json={"query": "vendor delays in production", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["ticker"] == "SUPP"
    assert [p["rank"] for p in body["results"]] == [1, 2]
    assert body["took_ms"] >= 0


def test_all_three_methods_are_reachable(client):
    for method in ("hybrid", "bm25", "dense"):
        r = client.post("/search", json={"query": "hackers customer information",
                                         "k": 2, "method": method})
        assert r.status_code == 200, method
        assert r.json()["method"] == method
        assert r.json()["results"][0]["ticker"] == "CYBR", method


def test_every_passage_carries_provenance(client):
    """
    A result a caller cannot trace back to a filing is not a result. CIK,
    ticker, year and chunk id are the fields that make it checkable.
    """
    r = client.post("/search", json={"query": "interest rates debt service"})
    for p in r.json()["results"]:
        assert p["cik"] and p["ticker"] and p["filing_year"] and p["chunk_id"]


@pytest.mark.parametrize("payload", [
    {"query": "ab"},                        # under the length floor
    {"query": "valid query", "k": 0},       # k below range
    {"query": "valid query", "k": 500},     # k above range
    {"query": "valid query", "method": "magic"},
    {},                                     # missing required field
])
def test_bad_requests_are_rejected_before_any_work(client, payload):
    """
    422 from validation, not 500 from a handler. Validation at the boundary is
    the reason to define request models at all — without it, `k=500` would
    reach the index and `method="magic"` would fall through to a branch that
    does not exist.
    """
    assert client.post("/search", json=payload).status_code == 422


# ────────────────────────────────────────────────────────────────── answer
def test_answer_is_grounded_in_its_citations(client):
    r = client.post("/answer", json={"query": "hackers breach customer information"})
    assert r.status_code == 200
    body = r.json()
    assert not body["abstained"]
    joined = " ".join(c["text"] for c in body["citations"])
    for sentence in body["answer"].split(". "):
        assert sentence.strip(". ") in joined


def test_abstention_is_a_200_not_an_error(client):
    """
    A refusal is a successful response. The request was well-formed and the
    service did its job — returning 404 would tell a caller to retry, which is
    exactly wrong.
    """
    r = client.post("/answer", json={"query": "melting point of tungsten alloys"})
    assert r.status_code == 200
    body = r.json()
    assert body["abstained"] is True
    assert body["answer"] == ""


def test_confidence_is_exposed_to_the_caller(client):
    """A client that cannot see the margin cannot tell a confident hit from a
    marginal one, which defeats the abstention work."""
    hit = client.post("/answer", json={"query": "vendor delays in production"}).json()
    miss = client.post("/answer", json={"query": "offside rule in football"}).json()
    assert hit["confidence"] > miss["confidence"]
    assert 0.0 <= miss["confidence"] <= 1.0


# ─────────────────────────────────────────────────────────────────── score
def test_score_returns_frequencies_and_the_null_result_caveat(client):
    text = ("The Company faces adverse risk of loss. " * 30)
    r = client.post("/score", json={"text": text})
    assert r.status_code == 200
    body = r.json()
    assert body["n_words"] > 0
    assert 0.0 <= body["lm_negative"] <= 1.0
    # The endpoint must ship the stage-4 null result with the numbers, or a
    # caller will build a risk score on a feature set measured not to carry one.
    assert "no incremental" in body["note"].lower()


def test_score_rejects_text_too_short_to_be_meaningful(client):
    assert client.post("/score", json={"text": "too short"}).status_code == 422


# ────────────────────────────────────────────────────────────────── metrics
def test_metrics_report_percentiles_per_route(client):
    for _ in range(5):
        client.post("/search", json={"query": "vendor delays in production"})
    client.post("/answer", json={"query": "hackers breach customer information"})

    m = client.get("/metrics").json()
    assert m["search"]["n"] == 5
    assert m["answer"]["n"] == 1
    assert m["search"]["p95"] >= m["search"]["p50"]
    assert m["search"]["max"] >= m["search"]["p99"]
