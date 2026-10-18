"""
Stage 7 — the retrieval and answering stack behind an HTTP API.

WHY THIS STAGE EXISTS AT ALL
----------------------------
Everything up to here runs in a notebook, where a 5 ms median latency is a
number in a cell. That number tells you almost nothing about what happens when
ten people query at once, because the interesting costs only appear under
concurrency: the model is CPU-bound and holds a lock, the index is 70 MB of
float32 that must not be reloaded per request, and an async framework will
happily accept a thousand connections it cannot serve.

So the point of this stage is not "wrap it in FastAPI". It is to measure the
things a notebook cannot show, and to have a defensible answer when someone
asks how it behaves in production.

THE THREE DECISIONS WORTH DEFENDING
-----------------------------------
1. The index loads ONCE, at startup, into module state. Loading per request
   would add ~2 s and 70 MB of churn to every call. This is what `lifespan`
   is for.

2. Every handler is `async def`, but the work inside is CPU-bound — embedding
   a query is a matrix multiply, not IO. Running it directly on the event loop
   would block every other request for its duration, which turns a 5 ms median
   into a queue. So the work is pushed to a thread pool. That is the single
   most common mistake in FastAPI services and the one worth being able to
   explain.

3. The service reports its own latency percentiles at `/metrics`. A p50 is
   almost useless on its own — the p95 is what a user notices and what an SLA
   is written against.

Run:  uvicorn api:app --host 127.0.0.1 --port 8000
      python loadtest.py            # measure it under concurrent load
Docs: http://127.0.0.1:8000/docs    # generated from the type hints
"""
from __future__ import annotations

import json
import pickle
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from config import PROCESSED
from rag import DEFAULT_ABSTAIN_AT, RagPipeline
from retrieval import reciprocal_rank_fusion

# ═══════════════════════════════════════════════════════ request/response
# Pydantic models are not decoration. They validate every request before a
# handler runs, and FastAPI generates the OpenAPI schema and the /docs page
# from them, so the contract cannot drift from the code.


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=50)
    method: Literal["hybrid", "bm25", "dense"] = "hybrid"


class Passage(BaseModel):
    rank: int
    chunk_id: str
    cik: int
    ticker: str
    filing_year: int
    text: str


class SearchResponse(BaseModel):
    query: str
    method: str
    results: list[Passage]
    took_ms: float


class AnswerRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    mode: Literal["extractive", "generative"] = "extractive"


class AnswerResponse(BaseModel):
    query: str
    answer: str
    abstained: bool
    # Confidence is exposed deliberately. A client that only sees the answer
    # cannot tell a confident hit from a marginal one, and the whole point of
    # the abstention work is that the caller should be able to see the margin.
    confidence: float
    citations: list[Passage]
    took_ms: float


class ScoreRequest(BaseModel):
    text: str = Field(min_length=200, max_length=200_000)


class ScoreResponse(BaseModel):
    n_words: int
    lm_negative: float
    lm_positive: float
    lm_uncertainty: float
    lm_litigious: float
    lm_constraining: float
    note: str
    took_ms: float


# ══════════════════════════════════════════════════════════════════ state
class State:
    pipeline: RagPipeline | None = None
    lm: dict | None = None
    meta: dict = {}
    latencies: dict[str, deque] = {}


state = State()


def _record(route: str, ms: float) -> None:
    state.latencies.setdefault(route, deque(maxlen=5000)).append(ms)


def load_state() -> None:
    """
    Load the index into memory once.

    70 MB of float32 embeddings, a pickled BM25 index and a 250 MB embedding
    model. Doing this per request would dominate every measurement and is the
    reason `lifespan` exists.
    """
    chunks = pd.read_parquet(PROCESSED / "chunks.parquet")
    emb = np.load(PROCESSED / "chunk_embeddings.npy")
    meta = json.loads((PROCESSED / "index_meta.json").read_text())
    with open(PROCESSED / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer(meta["model"], device=device)

    def encode(texts):
        return st.encode(list(texts), convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)

    state.pipeline = RagPipeline(chunks, emb, bm25, encode)
    state.meta = {**meta, "device": device, "n_chunks": int(len(chunks)),
                  "n_firms": int(chunks.cik.nunique()),
                  "abstain_at": DEFAULT_ABSTAIN_AT}

    from text_features import load_lm
    state.lm = load_lm()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if state.pipeline is None:          # tests inject their own
        load_state()
    yield


app = FastAPI(
    title="EDGAR risk-factor search",
    description=("Semantic and lexical search over 10-K Item 1A passages, with "
                 "grounded answers that cite their source and abstain when the "
                 "corpus cannot answer."),
    version="1.0.0",
    lifespan=lifespan,
)


def _pipeline() -> RagPipeline:
    if state.pipeline is None:
        raise HTTPException(503, "index not loaded")
    return state.pipeline


def _passages(idx, scores=None) -> list[Passage]:
    p = _pipeline()
    out = []
    for rank, i in enumerate(idx, start=1):
        r = p.chunks.iloc[int(i)]
        out.append(Passage(rank=rank, chunk_id=str(r["chunk_id"]),
                           cik=int(r["cik"]), ticker=str(r["ticker"]),
                           filing_year=int(r["filing_year"]),
                           text=str(r["text"])))
    return out


# ══════════════════════════════════════════════════════════════ endpoints
@app.get("/health")
async def health():
    return {"status": "ok" if state.pipeline is not None else "loading",
            **state.meta}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    Retrieve passages. `method` exposes all three so the stage-5 benchmark is
    reproducible against the live service rather than only in a notebook.
    """
    t0 = time.perf_counter()
    p = _pipeline()

    def work():
        if req.method == "bm25":
            return [int(i) for i in p.bm25.top_k(req.query, req.k)]
        qvec = p.encode([req.query])[0]
        dense = list(np.argsort(-(p.emb @ qvec))[:max(req.k, 20)])
        if req.method == "dense":
            return [int(i) for i in dense[:req.k]]
        lex = list(p.bm25.top_k(req.query, max(req.k, 20)))
        return [int(i) for i in reciprocal_rank_fusion([lex, dense], top_n=req.k)]

    # CPU-bound: a matrix multiply against 47,000 vectors plus a sparse scan.
    # Left on the event loop it would block every other request for its whole
    # duration, and the p95 under load would be a queue length rather than a
    # latency.
    idx = await run_in_threadpool(work)
    ms = (time.perf_counter() - t0) * 1000
    _record("search", ms)
    return SearchResponse(query=req.query, method=req.method,
                          results=_passages(idx), took_ms=round(ms, 2))


@app.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest):
    """
    A grounded answer, or an explicit refusal.

    `abstained: true` with an empty answer is a SUCCESSFUL response, not an
    error — the corpus genuinely could not answer. Returning 404 here would be
    wrong: the request was well-formed and the service did its job.
    """
    t0 = time.perf_counter()
    p = _pipeline()
    a = await run_in_threadpool(lambda: p.answer(req.query, mode=req.mode, k=req.k))
    ms = (time.perf_counter() - t0) * 1000
    _record("answer", ms)

    cites = [Passage(rank=n, chunk_id=c.chunk_id, cik=c.cik, ticker=c.ticker,
                     filing_year=c.filing_year, text=c.text)
             for n, c in enumerate(a.citations, start=1)]
    return AnswerResponse(query=a.query, answer=a.text, abstained=a.abstained,
                          confidence=round(a.confidence, 4), citations=cites,
                          took_ms=round(ms, 2))


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    """
    Loughran-McDonald sentiment profile for arbitrary filing text.

    The honest note in the response body is not boilerplate. Stage 4 measured
    whether these features carry incremental credit signal over accounting
    ratios and found they do not (delta AUC -0.0022, p 0.313). Shipping the
    endpoint without saying so would invite a caller to build on a null result.
    """
    t0 = time.perf_counter()
    if state.lm is None:
        raise HTTPException(503, "lexicon not loaded")

    from text_features import count_categories
    counts = await run_in_threadpool(lambda: count_categories(req.text, state.lm))
    ms = (time.perf_counter() - t0) * 1000
    _record("score", ms)

    return ScoreResponse(
        n_words=int(counts.get("n_words", 0)),
        lm_negative=float(counts.get("lm_negative", 0.0)),
        lm_positive=float(counts.get("lm_positive", 0.0)),
        lm_uncertainty=float(counts.get("lm_uncertainty", 0.0)),
        lm_litigious=float(counts.get("lm_litigious", 0.0)),
        lm_constraining=float(counts.get("lm_constraining", 0.0)),
        note=("Frequencies only. Stage 4 found these carry no incremental "
              "credit signal over accounting ratios (dAUC -0.0022, p=0.313); "
              "do not use as a standalone risk score."),
        took_ms=round(ms, 2))


@app.get("/metrics")
async def metrics():
    """
    Latency percentiles per route, from this process's own history.

    p50 alone is close to useless — it describes the request nobody complains
    about. p95 is what a user notices and what an SLA is written against, so
    both are reported and so is the sample count behind them.
    """
    out = {}
    for route, vals in state.latencies.items():
        if not vals:
            continue
        a = np.array(vals)
        out[route] = {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), 2),
                      "p95": round(float(np.percentile(a, 95)), 2),
                      "p99": round(float(np.percentile(a, 99)), 2),
                      "max": round(float(a.max()), 2)}
    return out or {"note": "no requests served yet"}
