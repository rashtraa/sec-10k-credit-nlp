"""
Regression suite for stage 5 — chunking, BM25, dense search, fusion, metrics.

The same discipline as `test_sections.py`: every test encodes a way this
layer could be silently wrong and still look fine in a notebook.

The most important two:

  test_bm25_matches_reference    the fast sparse BM25 is arithmetically
                                 identical to the textbook nested-dict
                                 version it replaced

  test_headings_never_indexed    the evaluation queries are heading text, so
                                 if a heading leaked into the passage body
                                 BM25 would win by string matching and the
                                 whole benchmark would be meaningless

Run:  pytest test_retrieval.py -v
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from chunking import MAX_CHARS, MIN_CHARS, chunk_section, heading_queries
from retrieval import (BM25, build_relevance_sets, evaluate,
                       reciprocal_rank_fusion, tokenize)

# ══════════════════════════════════════════════════════════ fixtures
CORPUS = [
    "the supply chain may be disrupted by vendor failure and shortage",
    "our vendors may fail to deliver critical components on schedule",
    "interest rate increases could raise the cost of our borrowings",
    "cybersecurity incidents may compromise customer data and systems",
    "the supply chain risk is material and vendor concentration is high",
    "regulatory change in the european union could increase compliance cost",
]


def _reference_bm25(corpus_tokens, query, k1=1.5, b=0.75):
    """The nested-dict implementation, kept only as a correctness oracle."""
    N = len(corpus_tokens)
    doc_len = [len(d) for d in corpus_tokens]
    avgdl = sum(doc_len) / N
    tf = {}
    for i, doc in enumerate(corpus_tokens):
        for t in doc:
            tf.setdefault(t, {})
            tf[t][i] = tf[t].get(i, 0) + 1
    scores = [0.0] * N
    for t in tokenize(query):
        post = tf.get(t)
        if not post:
            continue
        idf = np.log(1.0 + (N - len(post) + 0.5) / (len(post) + 0.5))
        for i, f in post.items():
            denom = f + k1 * (1 - b + b * doc_len[i] / avgdl)
            scores[i] += idf * (f * (k1 + 1)) / denom
    return np.array(scores, dtype=np.float64)


# ══════════════════════════════════════════════════════════ BM25
def test_bm25_matches_reference():
    """
    The sparse rewrite exists for memory, not for different answers. Nested
    dicts needed several GB on 200k passages; this must give the same numbers.
    """
    toks = [tokenize(d) for d in CORPUS]
    bm = BM25(toks)
    for query in ["supply chain vendor",
                  "interest rate",
                  "cybersecurity customer data",
                  "vendor vendor vendor",           # repeated terms
                  "words that appear nowhere"]:
        got = bm.score(query).astype(np.float64)
        want = _reference_bm25(toks, query)
        assert np.allclose(got, want, atol=1e-4), query


def test_bm25_ranks_the_obvious_answer_first():
    bm = BM25([tokenize(d) for d in CORPUS])
    assert bm.top_k("cybersecurity incidents customer data", 3)[0] == 3
    assert bm.top_k("european union compliance", 3)[0] == 5


def test_bm25_unknown_query_scores_zero():
    """An out-of-vocabulary query must return zeros, not crash or rank."""
    bm = BM25([tokenize(d) for d in CORPUS])
    s = bm.score("zzzqqq nonexistentword")
    assert s.shape == (len(CORPUS),)
    assert float(s.sum()) == 0.0


def test_bm25_length_normalisation_bites():
    """
    A term buried in a long document must score below the same term in a
    short one. This is the `b` parameter doing its job; if length
    normalisation were dropped, verbose filings would win every query.
    """
    short = "vendor concentration risk"
    long = short + " " + " ".join(f"filler{i}" for i in range(400))
    bm = BM25([tokenize(short), tokenize(long)])
    s = bm.score("vendor concentration")
    assert s[0] > s[1]


def test_bm25_saturation():
    """
    Ten occurrences must be worth less than ten times one occurrence,
    otherwise keyword stuffing would beat relevance.
    """
    one = "covenant breach"
    ten = " ".join(["covenant"] * 10) + " breach"
    bm = BM25([tokenize(one), tokenize(ten)])
    s = bm.score("covenant")
    assert s[1] > s[0]
    assert s[1] < 10 * s[0]


# ══════════════════════════════════════════════════════════ fusion
def test_rrf_prefers_agreement():
    """A document both systems rank highly must beat one only one system likes."""
    a = [1, 2, 3, 4]
    b = [2, 9, 8, 7]
    fused = reciprocal_rank_fusion([a, b], top_n=4)
    assert fused[0] == 2                      # ranked 2nd and 1st


def test_rrf_blocks_one_confident_but_lonely_system():
    """
    The reason RRF is used instead of score averaging: a document one system
    is very confident about, but the other never returns at all, must not
    beat a document both systems agree is decent.
    """
    a = [99, 4, 5]          # 99 ranked first by a, absent from b
    b = [4, 5, 6]           # 4 is 2nd in a, 1st in b
    fused = reciprocal_rank_fusion([a, b], top_n=3)
    assert fused[0] == 4
    assert fused.index(99) > 0


def test_rrf_handles_disjoint_lists():
    fused = reciprocal_rank_fusion([[1, 2], [3, 4]], top_n=4)
    assert set(fused) == {1, 2, 3, 4}
    assert fused[0] in (1, 3)                 # the two rank-1 documents tie


# ══════════════════════════════════════════════════════════ metrics
def test_evaluate_known_values():
    rankings = [[7, 1, 2], [1, 7, 2], [1, 2, 3]]   # gold 7 at rank 1, 2, miss
    r = evaluate(rankings, [7, 7, 7], "t")
    assert r.recall_at_1 == pytest.approx(1 / 3)
    assert r.recall_at_5 == pytest.approx(2 / 3)
    assert r.mrr == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_evaluate_accepts_relevance_sets():
    """Any member of the set counts, and MRR uses the FIRST one found."""
    rankings = [[4, 9, 1]]
    assert evaluate(rankings, [{9, 1}], "t").mrr == pytest.approx(0.5)
    assert evaluate(rankings, [9], "t").mrr == pytest.approx(0.5)
    assert evaluate(rankings, [1], "t").mrr == pytest.approx(1 / 3)


def test_recall_is_monotonic_and_mrr_bounded():
    rng = np.random.default_rng(0)
    rankings = [list(rng.permutation(50)[:10]) for _ in range(200)]
    gold = list(rng.integers(0, 50, 200))
    r = evaluate(rankings, gold, "t")
    assert r.recall_at_1 <= r.recall_at_5 <= r.recall_at_10
    assert 0.0 <= r.mrr <= r.recall_at_10 + 1e-9


# ══════════════════════════════════════════════════════════ relevance sets
def test_relevance_sets_group_near_duplicates():
    """
    Boilerplate reused across firms must be treated as interchangeable,
    otherwise Recall measures corpus duplication rather than retrieval.
    """
    base = ("we depend on a limited number of suppliers for critical "
            "components and any interruption in supply would materially "
            "harm our results of operations and financial condition")
    texts = [base, base.replace("harm", "damage"), base + " in future periods",
             "interest rate increases could raise the cost of our borrowings",
             "cybersecurity incidents may compromise customer data"]
    sets = build_relevance_sets([0], texts, threshold=0.85)
    assert {0, 1, 2} <= sets[0]
    assert 3 not in sets[0] and 4 not in sets[0]


def test_relevance_set_of_unique_passage_is_itself():
    texts = ["alpha beta gamma delta epsilon zeta eta theta iota kappa",
             "completely different words about shipping vessels and ports",
             "another unrelated passage regarding pension plan obligations"]
    sets = build_relevance_sets([0], texts, threshold=0.85)
    assert sets[0] == {0}


def test_relevance_sets_align_with_query_order():
    """A batching bug here would silently score every query against the
    wrong gold set, and the metrics would still look plausible."""
    rng = np.random.default_rng(3)
    pool = [f"term{i:03d}" for i in range(120)]
    texts = [" ".join(rng.choice(pool, size=10, replace=False)) for _ in range(40)]
    gold = [3, 17, 39, 0]
    sets = build_relevance_sets(gold, texts, threshold=0.85, batch=2)
    assert sets == [{3}, {17}, {39}, {0}]


# ══════════════════════════════════════════════════════════ chunking
SECTION = """Item 1A. Risk Factors

We depend on a limited number of suppliers.

Our manufacturing operations rely on a small number of vendors for critical
components. If any of these vendors were to fail to deliver on schedule, or
to cease operations entirely, we could experience material delays in
production that would harm our results of operations. We do not have
long-term contracts with most of these vendors and they may raise prices at
short notice, which would compress our margins in ways we cannot fully pass
through to customers in the current competitive environment we operate in.

Cybersecurity incidents could disrupt our business.

We rely on information systems to process transactions and to store
confidential customer information. Despite our security measures, our systems
may be vulnerable to attacks by hackers or breaches due to employee error or
malfeasance. Any such breach could compromise our networks and the
information stored there could be accessed, publicly disclosed, lost or
stolen, exposing us to litigation and regulatory penalties that could be
significant relative to our earnings in any given reporting period.
"""


def test_headings_never_indexed():
    """
    The control the whole benchmark rests on. Queries ARE heading text; if
    the heading were also inside the indexed body, BM25 would win by finding
    a repeated string and the comparison would measure nothing.
    """
    chunks = chunk_section(SECTION, 1234, "TEST", 2023)
    headed = [c for c in chunks if c.heading]
    assert headed, "no headings detected — the fixture or the detector broke"
    for c in headed:
        assert c.heading not in c.text


def test_short_body_heading_is_not_swallowed():
    """
    The regression that motivated `_maybe_heading`.

    A heading followed by an explanation just under BODY_MIN failed the
    strict heading test, so the merge loop absorbed it into the PREVIOUS
    chunk — leaving heading text inside an indexed passage. Since queries are
    heading text, BM25 could then win by literal string match and the whole
    lexical-vs-semantic comparison would be measuring nothing.

    Caught on a synthetic corpus where a chunk came back 661 characters long
    containing two topics and one buried heading.
    """
    short_body = ("Our exposure to data privacy arises in the ordinary course "
                  "of business. Adverse developments could reduce demand for "
                  "our products, increase our costs, or require capital "
                  "expenditure we may be unable to fund on acceptable terms.")
    assert len(short_body) < 300, "fixture must sit under the OLD 300-char BODY_MIN"

    section = (
        "We depend on a limited number of suppliers.\n\n"
        "Our manufacturing operations rely on a small number of vendors for "
        "critical components. If any of these vendors were to fail to deliver "
        "on schedule, or to cease operations entirely, we could experience "
        "material delays in production that would harm our results.\n\n"
        "Risk relating to data privacy could affect our results.\n\n"
        + short_body)

    chunks = chunk_section(section, 1, "T", 2023)
    for c in chunks:
        assert "Risk relating to data privacy" not in c.text, \
            f"heading leaked into an indexed passage: {c.text[:120]!r}"


def test_no_detected_heading_appears_in_any_body():
    """
    Corpus-level version of the control: across many synthetic filings, no
    heading text may appear inside ANY passage — not just its own.
    """
    rng = np.random.default_rng(5)
    topics = [(f"Risk relating to {w} could affect our results.",
               f"Our exposure to {w} arises from the ordinary course of business. "
               f"Adverse developments in {w} could reduce demand for our products, "
               f"increase our costs, or require additional capital expenditure "
               f"that we may be unable to fund on acceptable terms.")
              for w in ["foreign currency", "regulatory change", "data privacy",
                        "commodity prices", "labour availability", "tax legislation"]]

    all_chunks = []
    for f in range(25):
        picks = rng.choice(len(topics), size=4, replace=False)
        section = "\n\n".join(f"{topics[p][0]}\n\n{topics[p][1]}" for p in picks)
        all_chunks += chunk_section(section, 1000 + f, f"T{f}", 2023)

    headings = {c.heading for c in all_chunks if c.heading}
    assert headings
    for c in all_chunks:
        for h in headings:
            assert h not in c.text


def test_runon_block_is_resplit_into_paragraphs():
    """
    Real 10-K text, second failure mode. The HTML-to-text converter emits one
    newline per block element, so a filer whose paragraphs are unspaced
    `<div>`s arrives as a single enormous "paragraph" with no blank lines.

    With no paragraph boundaries there are no headings to detect, the section
    is cut into fixed-size pieces, and headings end up buried inside indexed
    passages. Found on the real corpus: 10 of the 200 longest headings were
    sitting inside passage bodies.
    """
    topic = ("Our manufacturing operations rely on a small number of vendors "
             "for critical components. If any of these vendors were to fail to "
             "deliver on schedule, or to cease operations entirely, we could "
             "experience material delays in production that would harm our "
             "results of operations and our financial condition.")
    # single newlines only — no blank lines anywhere
    runon = "\n".join([
        "We depend on a limited number of suppliers.", topic,
        "Cybersecurity incidents could disrupt our business.", topic,
        "Increases in interest rates could raise our borrowing costs.", topic,
        "We face intense competition in all of our markets.", topic,
    ])
    assert "\n\n" not in runon

    chunks = chunk_section(runon, 1, "T", 2023)
    assert [c for c in chunks if c.heading], "no heading recovered from run-on"
    for c in chunks:
        assert "We depend on a limited number of suppliers." not in c.text


def test_wrapped_lines_are_rejoined_not_split():
    """
    The other side of the same rule: source HTML often hard-wraps text, and
    those newlines are NOT paragraph breaks. Splitting on them would shred
    sentences into fragments too short to index.
    """
    from chunking import _relines

    wrapped = ("Our exposure to foreign currency arises from the ordinary\n"
               "course of business and from our international operations,\n"
               "which we cannot fully hedge on acceptable terms.")
    out = _relines(wrapped)
    assert len(out) == 1
    assert "ordinary course of business" in out[0]


def test_strip_leaked_headings_removes_only_the_heading():
    pd = pytest.importorskip("pandas")
    from build_index import strip_leaked_headings

    heading = "We depend on a limited number of suppliers."
    body = ("Our manufacturing operations rely on a small number of vendors "
            "for critical components and any interruption would materially "
            "harm our results of operations in the affected periods, which "
            "we may be unable to mitigate through alternative sourcing.")
    df = pd.DataFrame([
        {"heading": heading, "text": body, "n_chars": len(body)},
        {"heading": "", "text": f"{heading} {body}", "n_chars": 0},
    ])
    out = strip_leaked_headings(df)
    assert len(out) == 2
    for t in out["text"]:
        assert heading not in t
        assert "alternative sourcing" in t          # explanation survives


def test_strip_leaked_headings_removes_welded_heading():
    """
    The residual the fragment pass cannot see, and the audit can.

    Filers write `<p><b>Supply chain risk</b> Our operations rely...</p>`,
    which reaches us as ONE sentence with the heading welded to the front. It
    is not a whole fragment, so fragment matching misses it; raw substring
    matching finds it. On the real corpus 9 of the 300 longest headings
    survived pass one this way.
    """
    pd = pytest.importorskip("pandas")
    from build_index import strip_leaked_headings

    heading = "Supply chain and vendor concentration risk"   # no full stop
    tail = ("Our operations rely on a small number of vendors for critical "
            "components and any interruption would materially harm our "
            "results of operations, which we may be unable to mitigate "
            "through alternative sourcing arrangements in the near term.")
    df = pd.DataFrame([
        {"heading": heading, "text": tail, "n_chars": len(tail)},
        {"heading": "", "text": f"{heading} {tail}", "n_chars": 0},
    ])
    out = strip_leaked_headings(df)
    for t in out["text"]:
        assert heading.lower() not in t.lower()
        assert "alternative sourcing" in t


def test_strip_leaked_headings_leaves_clean_corpus_untouched():
    pd = pytest.importorskip("pandas")
    from build_index import strip_leaked_headings

    body = ("Our exposure to commodity prices arises in the ordinary course "
            "of business and adverse movements could increase our costs "
            "materially in any given reporting period, which we may be "
            "unable to pass through to our customers.")
    df = pd.DataFrame([{"heading": "Commodity prices could raise our costs.",
                        "text": body, "n_chars": len(body)}])
    out = strip_leaked_headings(df)
    assert out["text"].iloc[0] == body


def test_chunk_size_bounds():
    chunks = chunk_section(SECTION, 1234, "TEST", 2023)
    assert chunks
    for c in chunks:
        assert MIN_CHARS <= c.n_chars
        assert c.n_chars <= MAX_CHARS
        assert c.n_chars == len(c.text)


def test_no_chunk_exceeds_the_model_window():
    """
    Filing prose produces sentences over 1,000 characters, and a
    sentence-boundary splitter cannot cut them. Left alone they overflow the
    embedding model's 256-token window and are silently truncated — 2.1% of
    real passages, measured — while BM25 still indexes every word. The two
    systems would not be seeing the same text.
    """
    monster = "The Company " + ", ".join(f"subsidiary {i}" for i in range(400)) + "."
    assert len(monster) > 3 * MAX_CHARS
    section = f"A risk relating to our corporate structure.\n\n{monster}"
    for c in chunk_section(section, 1, "T", 2023):
        assert c.n_chars <= MAX_CHARS


def test_page_furniture_never_becomes_a_query():
    """
    Running headers are short lines followed by longer ones, which is exactly
    the heading pattern — so `12 Table of Contents` was being detected as a
    risk heading and turned into an evaluation query. Retrieving the passage
    that follows a page break is a meaningless task. Found on the real corpus:
    the leak repair's first two examples were both `NN table of contents`.
    """
    body = ("Our exposure to commodity prices arises in the ordinary course "
            "of business and adverse movements could increase our costs "
            "materially in any reporting period, which we may be unable to "
            "pass through to customers on acceptable terms.")
    section = "\n\n".join([
        "Commodity price movements could increase our costs.", body,
        "12", "Table of Contents", "PART I", "Item 1A. Risk Factors", body,
    ])
    chunks = chunk_section(section, 1, "T", 2023)
    headings = {c.heading for c in chunks if c.heading}
    assert "Commodity price movements could increase our costs." in headings
    for junk in ("12", "Table of Contents", "PART I", "Item 1A. Risk Factors"):
        assert junk not in headings
        for c in chunks:
            assert junk.lower() not in c.text.lower()


def test_heading_fragments_rejected():
    """One- or two-word labels are section dividers, not risk statements."""
    from chunking import _too_few_words
    for junk in ("Risks", "General", "Other Matters", "Operational Risks"):
        assert _too_few_words(junk), junk
    assert not _too_few_words("We depend on a limited number of suppliers.")


class _StubTokenizer:
    """Whitespace tokenizer with real offsets, standing in for the model's."""

    def __call__(self, text, add_special_tokens=True, truncation=True,
                 return_offsets_mapping=False):
        many = not isinstance(text, str)
        texts = text if many else [text]
        ids, offs = [], []
        for t in texts:
            spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", t)]
            ids.append(list(range(len(spans))))
            offs.append(spans)
        out = {"input_ids": ids if many else ids[0]}
        if return_offsets_mapping:
            out["offset_mapping"] = offs if many else offs[0]
        return out


class _StubModel:
    max_seq_length = 62
    tokenizer = _StubTokenizer()


def test_split_to_token_limit_respects_the_window():
    """
    A character ceiling is an average, and filing prose is not average
    everywhere — tables, dollar figures and ALL-CAPS defined terms tokenise
    far more densely. 1.9% of real passages still hit the window at an
    850-character cap. The limit has to be enforced in tokens.
    """
    pd = pytest.importorskip("pandas")
    from build_index import split_to_token_limit

    long_text = " ".join(f"riskfactor{i:04d}" for i in range(400))
    df = pd.DataFrame([{"heading": "A real risk heading about our business.",
                        "text": long_text, "n_chars": len(long_text),
                        "chunk_id": "1_2023_0", "cik": 1,
                        "ticker": "T", "filing_year": 2023}])

    out = split_to_token_limit(df, _StubModel())
    limit = _StubModel.max_seq_length - 2

    assert len(out) > 1, "an over-long passage must be split"
    for t in out["text"]:
        n = len(_StubTokenizer()(t, return_offsets_mapping=False)["input_ids"])
        assert n <= limit, f"{n} tokens exceeds the {limit}-token window"

    # The heading belongs to the first piece only: a continuation is not what
    # the heading introduced, so it must not generate an evaluation query.
    assert (out["heading"] != "").sum() == 1
    assert out["chunk_id"].is_unique
    # No text is lost.
    assert "riskfactor0399" in " ".join(out["text"])


def test_split_to_token_limit_leaves_short_passages_alone():
    pd = pytest.importorskip("pandas")
    from build_index import split_to_token_limit

    text = "a short passage about supply chain risk " * 5
    df = pd.DataFrame([{"heading": "h", "text": text, "n_chars": len(text),
                        "chunk_id": "1_2023_0", "cik": 1, "ticker": "T",
                        "filing_year": 2023}])
    out = split_to_token_limit(df, _StubModel())
    assert len(out) == 1


def test_section_dividers_and_bullets_are_not_queries():
    """
    Both found by printing ten random queries off the real corpus.

    A divider names a GROUP of risks, so no single passage answers it and
    whatever a system returns is scored arbitrarily. A bullet fragment is a
    clause lifted out of a list, with no independent meaning. Roughly one in
    five queries was one or the other.
    """
    from chunking import is_usable_query

    for junk in ("Risks Related to Our Organization and Structure:",
                 "Risks Relating to Our Business",
                 "General Risk Factors",
                 "Certain Risks",
                 "\u2022the price, availability and consumer demand for alternative fuels;",
                 "- our ability to obtain additional financing on acceptable terms,"):
        assert not is_usable_query(junk), junk

    for real in ("It may be difficult to sell real estate timely, which may "
                 "limit our flexibility.",
                 "We have no current project that will generate revenues in "
                 "the near future.",
                 "If we experience loan losses in excess of our allowance, our "
                 "earnings will be adversely affected"):
        assert is_usable_query(real), real


def test_chunk_ids_unique_within_section():
    chunks = chunk_section(SECTION, 1234, "TEST", 2023)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_heading_queries_only_from_headed_chunks():
    chunks = chunk_section(SECTION, 1234, "TEST", 2023)
    qs = heading_queries(chunks)
    assert len(qs) == sum(1 for c in chunks if c.heading)
    ids = {c.chunk_id for c in chunks}
    assert all(q["relevant_chunk_id"] in ids and q["query"] for q in qs)


def test_no_chunks_from_empty_or_tiny_section():
    assert chunk_section("", 1, "T", 2023) == []
    assert chunk_section("Too short.", 1, "T", 2023) == []


# ══════════════════════════════════════════════════════════ dense
def test_dense_index_retrieves_itself():
    """
    Plumbing check with synthetic vectors — no model download required.
    A passage's own embedding must retrieve that passage at rank 1, and
    inner-product-on-normalised-vectors must equal cosine.
    """
    faiss = pytest.importorskip("faiss")
    from retrieval import DenseIndex

    rng = np.random.default_rng(7)
    emb = rng.normal(size=(500, 64)).astype("float32")
    idx = DenseIndex(emb)
    for i in (0, 123, 499):
        assert idx.top_k(emb[i], 5)[0] == i


def test_dense_index_finds_the_near_neighbour():
    pytest.importorskip("faiss")
    from retrieval import DenseIndex

    rng = np.random.default_rng(11)
    emb = rng.normal(size=(300, 32)).astype("float32")
    query = emb[42] + 0.01 * rng.normal(size=32).astype("float32")
    assert DenseIndex(emb).top_k(query, 5)[0] == 42
