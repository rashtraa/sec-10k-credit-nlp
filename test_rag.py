"""
Regression suite for stage 6 — grounding, attribution, abstention.

Every test here runs with a STUB ENCODER: a deterministic bag-of-words
projection standing in for the sentence-transformer. That is deliberate. These
tests check the pipeline's logic — does it cite what it quoted, does it refuse
when nothing matches, does it never emit a sentence that is not in a cited
passage — and none of that should depend on a 90 MB download, a network, or
which version of torch is installed. A test that fails for those reasons
teaches you nothing.

Run:  pytest test_rag.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from rag import (Answer, Citation, RagPipeline, grounded_verbatim,
                 grounding_overlap, split_sentences, unsupported_terms)
from retrieval import BM25, tokenize

pd = pytest.importorskip("pandas")


# ────────────────────────────────────────────────────────────── fixtures
PASSAGES = [
    ("SUPP", "Our manufacturing operations rely on a small number of vendors "
             "for critical components. If any vendor fails to deliver on "
             "schedule we could experience material delays in production. "
             "We do not have long-term contracts with most of these vendors."),
    ("CYBR", "We rely on information systems to process transactions and store "
             "confidential customer information. Our systems may be vulnerable "
             "to attacks by hackers or to breaches caused by employee error. "
             "Any breach could expose us to litigation and regulatory penalty."),
    ("RATE", "A substantial portion of our indebtedness bears interest at "
             "variable rates. If market interest rates increase our debt "
             "service obligations would rise materially. That would reduce "
             "cash available for capital expenditure."),
    ("COMP", "The markets in which we operate are highly competitive. Many "
             "competitors have greater financial and technical resources than "
             "we do. They may respond more quickly to new technologies."),
]


def _stub_encoder(vocab):
    """
    Deterministic hashing encoder: a text becomes an L2-normalised
    bag-of-vocabulary vector. Similar wording lands nearby, which is all the
    pipeline needs from an encoder, and it is reproducible to the bit.

    Tokens of three characters or fewer are ignored. Without that, an entirely
    off-topic question still scores well above zero purely on "is", "the" and
    "of" — which would make the abstention tests pass or fail on stopword
    frequency rather than on the logic they are meant to check.
    """
    index = {w: i for i, w in enumerate([v for v in vocab if len(v) > 3])}

    def encode(texts):
        out = np.zeros((len(texts), len(index)), dtype="float32")
        for r, t in enumerate(texts):
            for w in tokenize(t):
                if w in index:
                    out[r, index[w]] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)

    return encode


@pytest.fixture
def pipe():
    df = pd.DataFrame([
        {"chunk_id": tag, "cik": 100 + i, "ticker": tag, "filing_year": 2023,
         "text": text, "heading": "", "n_chars": len(text)}
        for i, (tag, text) in enumerate(PASSAGES)])

    vocab = sorted({w for _, t in PASSAGES for w in tokenize(t)})
    encode = _stub_encoder(vocab)
    emb = encode(df["text"].tolist())
    bm25 = BM25([tokenize(t) for t in df["text"]])
    # 0.15 suits the stub encoder's scale — a two-word query against a
    # 300-character passage scores ~0.20 under bag-of-words cosine. The
    # real encoder's scale is different, which is exactly why the real
    # threshold is chosen from the ROC curve rather than copied from here.
    return RagPipeline(df, emb, bm25, encode, abstain_at=0.15)


# ────────────────────────────────────────────────────────────── retrieval
def test_retrieves_the_right_passage(pipe):
    assert pipe.chunks.iloc[pipe.retrieve("vendor delays in production", k=3)[0]]["ticker"] == "SUPP"
    assert pipe.chunks.iloc[pipe.retrieve("hackers breach customer data", k=3)[0]]["ticker"] == "CYBR"


def test_confidence_is_bounded_and_ordered(pipe):
    """
    The abstention decision is a threshold on this number, so it has to mean
    the same thing across queries — bounded, and higher for a question the
    corpus can answer than for one it cannot.
    """
    on_topic = pipe.confidence("interest rates and debt service",
                               pipe.retrieve("interest rates and debt service"))
    off_topic = pipe.confidence("penguin migration in antarctica",
                                pipe.retrieve("penguin migration in antarctica"))
    assert -1.0 <= off_topic <= on_topic <= 1.0
    assert on_topic > off_topic


# ────────────────────────────────────────────────────────────── grounding
def test_extractive_answer_is_always_verbatim(pipe):
    """
    THE control this stage rests on. In extractive mode the answer is a span
    of a cited passage, so grounding must hold for every query — any failure
    is a pipeline bug, not a model limitation. An early version normalised
    whitespace on the answer but not the source and broke exactly this.
    """
    for query in ["vendor delays", "hackers and customer data",
                  "variable rate debt service", "competitors with more resources",
                  "long-term contracts with vendors"]:
        a = pipe.answer(query)
        assert not a.abstained, query
        assert grounded_verbatim(a), f"answer not verbatim in source: {query}"
        assert grounding_overlap(a) == pytest.approx(1.0)
        assert unsupported_terms(a) == []


def test_grounding_check_can_actually_fail():
    """
    A control that cannot fail proves nothing. This fabricates the failure the
    real check is looking for — a sentence that appears in no cited passage.
    """
    cite = Citation("X", 1, "X", 2023, 1.0,
                    "Our vendors may fail to deliver components on schedule.")
    honest = Answer("q", "Our vendors may fail to deliver components on schedule.",
                    [cite])
    invented = Answer("q", "The company expects a 40% decline in revenue next year.",
                      [cite])
    assert grounded_verbatim(honest)
    assert not grounded_verbatim(invented)
    assert grounding_overlap(invented) < 0.5
    assert "decline" in unsupported_terms(invented)


def test_citations_point_at_the_quoted_passage(pipe):
    """An answer whose citation is not where the text came from is worse than
    no citation, because a citation is what stops a reader checking."""
    a = pipe.answer("hackers breach confidential customer information")
    assert a.citations
    joined = " ".join(c.text for c in a.citations)
    for sent in split_sentences(a.text) or [a.text]:
        assert sent in joined


# ────────────────────────────────────────────────────────────── abstention
def test_abstains_on_an_unanswerable_question(pipe):
    a = pipe.answer("what is the melting point of tungsten")
    assert a.abstained
    assert a.text == ""


def test_abstention_respects_the_threshold(pipe):
    """The threshold is the knob; it must actually be the knob."""
    q = "competitors with greater financial resources"
    assert not pipe.answer(q).abstained
    pipe.abstain_at = 0.99
    assert pipe.answer(q).abstained
    pipe.abstain_at = 0.0
    assert not pipe.answer(q).abstained


def test_abstained_answer_is_trivially_grounded(pipe):
    """Refusing to answer cannot count as a grounding failure, or the metric
    would punish the behaviour it exists to encourage."""
    a = pipe.answer("what is the melting point of tungsten")
    assert grounded_verbatim(a)
    assert grounding_overlap(a) == 1.0


# ────────────────────────────────────────────────────────────── evaluation
def test_abstention_auc_separates_the_two_sets():
    from rag_eval import _auc
    answerable = np.array([0.9, 0.8, 0.7, 0.6])
    unanswerable = np.array([0.1, 0.2, 0.3])
    assert _auc(answerable, unanswerable) == pytest.approx(1.0)
    assert _auc(unanswerable, answerable) == pytest.approx(0.0)
    assert _auc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_threshold_sweep_is_monotonic():
    """
    Raising the bar can only reduce coverage and false answers. If this ever
    fails the sweep is computed wrong, and the table it produces is what a
    reader would use to pick the operating point.
    """
    from rag_eval import threshold_sweep
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "answerable": [True] * 200 + [False] * 200,
        "confidence": np.concatenate([rng.normal(0.6, 0.1, 200),
                                      rng.normal(0.3, 0.1, 200)]),
    })
    sw = threshold_sweep(df)
    assert sw["coverage (answerable answered)"].is_monotonic_decreasing
    assert sw["false answers (unanswerable answered)"].is_monotonic_decreasing


def test_evaluate_rag_end_to_end(pipe):
    from rag_eval import evaluate_rag

    answerable = pd.DataFrame([
        {"query": "vendor delays in production", "relevant_chunk_id": "SUPP"},
        {"query": "hackers breach customer information", "relevant_chunk_id": "CYBR"},
        {"query": "variable rate debt service obligations", "relevant_chunk_id": "RATE"},
    ])
    unanswerable = pd.DataFrame([
        {"query": "melting point of tungsten alloys", "relevant_chunk_id": "ZZZ"},
        {"query": "rules of the offside trap in football", "relevant_chunk_id": "ZZZ"},
    ])
    rep, rows = evaluate_rag(pipe, answerable, unanswerable, progress=False)

    assert rep.n_answerable == 3 and rep.n_unanswerable == 2
    assert rep.grounded_verbatim == pytest.approx(1.0)   # extractive guarantee
    assert rep.attribution_at_1 == pytest.approx(1.0)
    assert rep.answer_rate > rep.false_answer_rate
    assert 0.0 <= rep.abstention_auc <= 1.0
    assert len(rows) == 5


# ──────────────────────────────────────────── the two bugs found by running it
def test_grounded_verbatim_survives_a_sentence_join_that_does_not_round_trip():
    """
    The 1.5% "hallucination" in a mode that cannot hallucinate.

    `grounded_verbatim` used to recover the answer's sentences by re-splitting
    `answer.text`. Joining then splitting is not the identity: the splitter cuts
    on terminal punctuation followed by a capital, so a first sentence ending in
    an abbreviation ("...the U.S.") glued to a second starting lower-case came
    back as one string, which appears in no passage.

    Both sentences below ARE verbatim spans of their passages. The old
    implementation returned False. Anything below 1.0 in extractive mode is a
    measurement bug, and this is the shape of it.
    """
    src = ("Our operations are concentrated in the U.S. provided that "
           "regulatory approval is obtained we may expand abroad.")
    c = Citation(chunk_id="X1", cik=1, ticker="AAA", filing_year=2023,
                 score=1.0, text=src)
    a = Answer(query="q",
               text="Our operations are concentrated in the U.S. "
                    "provided that regulatory approval is obtained we may "
                    "expand abroad.",
               citations=[c],
               spans=[("Our operations are concentrated in the U.S.", "X1"),
                      ("provided that regulatory approval is obtained we may "
                       "expand abroad.", "X1")])
    assert grounded_verbatim(a) is True


def test_grounded_verbatim_rejects_a_span_stitched_across_two_citations():
    """
    Checking spans against their OWN passage, not against all citations
    concatenated. The concatenated version accepts a string that straddles the
    join — text that exists in the evaluation harness and in no filing.
    """
    c1 = Citation("A", 1, "AAA", 2023, 1.0, "The covenant limits dividends.")
    c2 = Citation("B", 2, "BBB", 2023, 1.0, "We repaid the revolver in full.")
    a = Answer(query="q", text="The covenant limits dividends.",
               citations=[c1, c2], spans=[("The covenant limits dividends.", "B")])
    assert grounded_verbatim(a) is False


def test_bigram_grounding_separates_quotation_from_recombination():
    """
    Why the unigram overlap came back 1.0000 for both modes and told us
    nothing: with three whole passages as the source, every content word of a
    short answer is present regardless of whether the answer is true.

    Both answers below use ONLY words from the source. The second reverses the
    meaning by re-pairing them. Unigram overlap scores them IDENTICALLY at a
    perfect 1.0000; bigram overlap separates them.

    Note what is and is not claimed. Bigram overlap does not collapse on a
    recombination — a re-ordering preserves most adjacent pairs, so the false
    answer here still scores around 0.86. The claim is only that it moves off
    1.0 where the unigram version cannot, which is enough to make the metric
    informative instead of saturated. Anything stronger would need an entailment
    model, and that is the honest next step rather than a number to quote now.
    """
    src = "Our indebtedness does restrict our ability to pay dividends."
    c = Citation("A", 1, "AAA", 2023, 1.0, src)
    faithful = Answer(query="q", text="Our indebtedness does restrict our "
                                      "ability to pay dividends.", citations=[c])
    recombined = Answer(query="q", text="Our ability to pay dividends does "
                                        "restrict our indebtedness.", citations=[c])

    assert grounding_overlap(faithful) == pytest.approx(1.0)
    assert grounding_overlap(recombined) == pytest.approx(1.0)   # blind
    assert grounding_overlap(faithful, n=2) == pytest.approx(1.0)
    assert grounding_overlap(recombined, n=2) < 1.0              # not blind
    # Never more permissive than the unigram version, on any answer.
    for a in (faithful, recombined):
        assert grounding_overlap(a, n=2) <= grounding_overlap(a) + 1e-9


def test_pick_threshold_respects_the_stated_budget():
    """
    The operating point is chosen by naming the cost, not by maximising a
    score. Whatever it returns must actually satisfy the constraint it was
    given — otherwise the number on the CV is not the number the system runs at.
    """
    from rag_eval import pick_threshold
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "answerable": [True] * 300 + [False] * 300,
        "confidence": np.concatenate([rng.normal(0.62, 0.10, 300),
                                      rng.normal(0.34, 0.10, 300)]),
    })
    got = pick_threshold(df, max_false_answer_rate=0.10)
    assert got["satisfiable"] is True
    assert got["false_answer_rate"] <= 0.10
    assert got["coverage"] > 0.0
    # A tighter budget can never buy more coverage.
    assert pick_threshold(df, 0.02)["coverage"] <= got["coverage"]


def test_retrieved_ids_are_recorded_even_when_abstaining(pipe):
    """
    Retrieval recall has to be measurable on questions the system refused,
    otherwise "we abstained" and "we never found it" are indistinguishable in
    the evaluation.
    """
    a = pipe.answer("melting point of tungsten alloys")
    assert a.abstained is True
    assert len(a.retrieved) > 0


def test_bigram_overlap_is_span_wise_not_concatenated():
    """
    Two artefacts of scoring joined text against joined sources, both removed
    by going span-wise.

    ANSWER SIDE — the pair straddling two extracted sentences was written by
    nobody. It is an artefact of the join, exists in no filing, and was counted
    as a miss. That is what held extractive bigram grounding at 0.9855 in a
    mode where it is 1.0 by construction.

    SOURCE SIDE — the pair straddling two concatenated passages is equally
    fictional and was counted as a hit, so a sentence stitched across two
    filings could pass as grounded.
    """
    p1 = ("The credit agreement contains a maximum leverage covenant tested "
          "quarterly by the administrative agent.")
    p2 = ("Separately the indenture governs the senior secured notes issued "
          "during the prior fiscal year.")
    s1, s2 = p1, p2
    c1 = Citation("A", 1, "AAA", 2023, 1.0, p1)
    c2 = Citation("B", 2, "BBB", 2023, 1.0, p2)

    # Answer side: both spans are verbatim, so this must be exactly 1.0.
    a = Answer(query="q", text=f"{s1} {s2}", citations=[c1, c2],
               spans=[(s1, "A"), (s2, "B")])
    assert grounding_overlap(a, n=2) == pytest.approx(1.0)

    # Source side: the same text with no spans falls back to concatenation and
    # cannot see that the two halves came from different filings.
    naive = Answer(query="q", text=f"{s1} {s2}", citations=[c1, c2])
    # The span-wise path is the strict one; the fallback is at least as
    # permissive. This documents the gap rather than pretending it is closed.
    assert grounding_overlap(naive, n=2) >= grounding_overlap(a, n=2) - 1e-9

    # And a span attributed to the wrong passage is caught, where the
    # concatenated version would have accepted it.
    mis = Answer(query="q", text=s1, citations=[c1, c2], spans=[(s1, "B")])
    assert grounding_overlap(mis, n=2) < 0.2


# ─────────────────────────────────────────────── generative mode, without the model
class _StubTokenizer:
    """Whitespace tokenizer with the one method the fitting code calls."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


@pytest.fixture
def stub_generator(monkeypatch):
    """
    Exercise `_generative` without a 250 MB download.

    THIS FIXTURE EXISTS BECAUSE THE PATH HAD NO COVERAGE AND SHIPPED A
    NameError. Every generative test previously required `flan-t5-base`, so in
    practice none ran, and `text = gen(prompt)` was deleted by an edit anchored
    on that line. The suite stayed green and the notebook crashed. An untested
    branch is an untested branch whether the excuse is a model download or
    anything else.

    Returns the list of prompts the generator was handed, so a test can assert
    on what the model was actually shown.
    """
    seen = []

    def fake_gen(prompt):
        seen.append(prompt)
        return "not stated" if "tungsten" in prompt else "Our vendors may fail to deliver."

    monkeypatch.setattr("rag._GEN", (fake_gen, _StubTokenizer()))
    return seen


def test_generative_mode_runs_and_cites(pipe, stub_generator):
    """The regression for the NameError: this path must simply execute."""
    pipe.abstain_at = 0.0
    a = pipe.answer("vendor delays in production", mode="generative")
    assert a.mode == "generative"
    assert a.text
    assert a.citations
    assert a.retrieved
    assert len(stub_generator) == 1


def test_generative_abstains_on_not_stated(pipe, stub_generator):
    """The prompt gives the model an explicit way out; taking it must count as
    an abstention, not as an answer whose text happens to read 'not stated'."""
    pipe.abstain_at = 0.0
    a = pipe.answer("melting point of tungsten alloys", mode="generative")
    assert a.abstained is True


def test_generative_context_is_fitted_to_the_window(pipe, stub_generator,
                                                    monkeypatch):
    """
    The model must never be graded on text it did not read.

    The first real run printed `546 > 512`: the encoder window silently dropped
    the tail of the prompt — the third retrieved passage — while the evaluation
    went on crediting that passage as a citation. With a tiny window here, the
    prompt must stay inside it and the citation count must match what fitted.
    """
    monkeypatch.setattr("rag._GEN_MAX_TOKENS", 120)
    pipe.abstain_at = 0.0
    a = pipe.answer("vendor delays in production", mode="generative")

    prompt = stub_generator[0]
    assert len(prompt.split()) <= 120
    # Every cited passage appears in the prompt the model was handed.
    for c in a.citations:
        assert c.text[:40] in prompt
    assert len(a.citations) <= 3


def test_generative_citations_track_the_window_size(pipe, stub_generator,
                                                    monkeypatch):
    """
    Citations are a function of what fitted, so widening the window must widen
    the citation list and narrowing it must shrink it. If these ever come out
    equal, `kept` has stopped being consulted and the model is being credited
    with passages it never saw again.
    """
    pipe.abstain_at = 0.0
    monkeypatch.setattr("rag._GEN_MAX_TOKENS", 120)
    narrow = len(pipe.answer("vendor delays in production", mode="generative").citations)
    monkeypatch.setattr("rag._GEN_MAX_TOKENS", 400)
    wide = len(pipe.answer("vendor delays in production", mode="generative").citations)
    assert 1 <= narrow < wide <= 3


def test_generative_refuses_an_impossible_window(pipe, stub_generator,
                                                 monkeypatch):
    """
    A window too small for the fixed instructions is a configuration error and
    must say so. The alternative — overflowing it — is the silent truncation
    that had flan-t5 being graded on a passage it never read.
    """
    monkeypatch.setattr("rag._GEN_MAX_TOKENS", 12)
    pipe.abstain_at = 0.0
    with pytest.raises(ValueError, match="configuration error"):
        pipe.answer("vendor delays in production", mode="generative")
