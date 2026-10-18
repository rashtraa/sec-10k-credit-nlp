"""
Stage 6a — retrieval-augmented answering over the filing index.

WHAT RAG ACTUALLY IS, MINUS THE ACRONYM
---------------------------------------
Retrieve the passages most likely to contain the answer, then produce an
answer *from those passages only*, with a citation back to the filing it came
from. The retrieval half is stage 5. This module is the answering half and the
part that decides whether the output can be trusted.

THE FAILURE MODE THAT MATTERS IN FINANCE
----------------------------------------
A retrieval system that returns the wrong paragraph is annoying. An answering
system that returns a confident, fluent, *wrong* sentence about a company's
debt covenants is dangerous, because nothing in the output signals that it was
invented. Fluency is not evidence.

So this module is built around three properties, in order of importance:

  GROUNDED     every sentence returned must be traceable to a specific passage
               of a specific filing. In extractive mode this is guaranteed by
               construction — the answer IS a span of the source — and the
               guarantee is *checked* rather than assumed, because a plumbing
               bug could break it silently.

  ATTRIBUTED   every answer carries CIK, ticker, filing year and passage id.
               An answer you cannot check is not an answer.

  ABSTAINING   when nothing retrieved is relevant, the system says so instead
               of answering from the closest thing it found. This is the
               single most valuable behaviour and the one most systems skip.

TWO MODES
---------
`extractive` (default, no model download beyond the embedder) selects the
sentences from the retrieved passages that best match the question. It cannot
hallucinate — the worst it can do is select the wrong sentence, which is
visible and measurable.

`generative` runs a small open-weights model (`flan-t5-base`, ~250 MB, CPU) to
write the answer from the retrieved context. It CAN hallucinate, which is
precisely why the faithfulness evaluation in `rag_eval.py` exists: with the
extractive mode as a 100%-grounded reference point, the generative mode's
grounding rate is a real measurement rather than a vibe.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from retrieval import reciprocal_rank_fusion, tokenize

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")

# Below this cosine similarity between question and best passage, the system
# refuses to answer.
#
# 0.45 was a guess and behaved like one: it answered 99.2% of the questions the
# corpus could not answer. 0.77 is not a guess. It is what
# `rag_eval.pick_threshold` returns on 400 answerable and 249 unanswerable
# questions when the constraint is "answer at most 10% of the unanswerable
# ones": it buys 51.5% coverage at an 8.0% false-answer rate.
#
# The budget is the input, not the output. A screening tool for an analyst who
# reads every hit wants a lower bar; anything feeding a published memo wants a
# higher one. Re-run the sweep in `07_rag.ipynb` and change this number — do
# not treat it as tuned.
DEFAULT_ABSTAIN_AT = 0.77


@dataclass
class Citation:
    chunk_id: str
    cik: int
    ticker: str
    filing_year: int
    score: float
    text: str

    def label(self) -> str:
        return f"{self.ticker} {self.filing_year} · {self.chunk_id}"


@dataclass
class Answer:
    query: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    abstained: bool = False
    mode: str = "extractive"

    # Every sentence of the answer paired with the chunk it was taken from.
    # Extractive mode fills this; generative mode cannot and leaves it empty.
    #
    # WHY THIS EXISTS — a bug found by the control it was meant to satisfy.
    # `grounded_verbatim` used to recover the answer's sentences by re-running
    # the sentence splitter over `text`. That is not a round trip: the splitter
    # only cuts on ".!?" followed by whitespace and a capital, so two extracted
    # sentences joined with a space came back as ONE string whenever the first
    # did not end in terminal punctuation or the second began lower-case — an
    # abbreviation, a bulleted fragment, a clause starting "provided that".
    # That single string appears in no passage, so a perfectly grounded answer
    # scored as ungrounded. It read as a 1.5% hallucination rate in a mode that
    # cannot hallucinate, which is the tell: extractive grounding is 1.0 by
    # construction, so anything below it is a defect in the measurement, not a
    # property of the system. Carrying the spans forward removes the guesswork.
    spans: list[tuple[str, str]] = field(default_factory=list)

    # chunk_ids the RETRIEVER returned, before the answerer chose which to
    # quote. Kept separate from `citations` so retrieval quality and citation
    # quality can be scored apart — a system that retrieves the right passage
    # and then quotes the wrong one has a different problem from one that never
    # retrieved it at all, and a single attribution number hides which.
    retrieved: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.abstained:
            return (f"Q: {self.query}\nA: [abstained — best match "
                    f"{self.confidence:.3f}, below threshold]")
        cites = "; ".join(c.label() for c in self.citations)
        return f"Q: {self.query}\nA: {self.text}\n   [{cites}]"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 25]


class RagPipeline:
    """
    Retrieval + answering over the stage-5 index.

    `encode` is injected rather than constructed here so the whole pipeline can
    be tested with a deterministic stand-in encoder — no model download, no
    network, and tests that fail for real reasons instead of flaky ones.
    """

    def __init__(self, chunks, embeddings: np.ndarray, bm25,
                 encode: Callable[[Sequence[str]], np.ndarray],
                 abstain_at: float = DEFAULT_ABSTAIN_AT):
        self.chunks = chunks.reset_index(drop=True)
        self.emb = embeddings
        self.bm25 = bm25
        self.encode = encode
        self.abstain_at = abstain_at
        self.texts = self.chunks["text"].tolist()

    # ────────────────────────────────────────────────────────── retrieval
    def retrieve(self, query: str, k: int = 5, pool: int = 20) -> list[int]:
        """
        Hybrid retrieval, because stage 5 measured that the two methods win on
        different queries: BM25 where the wording matches, embeddings where it
        does not. Fusing is not hedging — it is what the stratified result
        said to do.
        """
        qvec = self.encode([query])[0]
        dense_scores = self.emb @ qvec
        dense_rank = list(np.argsort(-dense_scores)[:pool])
        lex_rank = list(self.bm25.top_k(query, pool))
        fused = reciprocal_rank_fusion([lex_rank, dense_rank], top_n=k)
        return [int(i) for i in fused]

    def confidence(self, query: str, idx: Sequence[int]) -> float:
        """
        Cosine similarity between the question and the best retrieved passage.

        Bounded in [-1, 1] and comparable across queries, which BM25 scores are
        not — an unbounded sum cannot be thresholded consistently. This is the
        number the abstention decision is made on, so it has to mean the same
        thing every time.
        """
        if not len(idx):
            return 0.0
        qvec = self.encode([query])[0]
        return float(np.max(self.emb[list(idx)] @ qvec))

    # ────────────────────────────────────────────────────────── answering
    def _cite(self, i: int, score: float) -> Citation:
        r = self.chunks.iloc[i]
        return Citation(chunk_id=str(r["chunk_id"]), cik=int(r["cik"]),
                        ticker=str(r["ticker"]), filing_year=int(r["filing_year"]),
                        score=float(score), text=str(r["text"]))

    def answer(self, query: str, mode: str = "extractive", k: int = 5,
               n_sentences: int = 2) -> Answer:
        idx = self.retrieve(query, k=k)
        conf = self.confidence(query, idx)
        retrieved = [str(self.chunks.iloc[i]["chunk_id"]) for i in idx]

        if conf < self.abstain_at:
            return Answer(query=query, text="", confidence=conf,
                          abstained=True, mode=mode, retrieved=retrieved)

        if mode == "extractive":
            a = self._extractive(query, idx, conf, n_sentences)
        elif mode == "generative":
            a = self._generative(query, idx, conf)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        a.retrieved = retrieved
        return a

    def _extractive(self, query: str, idx: Sequence[int], conf: float,
                    n_sentences: int) -> Answer:
        """
        Score every sentence of the retrieved passages against the question and
        return the best few, each tagged with the filing it came from.

        Sentences are scored with the SAME embedding model that built the
        index. Using a different scorer would rank sentences by one notion of
        similarity inside passages selected by another, and the two halves
        would be pulling in different directions.
        """
        cand: list[tuple[str, int]] = []
        for i in idx[:3]:                       # 3 passages is ~10-20 sentences
            for s in split_sentences(self.texts[i]):
                cand.append((s, i))
        if not cand:
            return Answer(query=query, text="", confidence=conf,
                          abstained=True, mode="extractive")

        qvec = self.encode([query])[0]
        svecs = self.encode([s for s, _ in cand])
        scores = svecs @ qvec
        order = np.argsort(-scores)[:n_sentences]

        picked = [cand[j] for j in order]
        text = " ".join(s for s, _ in picked)
        seen, cites = set(), []
        for (_, i), j in zip(picked, order):
            if i not in seen:
                seen.add(i)
                cites.append(self._cite(i, float(scores[j])))
        spans = [(s, str(self.chunks.iloc[i]["chunk_id"])) for s, i in picked]
        return Answer(query=query, text=text, citations=cites, spans=spans,
                      confidence=conf, mode="extractive")

    def _generative(self, query: str, idx: Sequence[int], conf: float) -> Answer:
        """
        Small open-weights generator over the retrieved context.

        The prompt does two things that matter: it restricts the model to the
        context, and it gives it an explicit way out. A model with no way to
        say "not stated" will invent something, because producing text is the
        only behaviour it has.
        """
        gen, tok = _get_generator()

        # FIT THE CONTEXT TO THE MODEL, DO NOT LET IT BE CUT SILENTLY.
        #
        # The first run printed "Token indices sequence length is longer than
        # the specified maximum sequence length for this model (546 > 512)".
        # That is not cosmetic: flan-t5's encoder window is 512, so the tail of
        # the prompt was being dropped. The tail was the third retrieved
        # passage — and the evaluation was still crediting that passage as a
        # citation. The model was being graded on text it never saw.
        #
        # Passages are now added whole while they fit and stopped when they do
        # not, so the citations are exactly the passages the model read.
        head = ("Answer the question using ONLY the context below. If the "
                "context does not contain the answer, reply exactly: not "
                "stated.\n\nContext:\n")
        tail = f"\n\nQuestion: {query}\nAnswer:"
        budget = _GEN_MAX_TOKENS - _n_tokens(tok, head + tail) - 8
        if budget < 32:
            raise ValueError(
                f"the fixed prompt leaves only {budget} tokens for context in a "
                f"{_GEN_MAX_TOKENS}-token window. That is a configuration "
                f"error, not something to paper over by overflowing the window "
                f"— which is the bug this whole branch exists to prevent.")

        used, parts = 0, []
        for n, i in enumerate(idx[:3]):
            piece = f"[{n+1}] {self.texts[i]}\n\n"
            cost = _n_tokens(tok, piece)
            if used + cost > budget:
                break
            parts.append(piece)
            used += cost

        if not parts:
            # A single passage longer than the whole window. Truncate it TO THE
            # BUDGET rather than to a fixed character count — an earlier version
            # cut at 1,200 characters here and quietly overflowed the window
            # again, which is the exact failure the fitting loop above exists to
            # prevent. Caught by the test that pins the prompt length.
            piece = f"[1] {self.texts[idx[0]]}"
            while piece and _n_tokens(tok, piece + "\n\n") > budget:
                piece = piece[:int(len(piece) * 0.8)]
            parts = [piece + "\n\n"]

        kept = len(parts)
        prompt = head + "".join(parts) + tail
        text = gen(prompt)

        # Cite only what was actually in the window. Citing a passage the model
        # could not read would inflate attribution@cited and grounding alike.
        cites = [self._cite(i, 0.0) for i in idx[:kept]]
        if text.strip().lower().startswith("not stated"):
            return Answer(query=query, text=text, citations=cites,
                          confidence=conf, abstained=True, mode="generative")
        return Answer(query=query, text=text, citations=cites,
                      confidence=conf, mode="generative")


# ────────────────────────────────────────────────────── generator (optional)
_GEN = None
GEN_MODEL = "google/flan-t5-base"
_GEN_MAX_TOKENS = 512          # flan-t5 encoder window; exceeding it truncates


def _n_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _get_generator():
    """Returns (callable, tokenizer). The tokenizer is returned so the prompt
    can be fitted to the window instead of silently truncated inside it."""
    global _GEN
    if _GEN is None:
        from transformers import pipeline
        pipe = pipeline("text2text-generation", model=GEN_MODEL,
                        max_new_tokens=96)
        _GEN = (lambda p: pipe(p)[0]["generated_text"], pipe.tokenizer)
    return _GEN


# ────────────────────────────────────────────────────────────── grounding
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def grounded_verbatim(answer: Answer) -> bool:
    """
    Is every sentence of the answer literally present in the passage it is
    attributed to?

    This is the strictest possible faithfulness test and the reason extractive
    mode exists: it should return True 100% of the time, so any failure is a
    bug in the pipeline rather than a property of the model. A control that can
    only ever pass is worthless — this one can fail, and did. Twice.

    FAILURE ONE (fixed earlier): whitespace normalised on the answer but not on
    the source, so any sentence crossing a line break failed.

    FAILURE TWO (fixed here): the sentences were recovered by re-splitting
    `answer.text`, which is not the inverse of joining them. The splitter cuts
    on terminal punctuation followed by a capital, so two extracted sentences
    came back glued together whenever the first ended in an abbreviation or the
    second opened lower-case. The glued string is in no passage, so the answer
    scored ungrounded. Real corpus: 6 of 400. Extractive mode reported a 1.5%
    hallucination rate while being incapable of hallucinating — which is how
    the bug announced itself. Now the spans are carried, not reconstructed.

    Checking each span against ITS OWN passage rather than against all cited
    passages concatenated is also strictly harder: the concatenated version
    would accept a sentence that straddles the join between two sources, a
    string that exists in the evaluation and nowhere in the filings.
    """
    if answer.abstained or not answer.text:
        return True
    if answer.spans:
        by_id = {c.chunk_id: _norm(c.text) for c in answer.citations}
        return all(_norm(sent) in by_id.get(cid, "")
                   for sent, cid in answer.spans)
    # Generative mode has no spans — fall back to containment anywhere in the
    # cited context, which is the only question that can be asked of text the
    # system wrote rather than selected.
    src = _norm(" ".join(c.text for c in answer.citations))
    return all(_norm(s) in src
               for s in (split_sentences(answer.text) or [answer.text]))


def grounding_overlap(answer: Answer, n: int = 1) -> float:
    """
    Share of the answer's content n-grams that appear in the cited passages.

    The soft version of the check above, for generative mode where verbatim
    containment is the wrong bar.

    READ THE UNIGRAM VERSION (n=1) SCEPTICALLY. The cited context is three
    whole passages, roughly 2,000 words of the same register the answer is
    written in. Under that source, a short answer scoring 1.00 means almost
    nothing: "risk", "material", "adverse", "obligations" are in every filing
    ever written, so a model recombining stock phrases into a false claim
    scores exactly as well as one quoting correctly. The first evaluation run
    returned 1.0000 for both modes, which is not a result — it is a metric with
    no dynamic range on this corpus.

    n=2 is where it starts to measure something. Word ORDER is not free: a
    bigram absent from the source is a pairing the system invented, and
    inventing pairings is what "our debt does not restrict dividends" looks
    like when the source says it does. Reported alongside, never instead —
    the unigram number is kept so the gap between the two is visible.
    """
    if answer.abstained or not answer.text:
        return 1.0

    def grams(text: str) -> list[str]:
        toks = [t for t in tokenize(text) if len(t) > 2]
        if n == 1:
            return toks
        return [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]

    # SPAN-WISE, for the same reason `grounded_verbatim` is span-wise, and it
    # matters more here than it looks.
    #
    # Scoring the joined answer against the joined citations invents n-grams on
    # BOTH sides. On the answer side, the pair straddling two extracted
    # sentences was never written by anyone — it is an artefact of the join, it
    # exists in no filing, and it is counted as a miss. On the source side, the
    # pair straddling two concatenated passages is equally fictional and is
    # counted as a hit. The first depressed extractive bigram grounding to
    # 0.9855 in a mode where it is 1.0 by construction; the second would let a
    # sentence stitched across two filings pass as grounded.
    #
    # Both vanish once each span is scored against its own passage.
    if answer.spans:
        by_id = {c.chunk_id: set(grams(c.text)) for c in answer.citations}
        hits = total = 0
        for sent, cid in answer.spans:
            g = grams(sent)
            src = by_id.get(cid, set())
            hits += sum(t in src for t in g)
            total += len(g)
        return hits / total if total else 1.0

    src = set(grams(" ".join(c.text for c in answer.citations)))
    ans = grams(answer.text)
    if not ans:
        return 1.0
    return sum(t in src for t in ans) / len(ans)


def unsupported_terms(answer: Answer, limit: int = 8) -> list[str]:
    """The content words in the answer that appear in no cited passage."""
    if answer.abstained or not answer.text:
        return []
    src = set(tokenize(" ".join(c.text for c in answer.citations)))
    out = [t for t in tokenize(answer.text) if len(t) > 3 and t not in src]
    return sorted(set(out))[:limit]
