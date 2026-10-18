"""
Stage 6b — does the answering layer actually work, and does it know when it
doesn't?

Three questions, in the order they matter.

ATTRIBUTION — when the system answers, does it cite the right filing? An
answer with a confident wrong citation is worse than no answer, because the
citation is what makes a reader stop checking.

GROUNDING — is the answer traceable to the cited text? Extractive mode should
score 1.00 by construction; that is a plumbing control. Generative mode is
where the number becomes a measurement.

ABSTENTION — the one most systems skip. Given a question the corpus cannot
answer, does the system say so? This is treated here as a binary classifier
and scored with ROC-AUC, the same way the credit models in stage 4 were: a
system that abstains on everything scores perfectly on refusals and is
useless, so refusal rate alone proves nothing.

WHERE THE UNANSWERABLE QUESTIONS COME FROM
------------------------------------------
Not invented. The index holds a sample of filings; thousands of other filings
were parsed in stage 2 and left out. Their risk headings are real
analyst-style questions about companies genuinely absent from the index —
unanswerable by construction rather than by contrivance.

They are derived from `sections.parquet`, NOT from `heading_queries.parquet`.
The latter is generated from the indexed chunks, so every query in it points
at a passage that is present; reading the negative control out of it produced
an empty pool.

One catch, handled: risk language repeats across firms, so a question from an
unindexed filing might still be answerable from an indexed one. Candidates are
therefore screened by TF-IDF similarity against the whole index, and anything
with a plausible answer already present is dropped from the unanswerable set.
Without that screen the "unanswerable" questions would be partly answerable
and the AUC would be measuring the wrong thing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rag import Answer, grounded_verbatim, grounding_overlap


# ─────────────────────────────────────────────── the unanswerable question set
def build_unanswerable(chunks: pd.DataFrame, n: int = 400,
                       max_similarity: float = 0.30,
                       sections: pd.DataFrame | None = None,
                       scan: int = 600,
                       seed: int = 20260821) -> pd.DataFrame:
    """
    Real questions about filings the index has never seen.

    WHERE THEY COME FROM
    --------------------
    Your index holds a SAMPLE of filings; thousands more were parsed in stage 2
    and left out. Those filings' risk headings are genuine analyst-style
    questions about companies absent from the corpus — unanswerable by
    construction rather than by contrivance.

    An earlier version tried to read them out of `heading_queries.parquet`,
    which cannot work: that file is generated FROM the indexed chunks, so every
    query in it points at a passage that is present. The pool came back empty
    and sklearn raised on a zero-row matrix. The questions have to be derived
    from the sections themselves.

    THE SCREEN, AND WHY IT IS THE IMPORTANT PART
    --------------------------------------------
    Risk language repeats across firms, so a question from an unindexed filing
    might still be answerable from an indexed one — in which case it is not a
    negative control at all, and the abstention AUC would be measuring
    something else.

    So every candidate is checked by TF-IDF similarity against the whole index
    and anything with a plausible answer already present is discarded. The
    threshold is deliberately low: a false "unanswerable" poisons the control,
    while discarding a usable candidate costs only sample size.

    WHY THE THRESHOLD MOVED FROM 0.45 TO 0.30
    -----------------------------------------
    0.45 was too loose and the first evaluation run proved it. The system
    "wrongly" answered 399 of 400 unanswerable questions — and reading the
    failures showed most were not failures. "Our indebtedness could adversely
    affect our financial health" was answered from another firm's "Our
    indebtedness could adversely affect our financial condition", cited
    correctly, at confidence 0.883. That is the system working. The retained
    candidates topped out at 0.447 similarity, hard against the ceiling, which
    is the shape of a screen that is binding on nothing.

    A negative control has to be unanswerable in fact, not merely sourced from
    a held-out file. Risk language is boilerplate; being absent from the index
    is not the same as having no answer in it.

    WHY THE SCREEN IS TF-IDF AND NOT THE EMBEDDING MODEL
    ---------------------------------------------------
    Screening with the same encoder that produces `confidence` would be rigging
    the experiment: every question the encoder finds similar to the index would
    be thrown out of the negative set, so by construction the survivors score
    low confidence and the abstention AUC goes up without the system improving
    at all. The screen must be independent of the score being evaluated. TF-IDF
    is a lexical measure, the confidence is a dense one, and keeping them
    different is what makes the AUC an honest number.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    import chunking
    from config import INTERIM

    if sections is None:
        sections = pd.read_parquet(INTERIM / "sections.parquet")
    sections = sections[sections["item1a_status"].isin(("ok", "ok_loose"))]

    # Filings the index does not contain, identified by (firm, year).
    indexed = set(zip(chunks["cik"].astype(int), chunks["filing_year"].astype(int)))
    mask = [(int(c), int(y)) not in indexed
            for c, y in zip(sections["cik"], sections["filing_year"])]
    outside = sections[mask]
    if outside.empty:
        raise ValueError(
            "every parsed filing is already in the index, so there are no "
            "genuinely unanswerable questions. Rebuild the index with "
            "--sample so that some filings are held out.")

    outside = outside.sample(min(scan, len(outside)), random_state=seed)

    rows = []
    for r in outside.itertuples(index=False):
        for c in chunking.chunk_section(r.item1a, int(r.cik), str(r.ticker),
                                        int(r.filing_year)):
            if c.heading and chunking.is_usable_query(c.heading):
                rows.append({"query": c.heading, "cik": int(r.cik),
                             "filing_year": int(r.filing_year)})
    if not rows:
        raise ValueError("no usable headings found in the held-out filings")

    pool = (pd.DataFrame(rows)
            .drop_duplicates(subset="query")
            .sample(frac=1.0, random_state=seed)
            .head(n * 4)
            .reset_index(drop=True))

    texts = chunks["text"].tolist()
    vec = TfidfVectorizer(lowercase=True, stop_words="english", min_df=2,
                          sublinear_tf=True)
    X = vec.fit_transform(texts)
    Q = vec.transform(pool["query"].tolist())

    best = np.zeros(Q.shape[0], dtype=np.float32)
    for s in range(0, Q.shape[0], 256):
        block = (Q[s:s + 256] @ X.T).tocsr()
        for r in range(block.shape[0]):
            lo, hi = block.indptr[r], block.indptr[r + 1]
            best[s + r] = block.data[lo:hi].max() if hi > lo else 0.0

    keep = pool[best <= max_similarity].head(n).copy()
    keep["best_similarity_to_index"] = best[best <= max_similarity][:len(keep)]
    # Diagnostics for the screen itself, so its effect can be argued about
    # rather than asserted. `rejected_share` is the honest one: a screen that
    # discards most of the pool is doing work regardless of where the survivors
    # happen to bunch up.
    keep.attrs["pool_size"] = int(len(pool))
    keep.attrs["rejected_share"] = float((best > max_similarity).mean())
    keep.attrs["pool_similarity_median"] = float(np.median(best))
    keep.attrs["threshold"] = float(max_similarity)
    # `relevant_chunk_id` is never satisfiable — it exists so the evaluator can
    # use one code path for both question sets.
    keep["relevant_chunk_id"] = "__not_in_index__"
    return keep.reset_index(drop=True)


# ────────────────────────────────────────────────────────────────── scoring
@dataclass
class RagReport:
    mode: str
    n_answerable: int
    n_unanswerable: int
    answer_rate: float          # share of answerable questions answered
    false_answer_rate: float    # share of UNanswerable questions answered
    retrieval_recall_at_k: float  # gold passage anywhere in the k RETRIEVED
    attribution_at_1: float     # right passage CITED first
    attribution_at_cited: float  # right passage cited anywhere in the citations
    citation_ceiling: float     # attribution@1 among questions retrieval solved
    grounded_verbatim: float
    grounding_overlap: float    # unigram — weak, see rag.grounding_overlap
    grounding_bigram: float     # word order — the one with dynamic range
    abstention_auc: float
    median_ms: float


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """
    Probability a random answerable question scores above a random
    unanswerable one. Computed by rank rather than by sweeping thresholds, so
    it is exact and ties count as half — the same estimator used for the
    credit models, kept identical on purpose.
    """
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    s = pd.Series(allv)
    ranks = s.rank(method="average").to_numpy()
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate_rag(pipeline, answerable: pd.DataFrame, unanswerable: pd.DataFrame,
                 mode: str = "extractive", k: int = 5,
                 progress: bool = True) -> tuple[RagReport, pd.DataFrame]:
    """
    Runs both question sets and returns the report plus the per-question rows,
    because the aggregate is for the CV and the rows are for understanding.
    """
    import time
    rows = []

    def run(df, answerable_flag):
        it = df.itertuples(index=False)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, total=len(df),
                      desc=f"{mode}/{'answerable' if answerable_flag else 'unanswerable'}")
        for r in it:
            t0 = time.time()
            a: Answer = pipeline.answer(r.query, mode=mode, k=k)
            ms = (time.time() - t0) * 1000
            cited = [c.chunk_id for c in a.citations]
            gold = getattr(r, "relevant_chunk_id", None)
            rows.append({
                "query": r.query, "answerable": answerable_flag,
                "abstained": a.abstained, "confidence": a.confidence,
                "text": a.text, "ms": ms,
                "cited_1": cited[0] if cited else None,
                "n_cited": len(cited),
                # Did the RETRIEVER find it? Separate question from whether the
                # answerer then quoted it. Collapsing the two into one
                # "attribution" number makes an answerer bug look like a
                # retrieval bug and hides which half to work on.
                "retrieved_k": gold in a.retrieved,
                "hit_1": bool(cited) and cited[0] == gold,
                "hit_cited": gold in cited,
                "grounded": grounded_verbatim(a),
                "overlap": grounding_overlap(a),
                "bigram": grounding_overlap(a, n=2),
            })

    run(answerable, True)
    run(unanswerable, False)
    df = pd.DataFrame(rows)

    ans = df[df.answerable]
    una = df[~df.answerable]
    answered = ans[~ans.abstained]

    # Among the questions the retriever DID solve, how often was the right
    # passage the one quoted? This is the answerer's own score with retrieval
    # held constant — the ceiling attribution could reach without touching
    # stage 5. If it is high, retrieval is the bottleneck; if it is low, the
    # sentence-selection step is picking the wrong passage out of a set that
    # contained the right one, which is a different fix entirely.
    solved = answered[answered.retrieved_k]

    def m(frame, col):
        return float(frame[col].mean()) if len(frame) else float("nan")

    report = RagReport(
        mode=mode,
        n_answerable=len(ans), n_unanswerable=len(una),
        answer_rate=float((~ans.abstained).mean()),
        false_answer_rate=float((~una.abstained).mean()),
        retrieval_recall_at_k=m(answered, "retrieved_k"),
        # Attribution is scored on questions the system CHOSE to answer.
        # Counting abstentions as attribution failures would reward a system
        # that answers everything badly over one that answers less and better.
        attribution_at_1=m(answered, "hit_1"),
        # NOT "@k". Extractive mode cites only the passages it actually quoted
        # — typically one or two of the k retrieved — so calling this
        # attribution@5 would imply five chances when there were two. The
        # denominator is the citation list, and the name now says so.
        attribution_at_cited=m(answered, "hit_cited"),
        citation_ceiling=m(solved, "hit_1"),
        grounded_verbatim=m(answered, "grounded"),
        grounding_overlap=m(answered, "overlap"),
        grounding_bigram=m(answered, "bigram"),
        abstention_auc=_auc(ans.confidence.to_numpy(), una.confidence.to_numpy()),
        median_ms=float(df.ms.median()),
    )
    return report, df


def report_table(reports: list[RagReport]) -> pd.DataFrame:
    return pd.DataFrame([{
        "mode": r.mode,
        "answered (answerable)": round(r.answer_rate, 4),
        "answered (unanswerable)": round(r.false_answer_rate, 4),
        "retrieval recall@k": round(r.retrieval_recall_at_k, 4),
        "attribution@1": round(r.attribution_at_1, 4),
        "attribution@cited": round(r.attribution_at_cited, 4),
        "citation ceiling": round(r.citation_ceiling, 4),
        "grounded verbatim": round(r.grounded_verbatim, 4),
        "grounding overlap (1gram)": round(r.grounding_overlap, 4),
        "grounding overlap (2gram)": round(r.grounding_bigram, 4),
        "abstention AUC": round(r.abstention_auc, 4),
        "median ms": round(r.median_ms, 1),
    } for r in reports])


def threshold_sweep(df: pd.DataFrame,
                    grid: tuple[float, ...] = (0.40, 0.50, 0.60, 0.65, 0.70,
                                               0.75, 0.80, 0.85, 0.90)) -> pd.DataFrame:
    """
    The abstention threshold is a business decision, not a hyperparameter.

    Every row is a different answer to "how wrong are we willing to be in
    exchange for how much coverage". Reported as a table so the choice is
    visible instead of buried in a constant at the top of a module.
    """
    ans = df[df.answerable]["confidence"].to_numpy()
    una = df[~df.answerable]["confidence"].to_numpy()
    out = []
    for t in grid:
        tp = (ans >= t).sum()
        fp = (una >= t).sum()
        out.append({
            "threshold": t,
            "coverage (answerable answered)": round(tp / max(len(ans), 1), 4),
            "false answers (unanswerable answered)": round(fp / max(len(una), 1), 4),
            "precision of answering": round(tp / max(tp + fp, 1), 4),
        })
    return pd.DataFrame(out)


def pick_threshold(df: pd.DataFrame, max_false_answer_rate: float = 0.10,
                   grid: np.ndarray | None = None) -> dict:
    """
    Choose the operating point by naming the cost, not by maximising a score.

    "Best threshold" is not a well-formed question here. F1 would weigh a
    wrong answer about a company's covenants equally against a missed one, and
    they are not equal — a confident wrong citation is worse than silence,
    because the citation is what makes a reader stop checking.

    So the constraint is stated first: at most `max_false_answer_rate` of the
    questions the corpus cannot answer may be answered anyway. Among the
    thresholds that satisfy it, take the one with the most coverage. That is a
    defensible sentence in an interview — "we chose 10% because that is what
    the reviewer's tolerance was, and here is the coverage it bought" — and
    "0.45 looked about right" is not.

    Returns the chosen threshold plus what it costs, or the tightest available
    point with `satisfiable=False` when no threshold on the grid meets the
    budget, which is itself the finding.
    """
    ans = df[df.answerable]["confidence"].to_numpy()
    una = df[~df.answerable]["confidence"].to_numpy()
    if grid is None:
        grid = np.round(np.arange(0.20, 0.951, 0.01), 3)

    best = None
    for t in grid:
        fa = float((una >= t).mean()) if len(una) else 0.0
        cov = float((ans >= t).mean()) if len(ans) else 0.0
        if fa <= max_false_answer_rate and (best is None or cov > best["coverage"]):
            best = {"threshold": float(t), "coverage": round(cov, 4),
                    "false_answer_rate": round(fa, 4), "satisfiable": True}
    if best is None:
        t = float(grid[-1])
        best = {"threshold": t,
                "coverage": round(float((ans >= t).mean()), 4),
                "false_answer_rate": round(float((una >= t).mean()), 4),
                "satisfiable": False}
    return best


def report_at_threshold(df: pd.DataFrame, t: float, mode: str,
                        original_threshold: float) -> RagReport:
    """
    Re-score an evaluation at a HIGHER abstention threshold without re-running
    the pipeline.

    Legitimate because raising the bar only removes answers: every question
    still answered at `t` was answered at the original threshold, so its
    citation and grounding were already measured and the new answered set is a
    strict subset of the old one. Confidence is recorded for abstentions too,
    which is why the abstention AUC is unchanged — AUC ranks scores and never
    consults the threshold.

    LOWERING the threshold is refused rather than approximated. Questions below
    the original bar were never answered, so there is no citation to score and
    no grounding to check; filling those in from nothing would invent the
    result. Re-run the pipeline instead.
    """
    if t < original_threshold - 1e-9:
        raise ValueError(
            f"cannot re-score at {t} below the threshold the pipeline actually "
            f"ran at ({original_threshold}) — the questions it abstained on "
            f"have no answers to grade. Re-run evaluate_rag with "
            f"pipe.abstain_at = {t}.")

    ans, una = df[df.answerable], df[~df.answerable]
    answered = ans[(~ans.abstained) & (ans.confidence >= t)]
    solved = answered[answered.retrieved_k]

    def m(frame, col):
        return float(frame[col].mean()) if len(frame) else float("nan")

    return RagReport(
        mode=f"{mode} @ {t:.2f}",
        n_answerable=len(ans), n_unanswerable=len(una),
        answer_rate=float(((~ans.abstained) & (ans.confidence >= t)).mean()),
        false_answer_rate=float(((~una.abstained) & (una.confidence >= t)).mean()),
        retrieval_recall_at_k=m(answered, "retrieved_k"),
        attribution_at_1=m(answered, "hit_1"),
        attribution_at_cited=m(answered, "hit_cited"),
        citation_ceiling=m(solved, "hit_1"),
        grounded_verbatim=m(answered, "grounded"),
        grounding_overlap=m(answered, "overlap"),
        grounding_bigram=m(answered, "bigram"),
        abstention_auc=_auc(ans.confidence.to_numpy(), una.confidence.to_numpy()),
        median_ms=float(df.ms.median()),
    )
