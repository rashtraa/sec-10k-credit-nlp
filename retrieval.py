"""
Stage 5b — three retrieval methods, and the metrics to compare them.

THE THREE METHODS
-----------------
BM25 (lexical). Ranks by word overlap, weighting rare words higher and
discounting long documents. It is the classic keyword search algorithm and it
is a genuinely hard baseline — decades of information-retrieval work have
failed to beat it as often as people expect.

Dense (semantic). Encodes query and passage into vectors with a neural network
trained so that text with similar MEANING lands nearby, then ranks by cosine
similarity. It can match "supply chain disruption" to "our vendors may fail to
deliver" with no words in common — something BM25 cannot do at all.

Hybrid. Runs both and fuses the rankings.

WHY COMPARE THEM RATHER THAN ASSUME
-----------------------------------
The default assumption is that dense beats lexical. On general prose it
usually does. On specialised jargon it frequently does not, for a specific
reason: embedding models are trained mostly on general web text, and terms
like "covenant", "impairment" or "securitisation" are either rare in that
training data or carry a different everyday meaning. Exact-token matching can
be worth more than approximate semantics when the vocabulary is technical and
the user knows it.

Which of those happens on 10-K risk factors is an empirical question. This
module exists to answer it with a number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# ══════════════════════════════════════════════════════════ BM25
class BM25:
    """
    Okapi BM25, implemented directly so the scoring is inspectable.

        score(q, d) = SUM over query terms of
                      IDF(t) * f(t,d)*(k1+1)
                      -----------------------------------------
                      f(t,d) + k1*(1 - b + b*|d|/avgdl)

    Three ideas, each doing real work:

      IDF(t)          rare words count for more than common ones
      saturation      the k1 term makes the 10th occurrence of a word worth
                      far less than the 2nd — mentioning "risk" twenty times
                      does not make a passage twenty times more relevant
      length norm     the b term stops long passages winning simply by
                      containing more words

    WHY THE SPARSE MATRIX RATHER THAN A DICT OF POSTINGS
    ----------------------------------------------------
    The textbook implementation stores `term -> {doc: count}` as nested Python
    dicts. That is readable and it does not survive this corpus. ~200,000
    passages at ~120 distinct terms each is ~25 million postings; as dict
    entries that is several GB of RAM before a single query runs.

    The same information is one scipy CSR matrix: 25M nonzeros at 8 bytes
    (float32 value + int32 column) is ~200 MB. Same arithmetic, ~20x less
    memory, and scoring becomes a column slice instead of a Python loop.

    The per-nonzero BM25 weight is precomputed once at build time, because
    every term of the formula except *which* columns the query selects depends
    only on the corpus. A query is then: pick those columns, sum the rows.
    """

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        from array import array
        from scipy import sparse

        self.k1, self.b = k1, b
        self.N = len(corpus_tokens)

        # ---- build the document-term count matrix --------------------------
        # array('i') rather than a Python list: 4 bytes per entry instead of
        # ~28, which is the difference between fitting in memory and not.
        vocab: dict[str, int] = {}
        # 'i' is 4 bytes and 'q' is 8 bytes on every platform; 'l' is not
        # (4 bytes on Windows, 8 on Linux), so it is avoided deliberately.
        indices, data = array("i"), array("i")
        indptr = array("q", [0])
        for doc in corpus_tokens:
            counts: dict[int, int] = {}
            for t in doc:
                j = vocab.get(t)
                if j is None:
                    j = vocab[t] = len(vocab)
                counts[j] = counts.get(j, 0) + 1
            indices.extend(counts.keys())
            data.extend(counts.values())
            indptr.append(len(indices))
        self.vocab = vocab
        V = max(len(vocab), 1)

        tf = sparse.csr_matrix(
            (np.frombuffer(data, dtype=np.int32).astype(np.float32),
             np.frombuffer(indices, dtype=np.int32).copy(),
             np.frombuffer(indptr, dtype=np.int64).copy()),
            shape=(self.N, V))
        del indices, data, indptr

        self.doc_len = np.asarray(tf.sum(axis=1)).ravel().astype(np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0

        # document frequency: each (doc, term) pair appears exactly once, so
        # counting column indices counts documents containing the term.
        df = np.bincount(tf.indices, minlength=V).astype(np.float32)
        # Robertson IDF with the +0.5 smoothing that keeps it positive
        self.idf = np.log(1.0 + (self.N - df + 0.5) / (df + 0.5)).astype(np.float32)

        # ---- collapse the whole formula into the stored values -------------
        rows = np.repeat(np.arange(self.N, dtype=np.int64), np.diff(tf.indptr))
        f = tf.data
        denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[rows] / self.avgdl)
        tf.data = (self.idf[tf.indices] * f * (self.k1 + 1) / denom).astype(np.float32)
        del rows, denom
        self.W = tf.tocsc()          # CSC: column slicing is what queries do

    def score(self, query: str) -> np.ndarray:
        # Repeated query terms are counted repeatedly, as in the original
        # formulation — a query saying "risk" twice weights it twice.
        cols = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not cols:
            return np.zeros(self.N, dtype=np.float32)
        return np.asarray(self.W[:, cols].sum(axis=1)).ravel().astype(np.float32)

    def top_k(self, query: str, k: int = 10) -> list[int]:
        s = self.score(query)
        if k >= self.N:
            return list(np.argsort(-s))
        idx = np.argpartition(-s, k)[:k]
        return list(idx[np.argsort(-s[idx])])


# ══════════════════════════════════════════════════════════ Dense
class DenseIndex:
    """
    Embedding search over a FAISS index.

    Vectors are L2-normalised and the index uses inner product, which for
    unit vectors IS cosine similarity — the same measure used for
    year-over-year document change in stage 3, applied here between a query
    and a passage instead of between two filings.
    """

    def __init__(self, embeddings: np.ndarray):
        import faiss
        self.dim = embeddings.shape[1]
        emb = np.ascontiguousarray(embeddings.astype("float32"))
        faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatIP(self.dim)   # exact search, no approximation
        self.index.add(emb)

    def top_k(self, query_vec: np.ndarray, k: int = 10) -> list[int]:
        import faiss
        q = np.ascontiguousarray(query_vec.reshape(1, -1).astype("float32"))
        faiss.normalize_L2(q)
        _, idx = self.index.search(q, k)
        return list(idx[0])


# ══════════════════════════════════════════════════════════ Hybrid
def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60,
                           top_n: int = 10) -> list[int]:
    """
    Combine rankings by RANK, never by score.

    BM25 scores are unbounded sums; cosine similarities live in [-1, 1]. They
    are not comparable, and normalising them to a common range is a fudge that
    silently depends on the score distribution of whichever corpus you happen
    to have.

    Reciprocal Rank Fusion sidesteps the problem: each list contributes
    1/(k + rank) for every document, and the constant k (conventionally 60)
    damps the influence of the very top positions so one confident-but-wrong
    system cannot dominate. Only the ORDER matters, which is the only thing
    the two methods genuinely agree on the meaning of.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]]


# ══════════════════════════════════════════════════════════ metrics
@dataclass
class RetrievalResult:
    method: str
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    n_queries: int
    median_ms: float


def build_relevance_sets(gold_idx: list[int], texts: list[str],
                         threshold: float = 0.85,
                         batch: int = 256) -> list[set[int]]:
    """
    One gold answer per query is wrong when the corpus contains duplicates.

    THE PROBLEM
    -----------
    Risk-factor boilerplate is reused across companies — law firms recycle
    language, and hundreds of firms write "we depend on a limited number of
    suppliers" almost identically. Mark exactly ONE of those passages
    correct and a system returning a perfectly relevant one is scored wrong
    every other time.

    Measured on a synthetic corpus built to have this property: 1,200
    passages carrying 16 distinct risk topics, a median of 86 near-identical
    passages per query. Under single-gold scoring BM25 recorded Recall@1 of
    0.005 and read as a total failure. Scored against relevance sets, the
    same rankings gave 0.930 — BM25 was returning a correct passage almost
    every time and being marked wrong for returning the wrong COPY of it.
    The strict number was measuring corpus duplication, not retrieval.

    THE FIX
    -------
    Standard information-retrieval practice: a query has a relevance SET, not
    a single answer. Here that set is the gold passage plus every passage
    that is a near-duplicate of it, by TF-IDF cosine similarity. Retrieving
    any member counts as a hit, which is what a user would actually want.

    The threshold matters and is reported: at 0.85 two passages must share
    most of their distinctive vocabulary to be treated as interchangeable.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    # min_df=2 drops words appearing in a single passage: they cannot make two
    # passages look alike, and dropping them shrinks the vocabulary a lot. On
    # a tiny or fully-disjoint corpus it can prune EVERYTHING, which sklearn
    # raises on — fall back rather than fail.
    try:
        vec = TfidfVectorizer(lowercase=True, stop_words="english",
                              min_df=2, sublinear_tf=True)
        X = vec.fit_transform(texts)      # L2-normalised by default
    except ValueError:
        vec = TfidfVectorizer(lowercase=True, stop_words="english",
                              min_df=1, sublinear_tf=True)
        X = vec.fit_transform(texts)

    # The similarity block stays SPARSE. Densifying it would allocate
    # batch x n_passages floats — at 256 x 200,000 that is 200 MB per batch,
    # for a result where all but a handful of entries are below threshold
    # anyway. Reading the CSR row structure directly costs nothing.
    sets: list[set[int]] = []
    for start in range(0, len(gold_idx), batch):
        rows = gold_idx[start:start + batch]
        sims = (X[rows] @ X.T).tocsr()     # cosine == dot for L2-normalised rows
        for k, r in enumerate(rows):
            lo, hi = sims.indptr[k], sims.indptr[k + 1]
            cols, vals = sims.indices[lo:hi], sims.data[lo:hi]
            near = set(cols[vals >= threshold].tolist())
            near.add(r)                    # the gold is always relevant
            sets.append(near)
    return sets


def evaluate(rankings: list[list[int]], relevant: list[int] | list[set[int]],
             method: str, median_ms: float = float("nan")) -> RetrievalResult:
    """
    Recall@k — in what share of queries does the correct passage appear in
    the top k? This is what matters for a retrieval system feeding an answer
    layer: if the right passage is not retrieved, nothing downstream can
    recover it.

    MRR (Mean Reciprocal Rank) — average of 1/(position of the correct
    passage). Rank 1 scores 1.0, rank 2 scores 0.5, rank 10 scores 0.1, and
    a miss scores 0. Unlike Recall@k it rewards putting the answer FIRST
    rather than merely somewhere in the list.
    """
    r1 = r5 = r10 = 0
    rr = 0.0
    for ranked, gold in zip(rankings, relevant):
        # Accept either a single gold index or a relevance set.
        gold_set = gold if isinstance(gold, (set, frozenset)) else {gold}

        if gold_set & set(ranked[:1]):
            r1 += 1
        if gold_set & set(ranked[:5]):
            r5 += 1
        if gold_set & set(ranked[:10]):
            r10 += 1
        # MRR uses the FIRST relevant hit — rank 1 scores 1.0, rank 4 scores
        # 0.25, no hit scores 0.
        for pos, doc in enumerate(ranked, start=1):
            if doc in gold_set:
                rr += 1.0 / pos
                break
    n = len(relevant)
    return RetrievalResult(method, r1 / n, r5 / n, r10 / n, rr / n, n, median_ms)


def results_table(results: list[RetrievalResult]):
    import pandas as pd
    return pd.DataFrame([{
        "method": r.method,
        "Recall@1": round(r.recall_at_1, 4),
        "Recall@5": round(r.recall_at_5, 4),
        "Recall@10": round(r.recall_at_10, 4),
        "MRR": round(r.mrr, 4),
        "median ms": round(r.median_ms, 2),
        "queries": r.n_queries,
    } for r in results])
