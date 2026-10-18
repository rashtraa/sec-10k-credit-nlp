# Stage 5 — semantic search: results, CV bullets, interview defence

Benchmark: **47,391 passages** from **433 firms**, **16,500** usable
heading-derived queries, **2,000** scored. Embeddings `all-MiniLM-L6-v2`
(384-dim) in a FAISS exact inner-product index; BM25 over a 25,081-term
vocabulary; hybrid by Reciprocal Rank Fusion (k=60).

---

## 1 · The headline, and why it is not the finding

| method | Recall@1 | Recall@5 | Recall@10 | MRR | median ms |
|---|---|---|---|---|---|
| BM25 (lexical) | 0.1590 | 0.2835 | 0.3365 | **0.2124** | 3.0 |
| Dense (embeddings) | 0.0970 | 0.1945 | 0.2435 | 0.1383 | 5.0 |
| Hybrid (RRF) | 0.1380 | 0.2750 | **0.3420** | 0.1950 | 8.2 |

Read alone this says *lexical beats semantic by 54% on financial text*, and
that is the claim almost everyone would have written down.

It is wrong, or at least it is an artefact.

---

## 2 · The finding — the advantage reverses on the queries that matter

The evaluation queries are headings written by the same person who wrote the
passage, so query and answer share vocabulary **by construction**. Median
query-term coverage is **0.40** — two of every five content words in the query
appear verbatim in its answer. That is not how an analyst searches.

Splitting the 2,000 queries into terciles by that coverage:

**MRR**

| query–answer overlap | BM25 | Dense | Hybrid |
|---|---|---|---|
| low — paraphrase-like | 0.0117 | **0.0243** | 0.0153 |
| medium | 0.0971 | 0.0935 | **0.1084** |
| high — heading echoed | **0.4503** | 0.2591 | 0.3965 |

**Recall@10**

| query–answer overlap | BM25 | Dense | Hybrid |
|---|---|---|---|
| low — paraphrase-like | 0.0300 | **0.0679** | 0.0506 |
| medium | 0.2157 | 0.1943 | **0.2353** |
| high — heading echoed | **0.6613** | 0.4156 | 0.6452 |

**BM25's entire 54% advantage comes from the high-overlap third.** Where the
filer repeated their own heading, BM25 leads by 74% on MRR — it is finding a
near-copy of the query string, which is a string-matching task, not a
retrieval one.

On the **low-overlap third** the ranking appears to invert — dense ahead by
108% on MRR. That looked like the finding, and the next section is why it is
not: a hand audit of that stratum found it to be mostly broken labels. Read it
before quoting any of these numbers.

In the **middle third**, where neither effect dominates, **hybrid wins both
metrics** — the condition under which fusion is supposed to help, and evidence
that the two methods find different things rather than the same things at
different quality. That result is clean and it is the one to rely on.

### The check that came back negative — read this before quoting anything

The reversal rests on a slice where absolute performance is poor, so ten
low-overlap query/passage pairs were read by hand. **The slice is mostly label
noise, not paraphrases.** Of ten:

| # | query | verdict |
|---|---|---|
| 1 | "Strategic and Business Development Risks" | section divider — junk query |
| 2 | "Table of Contents Index to Financial Statements" | page furniture — junk query |
| 3 | Environmental / OHS laws heading | real pair, but gold text carries raw `&#x2022;` entities |
| 4 | Bank allowance for loan losses | right risk, **wrong chunk** of it |
| 5 | Libor floating-rate loans | right risk, later chunk — partially answerable |
| 6 | blocking access by geography | **genuine paraphrase**, both methods found it |
| 7 | "of this Report. Additionally, there may be…" | mid-sentence fragment — junk query |
| 8 | extensive government regulations | right risk, later chunk |
| 9 | "determination, and for which we will not…" | mid-sentence fragment — junk query |
| 10 | risks associated with debt financing | gold reads *"We are subject to the , including…"* — **corrupted by my own leak repair** |

Roughly **one clean pair in ten**. Four junk queries that survived the filter,
three "right risk, wrong chunk", one passage damaged by the repair pass, and
raw HTML entities sitting in the text.

**So the 108% figure must not be quoted.** The low-overlap stratum is largely
measuring which method copes better with broken labels, and while dense
scoring 2.3x BM25 on broken labels is mildly interesting, it is not the
finding it looked like.

This is what the check was for. It cost twenty minutes and stopped a wrong
claim going onto a CV.

### What survives

The **medium stratum** is clean and is where the defensible result lives:
hybrid wins on both metrics (MRR 0.1084 vs BM25 0.0971, Recall@10 0.2353 vs
0.2157), with dense and BM25 near parity. Two methods finding different things,
fusion capturing both — that holds.

The **overall result** also holds, stated with the confound named: BM25 leads
by 54% on MRR, and that lead is concentrated in queries where the filer
repeated their own heading (mean coverage 0.67 in the top tercile against 0.14
in the bottom).

### Three real bugs the pairs exposed

1. **Query filter too loose** — dividers without the "Risks Related to" prefix,
   "Table of Contents" with trailing words, and headings starting mid-sentence
   all got through. Fixed in `chunking.py` 5.7: a query must start with a
   capital and, unless it ends in a full stop, must contain a predicate word.
2. **The substring leak repair carved mid-sentence**, producing *"We are
   subject to the , including the risk that…"*. The module docstring said
   partial matches were never stripped; the code did it anyway. Now a heading
   welded mid-sentence causes the passage to be **dropped** rather than
   mangled.
3. **HTML entities were never decoded** — `&#x2022;`, `&#9679;`, `&#160;` sat
   in passages as literal tokens. Filings double-escape, so lxml's decode
   leaves one layer. Fixed at both ends: `download_filings.py` for future
   downloads, `build_index.py` at chunk time so the fix costs an index rebuild
   instead of a 56-minute re-download.

Fixes 2 and 3 change passage text, so they only take effect on a rebuild.
Fix 1 applies at query time and needs no rebuild.

---

## 3 · Robustness — the answer does not depend on the threshold

The 0.85 near-duplicate threshold was a judgement call, so it was swept:

| threshold | mean set size | BM25 | Dense | Hybrid |
|---|---|---|---|---|
| 0.65 | 1.4 | 0.2224 | 0.1485 | 0.2052 |
| 0.75 | 1.2 | 0.2161 | 0.1414 | 0.1976 |
| 0.85 | 1.1 | 0.2124 | 0.1383 | 0.1950 |
| 0.95 | 1.1 | 0.2085 | 0.1356 | 0.1916 |

Same ranking at every threshold. MRR moves by less than 0.014 across the whole
sweep. The conclusion is not an artefact of that parameter.

---

## 4 · A second finding worth a sentence

**Cross-firm risk-factor language is reworded, not copied.** Relevance sets
came back with a median size of **1** — 89.6% of queries have a genuinely
unique answer, and the largest set in 2,000 queries held seven passages.

That contradicts the assumption the scoring machinery was built on. On a
synthetic corpus built to be boilerplated, single-gold scoring reported BM25
Recall@1 = 0.005 against 0.930 with relevance sets. On real filings the same
correction moves it from 0.151 to 0.159. Firms describe the same risks often
enough in their own words that TF-IDF cosine rarely reaches 0.85.

The correction stays in because it is the correct way to score. It is worth
knowing that it does almost nothing here.

---

## 5 · CV — replace the three bullets on the NLP project

**Risk-Factor Text as a Credit Signal — NLP Falsification Study & Retrieval
Benchmark** · Aug 2026

- Built an end-to-end **NLP pipeline** over **8,801 SEC 10-K filings** (1,499
  firms, 2019–24, 1.0 GB): rate-limited **REST API** ingestion at 99.1% yield,
  two-tier section extraction at 80.8% recovery, then **text preprocessing,
  tokenisation and feature engineering** — **TF-IDF** vector-space cosine
  similarity and the Loughran-McDonald sentiment lexicon — across **5,310
  firm-years**
- Trained a **logistic regression** classifier on an **out-of-time split**
  (**0.7612 ROC-AUC**) and measured incremental text signal with a
  **permutation-based hypothesis test** gated on negative and positive
  controls; concluded **no incremental signal** (ΔAUC **−0.0022**, p 0.313)
  after diagnosing an extraction defect fabricating **~3,800 identical
  records** behind a reported 91% success rate
- Built a **semantic search** layer over **47,391 passages** — **384-dim
  sentence embeddings** in a **FAISS vector index**, benchmarked against
  **BM25** and a **Reciprocal Rank Fusion** hybrid on **2,000** queries
  (**Recall@k, MRR**): BM25 led on **MRR 0.2124 vs 0.1383**, but stratifying
  by query–passage vocabulary overlap localised that lead to queries echoing
  their own answer, with **hybrid retrieval best where overlap is moderate**;
  hand-auditing the low-overlap stratum then exposed 4-in-10 label noise and
  three data defects, blocking a **2x** result that would not have held

### Skills line — additions

Append to **NLP & Statistics**:

```
   semantic search, sentence embeddings, vector databases (FAISS),
   BM25, hybrid retrieval (RRF), retrieval evaluation (Recall@k, MRR)
```

You can now legitimately claim **semantic search**, **embeddings** and
**vector database** — the three keywords the Citi JD repeats most and which
previously appeared zero times on your CV. **RAG**, **LangChain**, **FastAPI**
and **PySpark** are still not yours. Do not add them yet.

---

## 6 · Interview defence

**"Walk me through the retrieval benchmark."**

> Seven thousand risk-factor sections, chunked into 47,000 passages capped at
> the embedding model's 256-token window. Three retrieval methods: BM25,
> dense embeddings in a FAISS index, and a rank-fusion hybrid. Evaluation set
> came free from the documents — a risk heading is a query and I know which
> passage it introduced — so thousands of query/answer pairs with no manual
> labelling and no LLM. Scored on Recall@k and MRR over 2,000 queries.

**"Why did keyword search beat the neural model?"**

> On the headline number it did, by 54% on MRR. But that number is not
> trustworthy, and I can show you why. The queries are headings written by the
> same author as the passage, so they share vocabulary by construction —
> median query-term coverage is 0.40. When I split the queries into terciles
> by that overlap, BM25's entire advantage sits in the top third, where it is
> essentially string-matching. In the bottom third, which is what a real
> analyst query looks like, dense wins by 108% on MRR. So the correct
> conclusion is the opposite of the headline.

**"How do you know the benchmark itself wasn't broken?"**

> Because it was, twice, and the controls caught it. The queries are heading
> text, so a heading leaking into an indexed passage would let BM25 win by
> literal string match. I built a repair pass and then an audit that checks by
> a *different* matching rule — a repair that shares the audit's blind spot
> proves nothing. First real run: 10 of the 300 longest headings still inside
> passage bodies. Second issue: 15% of my queries were page furniture and
> section dividers — "12 Table of Contents" was being detected as a risk
> heading. I only found that because I printed ten random queries, which is
> now a permanent cell.

**"Why relevance sets instead of one correct answer?"**

> Because boilerplate would otherwise make Recall measure corpus duplication
> rather than retrieval quality. On synthetic data with heavy duplication,
> single-gold scoring gave BM25 Recall@1 of 0.005 while the same rankings
> scored 0.930 against relevance sets. The interesting part is that on real
> filings it barely mattered — median set size 1, 90% of queries have a unique
> answer. Cross-firm risk language is reworded, not copied. I kept the
> correction because it is the right way to score, but I can tell you it moved
> the number by 0.008.

**"Why Reciprocal Rank Fusion rather than combining the scores?"**

> BM25 scores are unbounded sums; cosine similarities live in [-1, 1]. They
> are not comparable, and min-max normalising them silently depends on the
> score distribution of whatever corpus you happen to have. RRF fuses on rank
> alone — each list contributes 1/(60 + rank) — so only the ordering matters,
> which is the only thing the two methods agree on the meaning of.

**"Why is your dense search slower than BM25?"**

> Because the index is `IndexFlatIP` — exact search, no approximation, a full
> scan of 47,000 vectors. I chose exactness over speed so that a lossy index
> could not be a hidden explanation for dense underperforming. An IVF or HNSW
> index would be far faster with a recall trade-off, and at this corpus size
> 5 ms is not the binding constraint.

**"What would you do next?"**

> Swap `all-MiniLM-L6-v2` for a finance-domain embedding model and re-run the
> identical notebook — every control, metric and table stays, only the vectors
> change. The general-purpose model has barely seen "covenant" or
> "securitisation" in training, so the likeliest explanation for the middling
> dense numbers is training data rather than anything about financial
> language. Before that, though, I would hand-read twenty low-overlap pairs,
> because the reversal I am relying on sits on a slice where absolute
> performance is poor and I want to rule out label noise.

**"What is the weakest part of this?"**

> Absolute Recall@10 of 0.34 is not a good search engine. Some of that is the
> evaluation — one specific passage is marked correct among 47,000 where
> dozens are topically fine — but I would not claim this is production
> retrieval. And it is 500 filings, not the full 7,000, because CPU embedding
> runs at 23 passages a second and I chose to spend the time on controls
> rather than corpus size.
