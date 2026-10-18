# Stage 5 — semantic search + retrieval benchmark

This is the piece that earns **semantic search, embeddings, vector database**
on your CV. It reuses the 7,027 sections you already extracted; nothing is
re-downloaded.

---

## 1 · Install

With the venv active, in `X:\Projects\edgar-credit-signal`:

```powershell
pip install -r requirements-stage5.txt
```

`torch` is the big one. If it crawls:

```powershell
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-stage5.txt
```

Check it landed:

```powershell
python -c "import torch, faiss, sentence_transformers; print('ok')"
```

---

## 2 · Run the tests first

```powershell
pytest test_retrieval.py -q
```

Check: `34 passed`, about 2 seconds, no model download. They cover BM25
arithmetic against a reference implementation, rank fusion, the metrics, the
relevance sets, and nine separate chunking and windowing controls. If these pass, the
machinery is sound before you spend an hour embedding.

---

## 3 · Trial index — 500 sections

```powershell
python build_index.py --sample 500
```

The model downloads on first run (~90 MB). Expect roughly 20 minutes.

The first line printed is `stage 5 code version 5.5 (MAX_CHARS=850)`.
If it says anything else, you are running stale code.

Three lines to check, in this order:

```
heading-leak repair: substring pass removed N further occurrences
heading-leak repair: removed N leaked heading fragments from M passages ...
heading-leak control: 0 of 300 sampled headings still appear inside a passage body (must be 0)
token-window split: N passages exceeded 254 tokens, split into N extra pieces
query filter: dropped N section dividers and bullet fragments
token-window control: content limit 254 tokens, ... 0 over the limit (0.00%)

  sample evaluation queries
    - <ten random queries>
```

- **repair** — how much damage there was. A non-zero N is expected and fine;
  it is the fix doing its job. Note the number, it goes in the writeup.
- **control** — must be **0**. This is checked by a different matching rule
  than the repair uses, so agreement means something.
- **token window** — the split line says how many needed re-splitting; the
  control line must then read **0 over the limit (0.00%)**. Anything else means
  passages are being cut before encoding while BM25 still indexes every word,
  so the dense side is judged on text it was never shown.
- **sample queries** — read all ten. They should look like sentences an
  analyst would type. Page furniture used to reach this list (`12 Table of
  Contents`); if anything like that appears, tell me before running the real
  build.

If control is not 0 or truncation is above 1%, stop and send me the lines.

---

## 4 · The benchmark index — 1,000 sections

```powershell
python build_index.py --sample 1000
```

**Use `--sample 1000`, not the full corpus.** Your CPU embeds about 28-40
passages a second; the full 7,027 sections is roughly 730,000 passages, which
is four hours or more. A thousand filings gives you ~105,000 passages and
several thousand evaluation queries — a completely respectable IR benchmark,
in about 50 minutes. Say "benchmarked on a 1,000-filing subsample" and no one
will blink.

Outputs land in `data\processed\`:

| file | what it is |
|---|---|
| `chunks.parquet` | every passage, with cik / ticker / year / heading |
| `chunk_embeddings.npy` | the 384-dim vectors |
| `heading_queries.parquet` | the free evaluation set |
| `bm25.pkl` | the lexical index |
| `index_meta.json` | throughput, device, dimensions — CV numbers live here |

---

## 5 · Run the benchmark

Open `05_retrieval_benchmark.ipynb`, pick the `.venv` kernel, Run All.

It reports **two** tables, and the gap between them is itself a finding:

- **strict** — one gold passage per query
- **relevance sets** — near-duplicates count as correct

Risk-factor boilerplate is copied across companies. Under strict scoring a
system that returns a perfectly relevant passage is marked wrong for
returning the wrong *copy* of it. On the synthetic corpus used to develop
this, the same rankings scored **Recall@1 = 0.005 strict and 0.930 against
relevance sets**. Report the relevance-set numbers as the headline and the
strict numbers as the diagnostic.

---

## 6 · What to actually look at

The most valuable cell is the last one — the disagreements. Read four by
hand. You want to be able to say in an interview, from memory:

> *"Dense won on queries where the heading and the body shared no vocabulary
> — 'supply chain disruption' finding 'our vendors may fail to deliver'.
> BM25 won on ones with a specific technical term, because the embedding
> model has barely seen 'securitisation' and treats it as noise."*

That sentence is worth more than the table.

---

## 7 · Optional — the full corpus on Kaggle

Embedding is the only GPU-bound step in this project. Upload
`data\interim\sections.parquet` as a private Kaggle dataset, run
`build_index.py` in a GPU notebook, download `chunk_embeddings.npy` and
`chunks.parquet` back. Roughly ten times faster, so the full corpus becomes
~25 minutes instead of four hours. Not required — the sampled CPU run
produces a valid benchmark on its own.

---

## Known limitation, state it if asked

The dense path was written and unit-tested against synthetic vectors, but the
embedding model itself could not be downloaded in the environment where the
code was written. If something breaks it will be at the
`SentenceTransformer(...)` line, and it will be a download or a torch install
rather than the retrieval logic — that part is covered by the tests.
