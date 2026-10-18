"""
Stage 5c — chunk the sections, embed them, and build both search indexes.

Two things happen here that determine everything downstream.

CHUNKING turns 7,027 documents into a few hundred thousand passages, which is
the unit a retrieval system actually returns.

EMBEDDING runs each passage through a neural network that maps text to a
384-dimensional vector, positioned so that passages with similar MEANING land
near each other. That is what makes semantic search possible: the query "supply
chain disruption" can find "our vendors may fail to deliver components" with
no shared vocabulary at all.

The model is `all-MiniLM-L6-v2` — a 6-layer transformer, ~90 MB, trained on a
billion sentence pairs. It runs on a CPU. On a GPU (Kaggle gives you one free)
it is roughly ten times faster.

Run:  python build_index.py            # all of it
      python build_index.py --sample 500   # 500 filings, for a quick trial
"""
from __future__ import annotations

import argparse
import json
import logging
import html
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunking import (MAX_CHARS, MIN_CHARS, STAGE_VERSION, chunk_section,
                      heading_queries, is_usable_query)
from config import INTERIM, PROCESSED

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("index")

USABLE = ("ok", "ok_loose")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNKS_PARQUET = PROCESSED / "chunks.parquet"
EMBEDDINGS_NPY = PROCESSED / "chunk_embeddings.npy"
QUERIES_PARQUET = PROCESSED / "heading_queries.parquet"
BM25_PICKLE = PROCESSED / "bm25.pkl"
META_JSON = PROCESSED / "index_meta.json"


_FRAG = re.compile(r"(\n+|(?<=[.!?])\s+)")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def strip_leaked_headings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove heading text that ended up inside an indexed passage.

    WHY THIS IS THE CONTROL THE BENCHMARK RESTS ON
    ----------------------------------------------
    The evaluation queries ARE heading text, and passages are supposed to
    contain the explanation only. A heading sitting inside a passage body is
    therefore a passage that can be retrieved by matching a literal string —
    which hands the contest to keyword search and makes the whole
    lexical-versus-semantic comparison measure nothing.

    It happens for a mundane reason. Heading detection needs paragraph
    boundaries, and whether a filing has any depends on how its HTML was
    authored. A filer whose paragraphs are `<div>`s with no spacing arrives as
    one enormous run-on block; `_relines` in `chunking.py` repairs most of
    those, but not all. Whatever survives is caught here.

    WHY STRIP RATHER THAN DROP
    --------------------------
    Dropping the passage loses real risk-factor text. Dropping the query
    shrinks the evaluation set and leaves the string still sitting in the
    index for some *other* near-identical query to match. Removing the
    heading fragment applies exactly the same rule every other heading gets —
    held out of the index — and leaves the explanation intact.

    Matching is on whole sentence/line fragments, normalised for whitespace
    and case, over windows of one and two fragments (a few headings run to two
    sentences). Partial matches are deliberately not stripped: cutting a
    passage mid-sentence would do more damage than the leak.
    """
    heads = {_norm(h) for h in df["heading"] if h}
    heads = {h for h in heads if len(h) >= 20}
    if not heads:
        return df

    new_texts, examples = [], []
    n_frag = n_chunks = 0

    for text in df["text"]:
        parts = _FRAG.split(text)
        content = list(range(0, len(parts), 2))
        drop: set[int] = set()
        for a in range(len(content)):
            if content[a] in drop:
                continue
            for w in (2, 1):                     # prefer the longer match
                if a + w > len(content):
                    continue
                lo, hi = content[a], content[a + w - 1]
                window = "".join(parts[lo:hi + 1])
                if _norm(window) in heads:
                    drop.update(range(lo, hi + 1))
                    n_frag += 1
                    if len(examples) < 3:
                        examples.append((_norm(window)[:90],
                                         _norm(text)[:110]))
                    break
        if drop:
            n_chunks += 1
            rebuilt = "".join(p for i, p in enumerate(parts) if i not in drop)
            new_texts.append(re.sub(r"\n{3,}", "\n\n", rebuilt).strip())
        else:
            new_texts.append(text)

    df = df.copy()
    df["text"] = new_texts

    n_sub = _substring_leak_pass(df)
    df["n_chars"] = df["text"].str.len()

    before = len(df)
    df = df[df["n_chars"] >= MIN_CHARS].reset_index(drop=True)

    log.info("heading-leak repair: substring pass removed %s further "
             "occurrences", f"{n_sub:,}")
    log.info("heading-leak repair: removed %s leaked heading fragments from "
             "%s passages (%.2f%% of the corpus); %s passages fell below the "
             "length floor and were dropped",
             f"{n_frag:,}", f"{n_chunks:,}", 100 * n_chunks / max(before, 1),
             f"{before - len(df):,}")
    for h, t in examples:
        log.info("  example leak | %r", h)
        log.info("       in body | %r...", t)
    return df


def _substring_leak_pass(df: pd.DataFrame) -> int:
    """
    Second pass: remove headings buried *inside* a sentence.

    WHY THE FIRST PASS MISSES THESE
    -------------------------------
    The fragment pass splits a passage on sentence and line boundaries and
    matches whole fragments. That catches a heading that sits as its own
    sentence, which is most of them. It cannot catch a heading with no
    terminal punctuation — 10-K filers routinely write
    `<p><b>Supply chain risk</b> Our operations rely on...</p>`, which comes
    out of the converter as ONE sentence with the heading welded to the front
    of it. Measured: 9 of the 300 longest headings survived pass one that way.

    So this pass matches by raw substring — deliberately the SAME rule the
    audit uses, so the audit cannot pass by agreeing with a repair that shares
    its blind spot.

    WHY IT IS NOT O(headings x passages)
    ------------------------------------
    A naive scan is 15,000 headings x 100,000 passages of substring search.
    Instead each heading is reduced to its RAREST word, and only passages
    containing that word are candidates — usually a handful. The inverted
    index that makes this possible costs one CountVectorizer pass.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    heads = sorted({h for h in df["heading"] if h and len(h) >= 20},
                   key=len, reverse=True)
    if not heads:
        return 0

    tok = re.compile(r"[a-z0-9']+")
    vec = CountVectorizer(lowercase=True, binary=True,
                          token_pattern=r"[a-z0-9']+")
    X = vec.fit_transform(df["text"]).tocsc()
    vocab = vec.vocabulary_
    doc_freq = np.diff(X.indptr)

    texts = df["text"].tolist()
    lowers = [t.lower() for t in texts]
    removed = 0
    mangled: set[int] = set()

    def _clean_cut(text: str, pos: int, n: int) -> bool:
        """
        Would removing text[pos:pos+n] leave a readable passage?

        Only if the match sits on a boundary. Carving a heading out of the
        MIDDLE of a sentence leaves wreckage — one real passage came back as
        "We are subject to the , including the risk that the cash provided by
        our operating activities will be insufficient". That is worse than the
        leak it was fixing, and it was the module docstring's stated policy
        all along; the code just did not implement it.
        """
        head_ok = pos == 0 or text[max(0, pos - 2):pos].strip()[-1:] in (".", "!", "?", "")
        tail = pos + n
        tail_ok = tail >= len(text) or text[tail:tail + 2].strip()[:1] in ("", *"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        return head_ok and tail_ok

    for h in heads:
        hl = h.lower()
        terms = [vocab[t] for t in set(tok.findall(hl)) if t in vocab]
        if not terms:
            continue
        col = min(terms, key=lambda c: doc_freq[c])
        for i in X.indices[X.indptr[col]:X.indptr[col + 1]]:
            # lower() can change length on a few exotic characters; skip those
            # rather than slice at a wrong offset.
            if len(lowers[i]) != len(texts[i]):
                continue
            # A heading can appear more than once in one passage.
            pos = lowers[i].find(hl)
            while pos >= 0:
                if not _clean_cut(texts[i], pos, len(hl)):
                    # Cannot be removed without mangling the sentence, so the
                    # passage goes instead. Losing one passage is cheaper than
                    # keeping a corrupted one AND cheaper than leaving a
                    # string-matchable heading in the index.
                    mangled.add(i)
                    break
                texts[i] = (texts[i][:pos] + texts[i][pos + len(hl):]).strip()
                lowers[i] = texts[i].lower()
                removed += 1
                pos = lowers[i].find(hl)

    df["text"] = texts
    if mangled:
        log.info("heading-leak repair: dropped %s passages where the heading "
                 "was welded mid-sentence and could not be cut cleanly",
                 f"{len(mangled):,}")
        df.loc[sorted(mangled), "text"] = ""      # falls below MIN_CHARS, dropped
    return removed


def audit_heading_leak(df: pd.DataFrame, n_heads: int = 300) -> int:
    """
    Independent check that the repair worked, by a different method.

    The repair matches whole fragments; this looks for the heading anywhere
    inside a body as a raw substring. Agreeing answers from two different
    matching rules is worth more than one rule checking itself.

    Sampled on both sides — an exhaustive cross-product is 300k x 300k
    substring scans, and this only needs to detect whether the failure mode is
    still present.
    """
    heads = sorted({h for h in df["heading"] if h}, key=len, reverse=True)[:n_heads]
    probe = df["text"].sample(min(20_000, len(df)), random_state=20260821)
    leaked = sum(int(probe.str.contains(h, regex=False, case=False).any())
                 for h in heads)
    log.info("heading-leak control: %d of %d sampled headings still appear "
             "inside a passage body (must be 0)", leaked, len(heads))
    if leaked:
        log.warning("BENCHMARK NOT VALID — keyword search can win by string "
                    "match on %d queries. Do not report these numbers.", leaked)
    return leaked


def build_chunks(sample: int | None = None) -> pd.DataFrame:
    log.info("stage 5 code version %s  (MAX_CHARS=%d)", STAGE_VERSION, MAX_CHARS)
    sec = pd.read_parquet(INTERIM / "sections.parquet")
    sec = sec[sec["item1a_status"].isin(USABLE)]
    if sample:
        sec = sec.sample(min(sample, len(sec)), random_state=20260821)
    # ---- decode HTML entities ------------------------------------------
    #
    # The converter emitted `&#x2022;`, `&#9679;` and `&#160;` as literal text,
    # so bullets and non-breaking spaces sat in passages as seven-character
    # tokens. Found by reading ten low-overlap query/passage pairs by hand —
    # no automated control was looking for it, because nothing had told me to
    # look. Decoding here rather than at download time so the fix costs an
    # index rebuild instead of a 56-minute re-download.
    before_ent = int(sec["item1a"].str.contains(r"&#\d|&#x[0-9a-f]|&[a-z]{2,6};",
                                                regex=True, case=False).sum())
    sec = sec.copy()
    sec["item1a"] = (sec["item1a"].map(html.unescape)
                     .str.replace("\xa0", " ", regex=False))
    after_ent = int(sec["item1a"].str.contains(r"&#\d|&#x[0-9a-f]",
                                               regex=True, case=False).sum())
    log.info("entity decode: %s of %s sections carried HTML entities, "
             "%s still do after decoding", f"{before_ent:,}", f"{len(sec):,}",
             f"{after_ent:,}")

    log.info("chunking %s sections", f"{len(sec):,}")

    rows = []
    for r in tqdm(sec.itertuples(index=False), total=len(sec), desc="chunking"):
        for c in chunk_section(r.item1a, int(r.cik),
                               str(r.ticker), int(r.filing_year)):
            rows.append(c.__dict__)

    df = pd.DataFrame(rows).reset_index(drop=True)
    # chunk_id was assigned per-section; make it globally unique.
    df["chunk_id"] = df["cik"].astype(str) + "_" + df["filing_year"].astype(str) \
                     + "_" + df.groupby(["cik", "filing_year"]).cumcount().astype(str)
    log.info("%s chunks from %s sections (%.1f per section)",
             f"{len(df):,}", f"{len(sec):,}", len(df) / max(len(sec), 1))

    # ---- deduplicate, or the benchmark measures the wrong thing ----------
    #
    # Risk-factor boilerplate is copied not only forward in time by one firm
    # but ACROSS firms — law firms reuse language, and "we depend on a
    # limited number of suppliers" is written near-identically by hundreds of
    # companies.
    #
    # That breaks the evaluation rather than the search. Each query has
    # exactly one passage marked correct. If forty near-identical passages
    # exist, a system that returns a perfectly relevant one is scored as
    # WRONG thirty-nine times out of forty. Recall would then measure how
    # duplicated the corpus is, not how good retrieval is.
    #
    # Exact duplicates are removed outright. Near-duplicates are handled at
    # query-selection time in the benchmark, which is stricter but needs the
    # embeddings, so it happens there rather than here.
    df = strip_leaked_headings(df)
    audit_heading_leak(df)
    return df


def dedupe_exact(df: pd.DataFrame) -> pd.DataFrame:
    norm = (df["text"].str.lower()
            .str.replace(r"[^a-z0-9 ]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())
    before = len(df)
    df = df[~norm.duplicated(keep="first")].reset_index(drop=True)
    log.info("removed %s exact-duplicate passages (%.1f%%), %s remain",
             f"{before - len(df):,}", 100 * (before - len(df)) / before,
             f"{len(df):,}")
    return df.reset_index(drop=True)


def load_model():
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("loading %s on %s", MODEL_NAME, device)
    return SentenceTransformer(MODEL_NAME, device=device), device


def split_to_token_limit(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Re-split any passage that still overflows the model's token window.

    WHY A CHARACTER CEILING IS NOT ENOUGH
    -------------------------------------
    Chunking caps passages at 850 characters, chosen from a ~4 characters per
    token rule of thumb against a 256-token window. That rule is an average,
    and filing prose is not average everywhere: tables, dollar figures, ticker
    symbols, defined terms in ALL CAPS and legal citations all tokenise far
    more densely. Measured on the real corpus, 1.9% of passages still hit the
    cap at 850 characters — silently losing text that BM25 still indexes.

    So the limit is enforced in the unit that actually matters. The tokenizer
    reports where each token starts in the original string, so an over-long
    passage is cut at the character offset of the last token that fits, on the
    nearest word boundary.

    The first piece keeps the heading; continuation pieces get none. A
    continuation is not what the heading introduced, so letting it generate an
    evaluation query would create a query whose "correct" answer is the wrong
    half of a passage.
    """
    tok = model.tokenizer
    limit = model.max_seq_length - 2          # room for [CLS] and [SEP]

    lengths = [len(x) for x in tok(df["text"].tolist(),
                                   add_special_tokens=False,
                                   truncation=False)["input_ids"]]
    over = [i for i, n in enumerate(lengths) if n > limit]
    if not over:
        log.info("token-window split: no passage exceeds %d tokens", limit)
        return df

    rows = df.to_dict("records")
    extra, n_pieces = [], 0
    for i in over:
        text = rows[i]["text"]
        pieces = []
        while True:
            enc = tok(text, add_special_tokens=False, truncation=False,
                      return_offsets_mapping=True)
            if len(enc["input_ids"]) <= limit:
                pieces.append(text)
                break
            cut = enc["offset_mapping"][limit][0]
            space = text.rfind(" ", 0, cut)
            if space > cut // 2:
                cut = space
            pieces.append(text[:cut].strip())
            text = text[cut:].strip()
            if not text:
                break

        rows[i]["text"] = pieces[0]
        for k, piece in enumerate(pieces[1:], start=1):
            if len(piece) >= MIN_CHARS:
                extra.append({**rows[i], "text": piece, "heading": "",
                              "chunk_id": f"{rows[i]['chunk_id']}s{k}"})
        n_pieces += len(pieces) - 1

    out = pd.DataFrame(rows + extra)
    out["n_chars"] = out["text"].str.len()
    out = out[out["n_chars"] >= MIN_CHARS].reset_index(drop=True)
    log.info("token-window split: %s passages exceeded %d tokens, split into "
             "%s extra pieces; corpus now %s passages",
             f"{len(over):,}", limit, f"{n_pieces:,}", f"{len(out):,}")
    return out


def embed(texts: list[str], model, device: str,
          batch_size: int = 256) -> tuple[np.ndarray, str, float]:
    """Returns (embeddings, device used, seconds taken)."""
    # ---- control: is any passage being silently truncated? ---------------
    #
    # The model takes 256 tokens and cuts the rest without warning. BM25
    # indexes every word. If passages overflow the window, the two systems are
    # not being shown the same text and the comparison is not a comparison.
    #
    # Checked with the real tokenizer on a sample rather than trusted to the
    # chars-per-token rule of thumb that set MAX_CHARS.
    # MEASURE WITHOUT TRUNCATION. The first version of this control counted a
    # passage as truncated whenever its encoded length REACHED max_seq_length —
    # but a passage that exactly fills the window is not truncated, it is full.
    # Since the truncating tokenizer caps every over-long input at exactly that
    # number, the two are indistinguishable from the encoded output alone. The
    # control was reporting 0.50% truncation on a corpus where nothing had been
    # cut: those were the pieces `split_to_token_limit` had just cut TO the
    # limit. Tokenising without truncation and comparing to the real limit is
    # the only way to tell the difference.
    probe = texts[:5000]
    limit = model.max_seq_length - 2               # [CLS] and [SEP]
    lens = [len(x) for x in model.tokenizer(probe, add_special_tokens=False,
                                            truncation=False)["input_ids"]]
    over = sum(1 for n in lens if n > limit)
    log.info("token-window control: content limit %d tokens, sampled %s "
             "passages, median %d, p99 %d, max %d, %d over the limit (%.2f%%)",
             limit, f"{len(probe):,}", int(np.median(lens)),
             int(np.percentile(lens, 99)), int(max(lens)),
             over, 100 * over / max(len(probe), 1))
    if over:
        log.warning("%d passages would be truncated — split_to_token_limit "
                    "should have made this impossible", over)

    t0 = time.time()
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    dt = time.time() - t0
    log.info("embedded %s passages in %.0fs (%.0f/sec) on %s",
             f"{len(texts):,}", dt, len(texts) / dt, device)
    return emb.astype("float32"), device, dt


def main(sample: int | None = None) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # ---- 1 · chunk ------------------------------------------------------
    chunks = build_chunks(sample)

    # The model is loaded here rather than at embedding time because the
    # tokenizer is needed first, to enforce the token limit exactly.
    model, device = load_model()
    chunks = split_to_token_limit(chunks, model)
    chunks = dedupe_exact(chunks)
    chunks.to_parquet(CHUNKS_PARQUET, index=False)

    # ---- 2 · evaluation queries, for free -------------------------------
    # A risk heading IS a query, and we know which passage it introduced.
    # The heading text is NOT part of the indexed passage, so a system has to
    # match meaning rather than find a repeated string.
    q = chunks[chunks["heading"].str.len() > 0][
        ["heading", "chunk_id", "cik", "filing_year"]
    ].rename(columns={"heading": "query", "chunk_id": "relevant_chunk_id"})

    # Not every heading is a retrieval task. Section dividers ("Risks Related
    # to Our Organization and Structure:") name a GROUP of risks, so no single
    # passage answers them; bullet fragments are clauses lifted out of a list.
    # Both were found by reading ten random queries off the real corpus.
    before_q = len(q)
    q = q[q["query"].map(is_usable_query)].reset_index(drop=True)
    log.info("query filter: dropped %s section dividers and bullet fragments "
             "(%.1f%%), %s queries remain",
             f"{before_q - len(q):,}", 100 * (before_q - len(q)) / max(before_q, 1),
             f"{len(q):,}")
    q.to_parquet(QUERIES_PARQUET, index=False)
    log.info("%s heading-derived queries (%.0f%% of chunks have a heading)",
             f"{len(q):,}", 100 * len(q) / len(chunks))

    # Eyeball control. Page furniture used to reach this list ("12 Table of
    # Contents"), which is a meaningless retrieval task. Every line below
    # should read like something an analyst would actually type.
    print("\n  sample evaluation queries")
    for text in q["query"].sample(min(10, len(q)),
                                  random_state=20260821):
        print(f"    - {text[:96]}")
    print()

    # ---- 3 · dense embeddings -------------------------------------------
    emb, device, dt = embed(chunks["text"].tolist(), model, device)
    np.save(EMBEDDINGS_NPY, emb)

    # ---- 4 · BM25 -------------------------------------------------------
    from retrieval import BM25, tokenize
    log.info("building BM25 index")
    t0 = time.time()
    bm25 = BM25([tokenize(t) for t in chunks["text"]])
    bm25_dt = time.time() - t0
    with open(BM25_PICKLE, "wb") as f:
        pickle.dump(bm25, f)
    log.info("BM25 built in %.0fs, %s unique terms", bm25_dt, f"{len(bm25.idf):,}")

    META_JSON.write_text(json.dumps({
        "model": MODEL_NAME, "device": device,
        "n_chunks": int(len(chunks)), "n_queries": int(len(q)),
        "embed_dim": int(emb.shape[1]),
        "embed_seconds": round(dt, 1),
        "embed_per_sec": round(len(chunks) / dt, 1),
        "bm25_seconds": round(bm25_dt, 1),
        "bm25_vocab": int(len(bm25.idf)),
        "sample": sample,
    }, indent=2))

    print(f"\n{'='*62}\nINDEX BUILT\n{'='*62}")
    print(f"  chunks           {len(chunks):>10,}")
    print(f"  eval queries     {len(q):>10,}")
    print(f"  embedding dim    {emb.shape[1]:>10,}")
    print(f"  device           {device:>10}")
    print(f"  embed throughput {len(chunks)/dt:>10,.0f} passages/sec")
    print(f"  BM25 vocabulary  {len(bm25.idf):>10,}")
    print(f"\n  chunk length  p10={chunks.n_chars.quantile(.1):,.0f}  "
          f"median={chunks.n_chars.median():,.0f}  "
          f"p90={chunks.n_chars.quantile(.9):,.0f}")
    print(f"\nwritten to {PROCESSED}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="use only N sections (quick trial)")
    a = ap.parse_args()
    main(a.sample)
