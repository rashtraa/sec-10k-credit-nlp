"""
Stage 2b — turn extracted sections into features.

Two families:

  1. LOUGHRAN-McDONALD CATEGORY RATES. Counts of Negative, Uncertainty,
     Litigious, Constraining and Modal words, as a share of total words.
     Loughran and McDonald built these lists because general-purpose
     sentiment dictionaries misread financial text: in Harvard-IV, words
     like "liability", "tax", "cost", "capital" and "board" are negative,
     when in a 10-K they are just accounting vocabulary. Using a finance
     dictionary is a defensible choice with a citable reason, which is
     worth more in an interview than a fancier model.

  2. YEAR-OVER-YEAR RISK-FACTOR DISSIMILARITY. Cosine distance between a
     firm's Item 1A this year and its own Item 1A last year, on TF-IDF
     vectors. Risk factor sections are heavily boilerplate and are largely
     copied forward, so a firm that rewrites a lot of it has done something
     unusual. This is the headline feature, and note it is *within-firm* —
     it asks how much this company changed relative to itself, not how it
     compares to others, so stable cross-firm differences in writing style
     difference out.

Run:  python text_features.py
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import INTERIM, PROCESSED

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("features")

SECTIONS = INTERIM / "sections.parquet"
OUT = PROCESSED / "text_features.parquet"

USABLE = ("ok", "ok_loose")   # ok_loose = found via the fallback anchor tier


LM_CATEGORIES = ["Negative", "Positive", "Uncertainty", "Litigious",
                 "Constraining", "Modal"]

TOKEN = re.compile(r"[A-Za-z']+")


def load_lm() -> dict[str, set[str]]:
    """
    Load the Loughran-McDonald master dictionary.

    It ships inside pysentiment2, so there is no manual download and no
    dependency on a hosted file that might move.
    """
    try:
        import pysentiment2 as ps
    except ImportError:
        raise SystemExit("pip install pysentiment2")

    path = Path(ps.__file__).parent / "static" / "LM.csv"
    lm = pd.read_csv(path)
    lm["Word"] = lm["Word"].astype(str).str.upper()

    cats: dict[str, set[str]] = {}
    for c in LM_CATEGORIES:
        # A non-zero entry is the year the word entered that category, not a
        # flag — any non-zero value means membership.
        cats[c] = set(lm.loc[lm[c] != 0, "Word"])
        log.info("LM %-13s %6d words", c, len(cats[c]))
    return cats


def count_categories(text: str, cats: dict[str, set[str]]) -> dict:
    words = [w.upper() for w in TOKEN.findall(text)]
    n = len(words)
    out = {"n_words": n}
    if n == 0:
        for c in LM_CATEGORIES:
            out[f"lm_{c.lower()}"] = np.nan
        return out
    # One pass over the tokens, membership-testing against each set.
    from collections import Counter
    counts = Counter(words)
    for c in LM_CATEGORIES:
        s = cats[c]
        hits = sum(v for w, v in counts.items() if w in s)
        out[f"lm_{c.lower()}"] = hits / n
    return out


def yoy_dissimilarity(df: pd.DataFrame) -> pd.Series:
    """
    Cosine distance between each firm-year's Item 1A and its own prior year.

    The vectorizer is fit on the whole corpus so that IDF weights are
    comparable across firms; the comparison itself is strictly within-firm
    and strictly backward in time.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    usable = df["item1a_status"].isin(USABLE)
    log.info("fitting TF-IDF on %s usable Item 1A sections", f"{usable.sum():,}")

    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=50_000,
        min_df=5,
        sublinear_tf=True,
    )
    X = vec.fit_transform(df.loc[usable, "item1a"])
    log.info("TF-IDF matrix: %s x %s", *X.shape)

    # Map row position in X back to the dataframe index.
    pos = {idx: i for i, idx in enumerate(df.index[usable])}

    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, grp in df.groupby("cik", sort=False):
        grp = grp.sort_values("filing_year")
        prev_idx, prev_year = None, None
        for idx, row in grp.iterrows():
            if row["item1a_status"] not in USABLE:
                prev_idx, prev_year = None, None
                continue
            if prev_idx is not None and row["filing_year"] - prev_year == 1:
                a, b = X[pos[idx]], X[pos[prev_idx]]
                # Rows are L2-normalised by TfidfVectorizer, so the dot
                # product IS the cosine similarity.
                #
                # Clipping is not cosmetic. For a filing copied forward
                # verbatim the dot product is 1 + 2e-16 rather than exactly
                # 1, so `1 - sim` lands at -2e-16. Left alone that is a
                # negative "distance", it breaks the [0,1] sanity gate, and
                # it would put a nonsensical value into a regressor. The
                # bound is a mathematical fact about non-negative unit
                # vectors, so enforcing it discards no information.
                sim = min(1.0, max(0.0, float(a.multiply(b).sum())))
                out.at[idx] = 1.0 - sim
            prev_idx, prev_year = idx, row["filing_year"]

    # Belt and braces. The per-pair clip above handles the usual case, but
    # accumulated float error in the sparse dot product can still leave a
    # value at -1e-17, which prints as "-0.0000" and fails a [0,1] gate
    # while looking like zero. The bound is a mathematical fact about
    # non-negative unit vectors, so enforcing it discards no information.
    n_out = int((~out.between(0.0, 1.0) & out.notna()).sum())
    if n_out:
        log.info("clipped %d value(s) outside [0,1] (float error)", n_out)
    return out.clip(0.0, 1.0)


def main() -> None:
    if not SECTIONS.exists():
        raise SystemExit("run extract_sections.py first")
    df = pd.read_parquet(SECTIONS)
    log.info("loaded %s filings", f"{len(df):,}")

    cats = load_lm()

    log.info("counting LM categories in Item 1A")
    feats = pd.DataFrame(
        [count_categories(t if ok in USABLE else "", cats)
         for t, ok in zip(df["item1a"], df["item1a_status"])],
        index=df.index,
    )
    df = pd.concat([df.drop(columns=["item1a", "item7"]), feats], axis=1)

    # Recompute against the text before we dropped it.
    src = pd.read_parquet(SECTIONS, columns=["cik", "filing_year",
                                             "item1a", "item1a_status"])
    df["risk_change"] = yoy_dissimilarity(src)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    keep = [c for c in df.columns if not c.startswith("item1a_anchors")]
    df[keep].to_parquet(OUT, index=False)

    # ---- report -----------------------------------------------------------
    rc = df["risk_change"].dropna()
    print(f"\n{'='*62}\nTEXT FEATURES   ({len(df):,} filings)\n{'='*62}")
    print(f"\nrisk_change (1 - cosine similarity vs own prior year)")
    print(f"    observations : {len(rc):,}")
    print(f"    mean         : {rc.mean():.4f}")
    print(f"    p10 / median / p90 : {rc.quantile(.10):.4f} / "
          f"{rc.median():.4f} / {rc.quantile(.90):.4f}")
    print(f"    min / max    : {rc.min():.4f} / {rc.max():.4f}")

    print("\nLoughran-McDonald rates (share of words in Item 1A)")
    for c in LM_CATEGORIES:
        s = df[f"lm_{c.lower()}"].dropna()
        if len(s):
            print(f"    {c:<13} mean={s.mean():.5f}  median={s.median():.5f}")

    print("\nsanity checks")
    print(f"    risk_change in [0,1]        : "
          f"{bool(rc.between(0, 1).all())}")
    print(f"    near-zero change (<0.05)    : {int((rc < 0.05).sum()):,}"
          f"   <- filings copied forward almost verbatim")
    print(f"    near-total rewrite (>0.80)  : {int((rc > 0.80).sum()):,}"
          f"   <- inspect a few of these by hand")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
