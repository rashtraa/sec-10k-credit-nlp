"""
Stage 2a — pull Item 1A and Item 7 out of every downloaded filing.

Writes data/interim/sections.parquet with one row per filing, carrying the
extracted text plus an explicit status for each section. The status column is
not bookkeeping: `absent` (the company legitimately omitted Item 1A, which
smaller reporting companies may do) and `tooshort` (we failed to find it)
mean different things, and pooling them would put parsing noise into the
year-over-year comparison that the whole project rests on.

Run:  python extract_sections.py
"""
from __future__ import annotations

import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FILINGS, INTERIM, MANIFEST
from sec_client import read_gz
from sections import extract_all

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract")

OUT = INTERIM / "sections.parquet"

USABLE = ("ok", "ok_loose")   # ok_loose = found via the fallback anchor tier



def process_one(row: dict) -> dict | None:
    """Worker. Must be module-level and picklable for Windows spawn."""
    path = FILINGS / row["local"]
    if not path.exists():
        return None
    try:
        text = read_gz(path)
    except Exception:
        return None

    secs = extract_all(text)
    a, b = secs["item1a"], secs["item7"]
    return {
        "cik": row["cik"],
        "ticker": row.get("ticker"),
        "company": row.get("company"),
        "filing_year": row["filing_year"],
        "filed": row["filed"],
        "doc_chars": len(text),
        "item1a_status": a.status,
        "item1a_chars": a.n_chars,
        "item1a_anchors": a.n_candidates,
        "item1a": a.text if a.ok else "",
        "item7_status": b.status,
        "item7_chars": b.n_chars,
        "item7": b.text if b.ok else "",
    }


def main(workers: int = 4) -> None:
    man = pd.read_parquet(MANIFEST)
    rows = man.to_dict("records")
    log.info("extracting sections from %s filings", f"{len(rows):,}")

    out: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_one, r) for r in rows]
        for f in tqdm(as_completed(futs), total=len(futs), desc="filings"):
            r = f.result()
            if r is not None:
                out.append(r)

    df = pd.DataFrame(out).sort_values(["cik", "filing_year"]).reset_index(drop=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    # ---- the report that decides whether this stage worked ---------------
    n = len(df)
    print(f"\n{'='*62}\nEXTRACTION REPORT   ({n:,} filings)\n{'='*62}")

    for item in ("item1a", "item7"):
        vc = df[f"{item}_status"].value_counts()
        ok = int(sum(vc.get(k, 0) for k in USABLE))
        print(f"\n{item}:")
        for status, cnt in vc.items():
            print(f"    {status:<10} {cnt:>6,}  ({cnt/n:>5.1%})")
        if ok:
            s = df.loc[df[f"{item}_status"].isin(USABLE), f"{item}_chars"]
            print(f"    length  p10={s.quantile(.10):>8,.0f}  "
                  f"median={s.median():>8,.0f}  p90={s.quantile(.90):>9,.0f}")

    # The bug this guards against: a constant masquerading as a
    # measurement. If many filings share one exact length, that length is
    # coming from the code, not from the companies.
    ok_len = df.loc[df["item1a_status"].isin(USABLE), "item1a_chars"]
    if len(ok_len):
        top = ok_len.value_counts().head(1)
        val, cnt = int(top.index[0]), int(top.iloc[0])
        flag = "  <-- INVESTIGATE" if cnt > len(ok_len) * 0.02 else ""
        print(f"\n  most common exact length: {val:,} chars, "
              f"{cnt:,} filings ({cnt/len(ok_len):.1%}){flag}")

    frac = df["item1a_chars"] / df["doc_chars"].clip(lower=1)
    runaway = int((df["item1a_status"].isin(USABLE) & (frac > 0.60)).sum())
    print(f"  Item 1A > 60% of filing: {runaway:,}")

    firms_ok = df.loc[df["item1a_status"].isin(USABLE)].groupby("cik").size()
    print(f"  firms with >=2 usable Item 1A years: "
          f"{int((firms_ok >= 2).sum()):,}  <- the year-over-year panel")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
