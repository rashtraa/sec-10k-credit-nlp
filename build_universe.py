"""
Step 1 — decide which firms are in the panel, and locate every 10-K they filed.

Strategy, and why it is shaped this way:

  a) EDGAR publishes a quarterly `form.idx` listing every filing made that
     quarter. Twenty-four requests therefore identify every 10-K filer in
     2019-2024, instead of 10,000 per-company lookups.

  b) We keep only firms with a ticker in company_tickers.json. Not snobbery:
     the label is a Merton distance-to-default, which needs an equity price.
     A firm with no traded equity cannot be labelled, so it cannot enter.

  c) We require a 10-K in every year (balanced panel). The headline text
     feature compares a firm's risk factors to its own previous year, so a
     firm with a filing gap yields a discontinuity rather than an observation.

  d) Only then do we hit data.sec.gov/submissions once per surviving firm to
     recover the primary document filename for each filing. Doing this after
     the filters, rather than before, saves ~8,500 requests.

Output: data/raw/manifest.parquet — one row per filing to download.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (INDEX_CACHE, MANIFEST, RAW, REQUIRE_BALANCED_PANEL,
                    TARGET_FIRMS, YEARS)
from sec_client import get

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("universe")

# form.idx is fixed-width-ish but ragged. Anchoring on the three fields with
# unambiguous shapes (CIK digits, ISO date, edgar/data path) parses it more
# reliably than column offsets, which have drifted between years.
IDX_LINE = re.compile(
    r"^(?P<form>\S[^ ]*(?: [^ ]+)*?)\s{2,}"
    r"(?P<name>.+?)\s{2,}"
    r"(?P<cik>\d{1,10})\s+"
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<path>edgar/data/\S+)\s*$"
)

# 10-K405 / 10-KSB are legacy variants; 10-K/A is an amendment. We take the
# original annual report only — an amendment restates part of a filing and
# would double-count the firm-year.
WANTED_FORMS = {"10-K"}


def fetch_form_index(year: int, qtr: int) -> list[dict]:
    """Download (and cache) one quarter's form index; return 10-K rows."""
    cache = INDEX_CACHE / f"form_{year}_QTR{qtr}.idx"
    if cache.exists():
        text = cache.read_text(encoding="utf-8", errors="replace")
    else:
        url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/form.idx"
        text = get(url)
        if text is None:
            log.warning("no index for %d QTR%d", year, qtr)
            return []
        cache.write_text(text, encoding="utf-8")

    rows = []
    for line in text.splitlines():
        if not line.startswith("10-K"):
            continue                      # form.idx is sorted by form type
        m = IDX_LINE.match(line)
        if not m:
            continue
        if m.group("form").strip() not in WANTED_FORMS:
            continue
        rows.append({
            "cik": int(m.group("cik")),
            "company": m.group("name").strip(),
            "filed": m.group("date"),
            "idx_path": m.group("path"),
            "filing_year": year,
        })
    return rows


def load_tickers() -> pd.DataFrame:
    """CIK -> ticker map. Needed because the label requires an equity price."""
    cache = RAW / "company_tickers.json"
    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        text = get("https://www.sec.gov/files/company_tickers.json")
        if text is None:
            raise RuntimeError("could not fetch company_tickers.json")
        cache.write_text(text)
        payload = json.loads(text)

    recs = payload.values() if isinstance(payload, dict) else payload
    df = pd.DataFrame(recs).rename(columns={"cik_str": "cik", "title": "company"})
    df["cik"] = df["cik"].astype(int)
    # A CIK can map to several tickers (share classes). Keep the first,
    # which in this file is the primary listing.
    return df.drop_duplicates(subset="cik", keep="first")[["cik", "ticker"]]


def fetch_primary_docs(cik: int) -> list[dict]:
    """
    Recover accession numbers and primary document filenames for one firm.

    The submissions JSON holds the most recent ~1000 filings inline. For a
    six-year window of annual reports that is always sufficient; older
    filings would live in the paginated files under `filings.files`.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    text = get(url, host="data.sec.gov")
    if text is None:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []

    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    out = []
    for i, form in enumerate(forms):
        if form not in WANTED_FORMS:
            continue
        filing_date = recent["filingDate"][i]
        year = int(filing_date[:4])
        if year not in YEARS:
            continue
        accession = recent["accessionNumber"][i]
        primary = recent.get("primaryDocument", [None] * len(forms))[i]
        if not primary:
            continue
        nodash = accession.replace("-", "")
        out.append({
            "cik": cik,
            "accession": accession,
            "filed": filing_date,
            "filing_year": year,
            "report_date": recent.get("reportDate", [""] * len(forms))[i],
            "primary_doc": primary,
            "url": (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                    f"{nodash}/{primary}"),
        })
    return out


def main(threads: int = 8) -> None:
    # ---- a) every 10-K filed in the window -------------------------------
    log.info("fetching %d quarterly form indexes", len(YEARS) * 4)
    idx_rows = []
    for year in YEARS:
        for qtr in (1, 2, 3, 4):
            idx_rows.extend(fetch_form_index(year, qtr))
    idx = pd.DataFrame(idx_rows)
    log.info("found %s 10-K filings by %s distinct CIKs",
             f"{len(idx):,}", f"{idx['cik'].nunique():,}")

    # ---- b) must have a traded ticker ------------------------------------
    tickers = load_tickers()
    idx = idx.merge(tickers, on="cik", how="inner")
    log.info("after ticker filter: %s filings, %s firms",
             f"{len(idx):,}", f"{idx['cik'].nunique():,}")

    # ---- c) balanced panel ------------------------------------------------
    if REQUIRE_BALANCED_PANEL:
        years_per_firm = idx.groupby("cik")["filing_year"].nunique()
        complete = years_per_firm[years_per_firm >= len(YEARS)].index
        idx = idx[idx["cik"].isin(complete)]
        log.info("after balanced-panel filter: %s firms",
                 f"{idx['cik'].nunique():,}")

    # Rank firms by total filing footprint as a crude size proxy, then cap.
    # (Real market cap arrives later with the price data; this is only to
    # make the cap deterministic rather than arbitrary.)
    keep = (idx.groupby("cik").size().sort_values(ascending=False)
            .head(TARGET_FIRMS).index)
    idx = idx[idx["cik"].isin(keep)]
    firms = sorted(idx["cik"].unique())
    log.info("universe fixed at %s firms", f"{len(firms):,}")

    # ---- d) primary document lookup --------------------------------------
    log.info("resolving primary documents (%d requests)", len(firms))
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = {ex.submit(fetch_primary_docs, c): c for c in firms}
        for f in tqdm(as_completed(futs), total=len(futs), desc="submissions"):
            rows.extend(f.result())

    man = pd.DataFrame(rows)
    if man.empty:
        raise RuntimeError("no filings resolved — check SEC_USER_AGENT")

    man = man.merge(tickers, on="cik", how="left")
    man = man.merge(idx[["cik", "company"]].drop_duplicates("cik"),
                    on="cik", how="left")
    man = man.sort_values(["cik", "filed"]).drop_duplicates(
        subset=["cik", "filing_year"], keep="first").reset_index(drop=True)
    man["local"] = man.apply(
        lambda r: f"{r['cik']}_{r['filing_year']}_{r['accession']}.txt.gz", axis=1)

    man.to_parquet(MANIFEST, index=False)
    log.info("manifest written: %s filings, %s firms -> %s",
             f"{len(man):,}", f"{man['cik'].nunique():,}", MANIFEST)
    print(man.groupby("filing_year").size().to_string())


if __name__ == "__main__":
    main()
