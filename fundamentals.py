"""
Stage 4a — financial statement data for the panel, from SEC XBRL.

WHY THE FRAMES API RATHER THAN companyfacts
-------------------------------------------
The obvious route is data.sec.gov/api/xbrl/companyfacts/CIK##########.json,
one request per firm. That is 1,500 requests returning 1-10 MB each — call
it 5 GB of JSON to extract about a dozen numbers per firm-year.

The frames API inverts the query: it returns ONE concept for EVERY filer in
one period, in a single response.

    /api/xbrl/frames/us-gaap/Assets/USD/CY2023Q4I.json

So the whole panel costs (concepts x years) requests — about 60 — instead of
1,500, and a few MB instead of gigabytes.

A caveat that must be stated rather than buried: frames assigns each filing
to a CALENDAR period. A firm with a June fiscal year-end lands in the
calendar frame its period best matches, so "CY2023" is not identical to
"fiscal 2023" for off-calendar filers. For a leverage ratio compared
within-firm across years this is immaterial, and the alternative — parsing
each firm's own fiscal calendar out of companyfacts — costs the 5 GB above.
The choice is deliberate and is recorded in the limitations.

Run:  python fundamentals.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import INTERIM, MANIFEST, RAW, YEARS
from sec_client import get

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fundamentals")

FRAMES_CACHE = RAW / "frames"
FRAMES_CACHE.mkdir(parents=True, exist_ok=True)
OUT = INTERIM / "fundamentals.parquet"

# (concept, is_instantaneous). Balance-sheet items are point-in-time and use
# the "I" suffix; income and cash-flow items span the year and do not.
#
# Several concepts have more than one commonly used tag. Filers are not
# consistent, so alternatives are tried in order and the first populated one
# wins — recorded per firm-year so the fallback is auditable.
CONCEPTS: dict[str, tuple[list[str], bool]] = {
    "assets":        (["Assets"], True),
    # Many filers never tag `Liabilities` directly — they tag
    # LiabilitiesAndStockholdersEquity (which equals Assets) and
    # StockholdersEquity, leaving liabilities to be derived. Requiring the
    # direct tag cost 2,117 firm-years (32% of what reached that step), so
    # the total is collected too and the difference taken below.
    "liabilities":   (["Liabilities"], True),
    "liab_and_eq":   (["LiabilitiesAndStockholdersEquity"], True),
    "equity":        (["StockholdersEquity",
                       "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], True),
    "cash":          (["CashAndCashEquivalentsAtCarryingValue",
                       "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                       "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations",
                       "Cash"], True),
    "ebit":          (["OperatingIncomeLoss"], False),
    "interest":      (["InterestExpense",
                       "InterestExpenseDebt",
                       "InterestIncomeExpenseNet"], False),
    "revenue":       (["Revenues",
                       "RevenueFromContractWithCustomerExcludingAssessedTax",
                       "SalesRevenueNet"], False),
    "cfo":           (["NetCashProvidedByUsedInOperatingActivities"], False),
    "net_income":    (["NetIncomeLoss"], False),
}


def fetch_frame(tag: str, year: int, instant: bool) -> pd.DataFrame | None:
    """One concept, one calendar period, every filer. Cached to disk."""
    period = f"CY{year}Q4I" if instant else f"CY{year}"
    cache = FRAMES_CACHE / f"{tag}_{period}.json"

    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/{period}.json"
        text = get(url, host="data.sec.gov")
        if text is None:
            return None
        cache.write_text(text)
        payload = json.loads(text)

    rows = payload.get("data", [])
    if not rows:
        return None
    df = pd.DataFrame(rows)[["cik", "val"]]
    df["cik"] = df["cik"].astype(int)
    df["year"] = year
    # A filer can appear twice (amended filings). Keep the last.
    return df.drop_duplicates(subset="cik", keep="last")


def main() -> None:
    man = pd.read_parquet(MANIFEST)
    panel = set(man["cik"].unique())
    log.info("collecting fundamentals for %s firms", f"{len(panel):,}")

    frames: dict[str, pd.DataFrame] = {}
    used_tag: dict[str, dict[int, str]] = {}

    total = sum(len(tags) for tags, _ in CONCEPTS.values()) * len(YEARS)
    with tqdm(total=total, desc="frames") as bar:
        for name, (tags, instant) in CONCEPTS.items():
            parts, chosen = [], {}
            for year in YEARS:
                best = None
                for tag in tags:
                    d = fetch_frame(tag, year, instant)
                    bar.update(1)
                    if d is None:
                        continue
                    d = d[d["cik"].isin(panel)]
                    # Prefer the tag that covers more of OUR panel, not the
                    # one that covers more filers generally.
                    if best is None or len(d) > len(best):
                        best, chosen[year] = d, tag
                if best is not None and len(best):
                    parts.append(best)
            if parts:
                frames[name] = (pd.concat(parts)
                                .rename(columns={"val": name})[["cik", "year", name]])
                used_tag[name] = chosen
                log.info("%-12s %6s firm-years", name,
                         f"{len(frames[name]):,}")
            else:
                log.warning("%-12s no data", name)

    # Outer-join so a firm-year missing one concept is not dropped entirely;
    # the ratio built from it becomes NaN and is handled at model time.
    df = None
    for name, part in frames.items():
        df = part if df is None else df.merge(part, on=["cik", "year"], how="outer")

    # Fill liabilities from the accounting identity where the direct tag is
    # absent: Liabilities = LiabilitiesAndStockholdersEquity - Equity.
    if {"liab_and_eq", "equity"} <= set(df.columns):
        derived = df["liab_and_eq"] - df["equity"]
        n_before = df["liabilities"].notna().sum()
        df["liabilities"] = df["liabilities"].fillna(derived)
        n_after = df["liabilities"].notna().sum()
        log.info("liabilities: %s tagged directly, %s recovered from the "
                 "identity", f"{n_before:,}", f"{n_after - n_before:,}")
        df = df.drop(columns=["liab_and_eq"])

    df = df.sort_values(["cik", "year"]).reset_index(drop=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"\n{'='*62}\nFUNDAMENTALS   ({len(df):,} firm-years, "
          f"{df['cik'].nunique():,} firms)\n{'='*62}\n")
    cov = (df.notna().mean().drop(["cik", "year"]).sort_values(ascending=False)
           .rename("coverage").to_frame())
    cov["coverage"] = cov["coverage"].map("{:.1%}".format)
    print(cov.to_string())
    print(f"\ntags actually used:")
    for name, per_year in used_tag.items():
        tags = sorted(set(per_year.values()))
        print(f"    {name:<12} {', '.join(tags)}")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
