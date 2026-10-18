"""
Where am I? Run this any time you come back to the project.

Reports what each stage has produced on disk and prints the exact next
command. Nothing here modifies anything.

    python status.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FILINGS, INTERIM, MANIFEST, PROCESSED, DATA


def mb(paths) -> float:
    return sum(p.stat().st_size for p in paths) / 1e6


def main() -> None:
    print(f"\nproject data: {DATA}\n")
    print(f"{'stage':<34}{'state':<34}")
    print("-" * 68)

    next_cmd = None

    # ---- stage 1a: manifest ---------------------------------------------
    if MANIFEST.exists():
        import pandas as pd
        man = pd.read_parquet(MANIFEST)
        print(f"{'1a build_universe':<34}"
              f"{len(man):,} filings, {man['cik'].nunique():,} firms")
    else:
        print(f"{'1a build_universe':<34}not run")
        next_cmd = next_cmd or "python build_universe.py"
        man = None

    # ---- stage 1b: downloads --------------------------------------------
    files = list(FILINGS.glob("*.txt.gz"))
    if files:
        line = f"{len(files):,} files, {mb(files):,.0f} MB"
        if man is not None:
            missing = len(man) - len(files)
            if missing > 0:
                line += f"  ({missing:,} still missing)"
                next_cmd = next_cmd or "python download_filings.py"
        print(f"{'1b download_filings':<34}{line}")
    else:
        print(f"{'1b download_filings':<34}not run")
        next_cmd = next_cmd or "python download_filings.py"

    # ---- stage 2a: sections ---------------------------------------------
    sec = INTERIM / "sections.parquet"
    if sec.exists():
        import pandas as pd
        d = pd.read_parquet(sec, columns=["item1a_status"])
        ok = int((d["item1a_status"] == "ok").sum())
        print(f"{'2a extract_sections':<34}"
              f"{len(d):,} rows, {ok:,} usable Item 1A ({ok/len(d):.0%})")
    else:
        print(f"{'2a extract_sections':<34}not run")
        if files:
            next_cmd = next_cmd or "python extract_sections.py"

    # ---- stage 2b: features ---------------------------------------------
    feat = PROCESSED / "text_features.parquet"
    if feat.exists():
        import pandas as pd
        d = pd.read_parquet(feat, columns=["risk_change"])
        n = int(d["risk_change"].notna().sum())
        print(f"{'2b text_features':<34}{len(d):,} rows, "
              f"{n:,} year-over-year observations")
    else:
        print(f"{'2b text_features':<34}not run")
        if sec.exists():
            next_cmd = next_cmd or "python text_features.py"

    print()
    if next_cmd:
        print(f"next:  {next_cmd}\n")
    else:
        print("all stages built so far are complete.\n")


if __name__ == "__main__":
    main()
