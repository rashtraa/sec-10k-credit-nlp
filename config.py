"""
Central configuration. Set SEC_USER_AGENT in your environment before
running anything (see "SEC access" below).

DISK LOCATION
-------------
Everything this project writes goes under DATA, which is derived in this
order:

  1. the EDGAR_DATA_ROOT environment variable, if set
  2. otherwise, a `data/` folder next to this repo

So if the repo lives on X:, the data lives on X:. Nothing is ever written to
your home directory by this file.

Package and model caches are a different matter — pip, HuggingFace and
PyTorch all default to C:\\Users\\<you>\\... regardless of where the code
sits, and the HuggingFace cache alone reaches ~1 GB once stage 3 downloads
an embedding model. `activate.ps1` redirects those. Run `python env_check.py`
to confirm nothing is pointing at C: before you start a long job.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- SEC access
# REQUIRED by the SEC: every request must carry a contact User-Agent.
# Set the SEC_USER_AGENT environment variable (or put it in .env, which is
# gitignored). Format: "Your Name your.email@domain.com"
# The placeholder default below will be rejected by the SEC — set it first.
SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "Your Name your.email@domain.com"
)

SEC_RATE_LIMIT = 8.0        # requests/sec. SEC max is 10; 8 leaves headroom.
SEC_MAX_RETRIES = 4
SEC_TIMEOUT = 30

# ---------------------------------------------------------------- universe
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

# A firm must have filed a 10-K in EVERY year above to enter the panel.
# The headline text feature compares a firm's risk factors to its own
# previous year, so a firm with a filing gap yields a discontinuity rather
# than an observation.
REQUIRE_BALANCED_PANEL = True

TARGET_FIRMS = 1500         # cap after the balanced-panel filter

# ---------------------------------------------------------------- paths
# Works whether this file sits at the project root (flat layout) or inside a
# src/ folder. Unzipping is easy to get subtly wrong, and a project that
# writes its data one directory above where you think it does is a bad way
# to find that out.
_here = Path(__file__).resolve().parent
ROOT = _here.parent if _here.name == "src" else _here

# Override with:  setx EDGAR_DATA_ROOT "X:\Projects\edgar-credit-signal\data"
_env_root = os.environ.get("EDGAR_DATA_ROOT", "").strip()
DATA = Path(_env_root).resolve() if _env_root else (ROOT / "data")

RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
FILINGS = RAW / "filings"          # one .txt.gz per filing
MANIFEST = RAW / "manifest.parquet"
INDEX_CACHE = RAW / "form_idx"

for _p in (RAW, INTERIM, PROCESSED, FILINGS, INDEX_CACHE):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- modelling
RANDOM_SEED = 20260821


def describe() -> str:
    return (f"repo   : {ROOT}\n"
            f"data   : {DATA}\n"
            f"filings: {FILINGS}")


if __name__ == "__main__":
    print(describe())
