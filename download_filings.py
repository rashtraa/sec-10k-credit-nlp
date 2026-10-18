"""
Step 2 — download every filing in the manifest, strip it to text, gzip it.

Two decisions worth stating:

  Text, not HTML, is what lands on disk. A 10-K primary document is 2-20 MB
  of inline-XBRL-laden HTML; 9,000 of them is 50-150 GB. The same filings as
  whitespace-normalised text, gzipped, are under a gigabyte. Nothing
  downstream reads the markup, so keeping it would cost two orders of
  magnitude of disk for no analytical gain.

  The run is resumable. Anything already on disk is skipped, so an
  interrupted download resumes where it stopped rather than restarting. At
  8 req/s a full pass is ~20 minutes of request time; a failed run that had
  to restart from zero would be the single most annoying way to lose an
  evening.
"""
from __future__ import annotations

import logging
from html import unescape
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FILINGS, MANIFEST
from sec_client import get, write_gz

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download")

MAX_BYTES = 60 * 1024 * 1024      # skip pathological filings

_WS = re.compile(r"[ \t\r\f\v ]+")
_NL = re.compile(r"\n{3,}")


def html_to_text(html: str) -> str:
    """
    Flatten filing HTML to text while preserving block boundaries.

    Block boundaries matter more than they look. Item headings ("Item 1A.
    Risk Factors") are found by regex downstream, and if a table cell runs
    into the heading on the same line the anchor is missed. Emitting a
    newline per block element keeps headings on lines of their own.
    """
    try:
        import lxml.html
        from lxml.html.clean import Cleaner
    except ImportError:  # pragma: no cover
        raise SystemExit("pip install lxml lxml_html_clean")

    # Inline XBRL hides a large machine-readable payload inside <ix:header>
    # and display:none spans. Left in, it swamps the prose.
    cleaner = Cleaner(scripts=True, javascript=True, comments=True,
                      style=True, inline_style=True, links=False,
                      meta=True, page_structure=False, embedded=True,
                      frames=True, forms=False, annoying_tags=False,
                      remove_tags=None, kill_tags=["ix:header", "script",
                                                   "style", "noscript"])
    try:
        doc = lxml.html.fromstring(html)
        # Hidden elements must go BEFORE the cleaner runs: inline_style=True
        # strips the style attribute, which destroys the evidence that the
        # element was hidden. Filings use display:none for XBRL scaffolding
        # and for prior-period boilerplate that is not part of this year's
        # disclosure — counting it would pollute the year-over-year
        # comparison that the headline text feature is built on.
        for el in doc.xpath('//*[@style]'):
            style = (el.get("style") or "").lower().replace(" ", "")
            if "display:none" in style or "visibility:hidden" in style:
                el.drop_tree()
        doc = cleaner.clean_html(doc)
    except Exception:
        # Malformed markup: fall back to a blunt tag strip rather than
        # dropping the filing entirely.
        txt = re.sub(r"<[^>]+>", " ", html)
        return _NL.sub("\n\n", _WS.sub(" ", txt)).strip()

    # Every block element, table cells included, ends with a newline.
    #
    # An earlier version gave <td> a space instead, reasoning that an item
    # heading is often split across two cells ("Item 1A." | "Risk Factors")
    # and a newline between them would break the match. That reasoning was
    # wrong twice over. The section regex separates the number from the
    # title with `\s*`, which spans newlines, so the split heading matches
    # either way. And filings routinely put headings in layout tables where
    # the heading sits in the SECOND cell of a row — with a space separator
    # that heading lands mid-line, invisible to an anchor that requires a
    # line start. On the real panel that cost 2,571 filings (29.6%),
    # reported as `noanchor`.
    for el in doc.iter():
        if el.tag in ("p", "div", "br", "tr", "td", "th", "li", "table",
                      "h1", "h2", "h3", "h4", "h5", "h6"):
            el.tail = "\n" + (el.tail or "")

    text = doc.text_content()
    # Filings double-escape: `&amp;#x2022;` survives lxml's first decode as the
    # literal seven characters `&#x2022;`, so bullets and non-breaking spaces
    # end up in the text as tokens. Found by reading query/passage pairs by
    # hand — no control was looking for it. One more unescape clears it.
    text = unescape(text)
    text = text.replace("\xa0", " ")
    text = _WS.sub(" ", text)
    text = "\n".join(ln.strip() for ln in text.split("\n"))
    return _NL.sub("\n\n", text).strip()


def fetch_one(row: dict) -> tuple[str, str]:
    """Returns (status, local_filename)."""
    dest = FILINGS / row["local"]
    if dest.exists() and dest.stat().st_size > 1024:
        return "skip", row["local"]

    raw = get(row["url"], binary=True)
    if raw is None:
        return "missing", row["local"]
    if len(raw) > MAX_BYTES:
        return "toobig", row["local"]

    try:
        html = raw.decode("utf-8", errors="replace")
        text = html_to_text(html)
    except Exception as e:
        log.warning("parse failure %s: %s", row["local"], e)
        return "parsefail", row["local"]

    if len(text) < 5000:
        # A real 10-K is never this short. Usually a wrapper page.
        return "tooshort", row["local"]

    write_gz(dest, text)
    return "ok", row["local"]


def main(threads: int = 8) -> None:
    if not MANIFEST.exists():
        raise SystemExit("run build_universe.py first")
    man = pd.read_parquet(MANIFEST)
    rows = man.to_dict("records")
    log.info("%s filings in manifest", f"{len(rows):,}")

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(fetch_one, r) for r in rows]
        for f in tqdm(as_completed(futs), total=len(futs), desc="filings"):
            status, _ = f.result()
            counts[status] = counts.get(status, 0) + 1

    log.info("done: %s", counts)
    on_disk = list(FILINGS.glob("*.txt.gz"))
    mb = sum(p.stat().st_size for p in on_disk) / 1e6
    log.info("%s files on disk, %.0f MB compressed", f"{len(on_disk):,}", mb)


if __name__ == "__main__":
    main()
