"""
Stage 5a — split risk-factor sections into retrievable passages.

WHY CHUNK AT ALL
----------------
A risk-factor section runs 25,000-150,000 characters. Two things break if you
treat it as one unit. Embedding models have a hard input limit — a few hundred
tokens — so anything longer is silently truncated and most of the document is
never encoded. And retrieval at document level is useless in practice: "which
of these 7,000 filings mentions supply-chain risk?" is a far less useful
question than "show me the paragraph that does".

WHY NOT FIXED-SIZE WINDOWS
--------------------------
The usual approach is to cut every N characters. It is simple and it splits
sentences in half, orphans the topic sentence from its explanation, and
produces passages that begin mid-clause.

Risk-factor sections have real structure worth using. Each individual risk is
written as a SHORT heading followed by a LONGER explanation:

    We depend on a limited number of suppliers.
    Our manufacturing operations rely on a small number of vendors for
    critical components. If any of these vendors were to fail...

That heading/body pattern is the natural chunk boundary, and it also gives us
a free evaluation set (see `heading_queries` below).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Bumped whenever chunking changes in a way that invalidates existing
# embeddings. Printed by build_index.py so a stale copy is obvious.
STAGE_VERSION = "5.7"

# A heading is short; a body is long. These bounds are deliberately generous —
# the cost of misclassifying is a slightly worse chunk, not a wrong answer.
HEADING_MIN, HEADING_MAX = 20, 260
BODY_MIN = 150           # an explanation shorter than this is not a body
BODY_RATIO = 1.5         # ...and it must be half again longer than its heading

# Chunk size is set by the EMBEDDING MODEL'S INPUT WINDOW, not by taste.
#
# all-MiniLM-L6-v2 accepts 256 word-piece tokens and silently truncates
# anything longer. At roughly 4 characters per token on filing prose that is
# about 1,000 characters — but token density varies with numbers, tables and
# unusual vocabulary, and at a 1,000-character ceiling 2.1% of real passages
# still overflowed. 850 measured clean. An earlier version targeted 1,200 with a 2,000
# ceiling, which meant the top decile of passages had half their text thrown
# away before encoding — while BM25 indexed every word of them. The dense
# side would have been losing a comparison it was never shown the text for.
#
# So the ceiling sits inside the window and both systems index identical text.
MODEL_TOKEN_LIMIT = 256
CHARS_PER_TOKEN = 4.0    # measured rule of thumb for English filing prose

TARGET_CHARS = 700       # chunk size we aim for
MAX_CHARS = 850          # hard ceiling, never exceeded
MIN_CHARS = 200          # shorter than this is merged forward

_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_ENDS_SENT = re.compile(r"[.!?:;][\"')\]”’]?$")

# A blank-line block longer than this is treated as a run-on and re-split on
# single newlines. See `_paragraphs`.
RUNON_CHARS = 2 * TARGET_CHARS


@dataclass
class Chunk:
    cik: int
    ticker: str
    filing_year: int
    chunk_id: str          # "{cik}_{year}_{index}"
    heading: str           # "" when the chunk had no detectable heading
    text: str              # body only — heading NOT included, see note below
    n_chars: int


# ── page furniture ────────────────────────────────────────────────────────
#
# 10-K text carries running headers, footers and page numbers that the
# HTML-to-text converter faithfully preserves as their own short lines. They
# then sit between real paragraphs, where they do two kinds of damage.
#
# They pollute the BODIES — "12 Table of Contents" welded into the middle of a
# risk explanation.
#
# Worse, they pollute the QUERIES. A short line followed by a longer one is
# exactly the heading pattern, so "12 Table of Contents" was being detected as
# a risk heading and turned into an evaluation query. Retrieving the passage
# that follows a page break is a meaningless task, and a benchmark part-built
# from such queries measures nothing. Caught on the real corpus: the leak
# repair's first two examples were both `NN table of contents`.
#
# Dropped at the paragraph level, before anything else sees them.
_FURNITURE = re.compile(r"""^\W*(
      \d{1,4}
    | (page\s*)?\d{0,4}\s*table\s+of\s+contents\s*\d{0,4}
    | part\s+[ivxlc]+\s*[-—:.]?\s*
    | item\s+\d{1,2}[a-z]?\s*[-—:.]?\s*(risk\s+factors)?\s*
    | risk\s+factors
    | unresolved\s+staff\s+comments
    | (annual\s+report|form)\s+10-?k.{0,40}
    | .{0,40}\bform\s+10-?k\b.{0,20}
    | \d{1,4}\s*\|\s*.{0,60}
)\W*$""", re.I | re.X)

# A group label, not a risk: "Risks Related to Our Business". Distinguished
# from a real heading that happens to start the same way by whether it is a
# SENTENCE — "Risk relating to foreign currency could affect our results."
# ends with a full stop; a table-of-contents label does not.
_DIVIDER = re.compile(
    r"^(general\s+|other\s+|additional\s+|certain\s+)?risks?\s+"
    r"(relat(ed|ing)\s+to|associated\s+with|concerning|regarding)\b", re.I)

# "Table of Contents Index to Financial Statements" — page furniture with
# extra words after it, so the anchored furniture pattern missed it.
_TOC_ANYWHERE = re.compile(r"table\s+of\s+contents", re.I)

# A heading is a CLAUSE: it has something doing something. A divider is a noun
# phrase — "Strategic and Business Development Risks". Without a POS tagger,
# the test is whether a predicate word appears at all. A heading that ends in a
# full stop is a sentence by construction and skips the check.
_PREDICATE = re.compile(r"""\b(
      may|might|could|would|will|shall|can|must
    | is|are|was|were|be|been|being|has|have|had|do|does|did
    | face|faces|depend|depends|rely|relies|fail|fails|lack|lacks
    | affect|affects|harm|harms|cause|causes|result|results|expose|exposes
    | incur|incurs|lose|loses|operate|operates|maintain|maintains
    | require|requires|increase|increases|reduce|reduces|experience|experiences
    | subject|dependent|unable|adversely
)\b""", re.I | re.X)


def _is_furniture(p: str) -> bool:
    return bool(_FURNITURE.match(p)) or len(p) < 3


def _too_few_words(p: str) -> bool:
    """
    A heading is a sentence about a risk. Four content words is a low bar that
    still rejects fragments like "Risks", "General" or "Other Matters".
    """
    return len(re.findall(r"[A-Za-z]{2,}", p)) < 4


def is_usable_query(p: str) -> bool:
    """
    Is this heading a retrieval task a person would actually pose?

    Two kinds of text pass the short-line-followed-by-a-longer-one test and
    are not queries. Both were found by printing ten random queries from the
    real corpus — a control that should have existed from the start.

    SECTION DIVIDERS — "Risks Related to Our Organization and Structure:".
    That is a table-of-contents entry for a group of risks, not a risk. There
    is no single passage that answers it, so whatever a system returns is
    scored arbitrarily.

    BULLET FRAGMENTS — "•the price, availability and consumer demand for
    alternative fuels;". A clause lifted out of a list. It has no independent
    meaning, and the passage it belongs to is the list it came from.

    Trailing punctuation is the giveaway for both: a heading is a sentence, so
    it ends with a full stop or nothing at all. A colon introduces a list; a
    semicolon or comma means the thought continues elsewhere.
    """
    # Strip a leading enumerator first: "(ii) Risks Related to Our
    # Indebtedness" is the same divider as "Risks Related to Our
    # Indebtedness", but the bracket hid it from every pattern below. Found
    # in the ten-query sample, which is the third distinct bug that sample
    # has caught.
    p = re.sub(r"^\s*[\(\[]?\s*([ivxlc]{1,4}|[a-z]|\d{1,2})\s*[\)\].:]\s*", "",
               p.strip(), flags=re.I)

    p = p.strip()
    if not p or _is_furniture(p) or _too_few_words(p):
        return False
    if p[0] in "•·◦‣▪-–—*":
        return False

    # Starts mid-sentence: "of this Report. Additionally, there may be other
    # factors..." is the tail of a paragraph the chunker cut, not a heading.
    # A heading starts a thought, so it starts with a capital.
    if p[0].islower():
        return False

    if _TOC_ANYWHERE.search(p):
        return False
    if _DIVIDER.match(p) and not p.endswith("."):
        return False
    # Noun-phrase divider with no verb in it at all.
    if not p.endswith(".") and not _PREDICATE.search(p):
        return False
    return p[-1] not in ":;,"


def _starts_new(line: str) -> bool:
    c = line[:1]
    return bool(c) and (c.isupper() or c.isdigit() or c in "\"'“‘(•-—")


def _relines(block: str) -> list[str]:
    """
    Re-split a run-on block on single newlines.

    WHY THIS IS NEEDED
    ------------------
    Paragraphs are normally separated by a blank line. But the HTML-to-text
    converter emits ONE newline per block element, so whether a filing comes
    out with blank lines at all depends on how its HTML was authored — and
    plenty of filers wrap every paragraph in a `<div>` with no spacing. Those
    filings arrive as a single 60,000-character "paragraph".

    That is not a cosmetic problem. With no paragraph boundaries there are no
    headings to detect, so the section gets cut into fixed-size pieces with
    headings buried inside them — and since the evaluation queries ARE heading
    text, a buried heading is a passage answerable by literal string match.

    THE RE-JOIN RULE
    ----------------
    Single newlines are ambiguous: some are real paragraph breaks, some are
    just where the source HTML happened to wrap. A break is real when the next
    line clearly STARTS something — a capital, a digit, a quote, a bullet —
    and the previous line either finished a sentence or was short enough to be
    a heading. Everything else is a wrapped continuation and gets joined back.
    """
    parts: list[str] = []
    for raw in block.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if parts and not (_starts_new(line) and
                          (_ENDS_SENT.search(parts[-1]) or len(parts[-1]) <= HEADING_MAX)):
            parts[-1] = f"{parts[-1]} {line}"
        else:
            parts.append(line)
    return parts


def _paragraphs(text: str) -> list[str]:
    """
    Let each filing's own convention decide.

    Re-lining is only applied where blank lines have clearly failed: a section
    with almost no blank-line blocks at all, or an individual block so long it
    cannot be one paragraph. A filing that already separates its paragraphs
    properly is left alone — inside such a paragraph, a line that happens to
    wrap right after a full stop would otherwise be split off as a spurious
    paragraph, and could then be mistaken for a heading and held out of the
    index entirely.
    """
    blocks = [p.strip() for p in _PARA.split(text) if p.strip()]
    runon = len(blocks) < 3

    out: list[str] = []
    for b in blocks:
        if "\n" in b and (runon or len(b) > RUNON_CHARS):
            out.extend(_relines(b))
        else:
            out.append(b.replace("\n", " "))
    return [p for p in out if not _is_furniture(p)]


def _looks_like_heading(p: str, nxt: str | None) -> bool:
    """
    A short paragraph immediately followed by a substantially longer one.

    Not "ends without a full stop" — most risk headings are complete sentences
    ("We depend on a limited number of suppliers."). Length contrast is the
    reliable signal, not punctuation.

    WHY THE THRESHOLDS ARE LOOSE, AND WHY THAT MATTERS
    --------------------------------------------------
    An earlier version demanded a body of 300+ characters and twice the
    heading's length. Anything that missed those bounds was not recognised as
    a heading, so the heading paragraph was treated as body text and the
    explanation after it was then merged in behind — putting heading text
    INSIDE an indexed passage.

    That is fatal rather than untidy. The evaluation queries ARE heading text,
    and the control the whole benchmark rests on is that headings are never
    indexed. One leak and BM25 wins by matching a literal string, which is
    exactly the artefact the comparison exists to rule out. Measured on a
    synthetic corpus this dragged BM25's Recall@10 to 0.09 — it was ranking
    passages that merely contained the heading TEMPLATE, from unrelated
    topics.

    So the test errs toward calling something a heading. The cost of a false
    positive is one short paragraph held out of the index; the cost of a false
    negative is a meaningless benchmark.
    """
    if not (HEADING_MIN <= len(p) <= HEADING_MAX):
        return False
    if not is_usable_query(p):
        return False
    if nxt is None or len(nxt) < BODY_MIN:
        return False
    return len(nxt) > BODY_RATIO * len(p)


def _split_long(body: str, limit: int = MAX_CHARS) -> list[str]:
    """Split an over-long body on sentence boundaries, never mid-sentence."""
    if len(body) <= limit:
        return [body]
    out, cur = [], ""
    for sent in _SENT.split(body):
        if cur and len(cur) + len(sent) + 1 > limit:
            out.append(cur.strip())
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur.strip():
        out.append(cur.strip())

    # Last resort: a SINGLE sentence longer than the ceiling cannot be split
    # on a sentence boundary, and filing prose does produce 1,200-character
    # sentences. Left alone they overflow the embedding model's window and get
    # silently truncated — measured at 2.1% of passages before this existed —
    # so they are cut on the nearest word boundary instead. An awkward split is
    # better than text the dense side never sees while BM25 indexes all of it.
    final = []
    for piece in out:
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit)
            if cut < limit // 2:
                cut = limit
            final.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            final.append(piece)
    return final


def chunk_section(text: str, cik: int, ticker: str, year: int) -> list[Chunk]:
    """Split one Item 1A section into passages."""
    paras = _paragraphs(text)
    chunks: list[Chunk] = []
    i = 0

    while i < len(paras):
        p = paras[i]
        nxt = paras[i + 1] if i + 1 < len(paras) else None

        if _looks_like_heading(p, nxt):
            heading, body, i = p, nxt, i + 2
        else:
            heading, body, i = "", p, i + 1

        # Absorb following short paragraphs that clearly continue the body,
        # stopping at anything that looks like the next heading.
        while (i < len(paras) and len(body) < TARGET_CHARS
               and not _looks_like_heading(paras[i],
                                           paras[i + 1] if i + 1 < len(paras) else None)):
            body = f"{body}\n\n{paras[i]}"
            i += 1

        for piece in _split_long(body):
            if len(piece) >= MIN_CHARS:
                chunks.append(Chunk(
                    cik=cik, ticker=ticker, filing_year=year,
                    chunk_id=f"{cik}_{year}_{len(chunks)}",
                    heading=heading, text=piece, n_chars=len(piece)))

    return chunks


def heading_queries(chunks: list[Chunk]) -> list[dict]:
    """
    Build an evaluation set for free, from the documents themselves.

    A risk heading IS a query. "We depend on a limited number of suppliers"
    is exactly what an analyst would type, and we know which passage it
    belongs to — so we have a query and its correct answer with no manual
    labelling and no LLM.

    THE BIAS THIS CREATES, AND THE FIX
    ----------------------------------
    The heading's words also appear in the body it introduces. Left alone,
    that hands the contest to keyword matching: BM25 wins by finding the
    literal heading text inside the passage, and the benchmark measures
    string overlap rather than retrieval quality.

    So chunks store the BODY ONLY — the heading is held out and never
    indexed. A system must match the query to the *explanation* on meaning,
    not to a repeated string. That is the whole point of the comparison, and
    without this control the result would be meaningless.
    """
    return [{"query": c.heading, "relevant_chunk_id": c.chunk_id,
             "cik": c.cik, "filing_year": c.filing_year}
            for c in chunks if c.heading]
