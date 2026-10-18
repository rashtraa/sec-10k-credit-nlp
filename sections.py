"""
Extract Item 1A (Risk Factors) and Item 7 (MD&A) from 10-K text.

This is the highest-risk component in the project, and it is worth being
explicit about why, because "I used a regex" is not a defensible answer to
an interviewer who knows filings.

The naive approach fails on every real 10-K:

  1. THE TABLE OF CONTENTS PROBLEM. Every 10-K contains the string
     "Item 1A. Risk Factors" at least twice — once in the table of contents
     near the front, once at the actual section. A regex that takes the
     first match extracts the table of contents and the two pages after it.
     Silently. The output looks like text, the pipeline runs green, and
     every downstream number is garbage.

  2. THE CROSS-REFERENCE PROBLEM. Filings say things like "see Item 1A.
     Risk Factors" inside other sections, adding more false anchors.

  3. THE INCORPORATION-BY-REFERENCE PROBLEM. Smaller reporting companies
     are permitted to omit Item 1A entirely, and some firms incorporate it
     by reference to another document. The section is legitimately absent,
     and the extractor must return "absent" rather than a stub.

The rule used here: find every candidate anchor for the opening item and
every anchor for the following item, form all valid (start, end) spans, and
keep the LONGEST one. The real section is nearly always an order of
magnitude longer than a contents-line or a cross-reference, so length is a
far more reliable discriminator than position.

Every span is then length-audited. Anything under MIN_SECTION_CHARS is
recorded as a extraction failure rather than passed downstream — a firm-year
with a 400-character "risk factors section" is a parsing bug, and letting it
through would put noise into the year-over-year comparison.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Item headings vary: "Item 1A.", "ITEM 1A -", "Item 1A:", with or without
# the title. Allow the punctuation and casing to float, require the item
# number to be exact.
def _anchor(item: str, title: str | None = None, *,
            line_start: bool = True) -> re.Pattern:
    # The `^` (with MULTILINE) is load-bearing, not tidiness.
    #
    # Without it the anchor matches "Item 1A" ANYWHERE, including inside a
    # sentence: `...see Item 1A, "Risk Factors" in this Annual Report...`
    # appears in the forward-looking-statements disclaimer of a large share
    # of filings. That match lands mid-paragraph and the span it opens is a
    # fragment of the disclaimer, not a section.
    #
    # On the real panel this produced EPR 2020: a 1,083-character "risk
    # factors section" consisting of safe-harbour boilerplate, which then
    # scored the single highest year-over-year dissimilarity in the whole
    # dataset (0.951) purely because it was compared against a genuine
    # section from the prior year.
    #
    # After the HTML-to-text conversion every heading sits at the start of
    # its own line, so requiring that costs nothing and removes the entire
    # class of mid-sentence false anchors.
    lead = r"^[ \t]*" if line_start else r""
    pat = rf"{lead}item\s*{item}\s*[.\-:—–)]?\s*"
    if title:
        pat += rf"(?:{title})?"
    return re.compile(pat, re.IGNORECASE | re.MULTILINE)


# Each section has ONE start anchor and a LIST of acceptable end anchors.
#
# A single end anchor is not enough, and assuming otherwise cost 731 filings
# (8.4% of the panel) on the first real run:
#
#   * Item 1B is frequently omitted outright — it is usually "None." and some
#     filers simply drop the heading.
#   * Filings for fiscal years ending after 15 December 2023 carry a new
#     Item 1C (Cybersecurity), mandated by the SEC's 2023 disclosure rule,
#     which sits between 1B and 2.
#   * Item 2 (Properties) always follows, so it is the backstop.
#
# The extractor takes the FIRST end anchor of ANY listed kind that appears
# after the start, so whichever heading the filer actually used, the section
# is bounded correctly.
def _pair(item, title, ends, line_start):
    return (_anchor(item, title, line_start=line_start),
            [_anchor(i, t, line_start=line_start) for i, t in ends])

_ITEM1A = ("1A", r"risk\s+factors",
           [("1B", r"unresolved\s+staff\s+comments"),
            ("1C", r"cybersecurity"),
            ("2",  r"properties")])
_ITEM7 = ("7", r"management.{0,3}s\s+discussion",
          [("7A", r"quantitative\s+and\s+qualitative"),
           ("8",  r"financial\s+statements\s+and\s+supplementary")])

# Two tiers per section: STRICT anchors must begin a line, LOOSE anchors may
# appear anywhere.
#
# Strict is tried first because a heading on its own line is strong evidence
# and mid-sentence mentions are excluded outright. But strict alone is too
# brittle: filings put headings inside layout tables, and a heading in the
# second cell of a row lands mid-line in the extracted text. Requiring a
# line start cost 2,571 filings (29.6%) on the real panel, reported as
# `noanchor`, and dropped the year-over-year sample from 5,151 to 3,760.
#
# So loose is a fallback, used only when strict yields nothing usable. The
# length floor, the longest-span rule and the run-on guard all still apply
# on the loose pass, which is what keeps a mid-sentence match from producing
# the kind of fragment that caused the EPR 2020 artifact.
ANCHORS = {
    "item1a": _pair(*_ITEM1A, line_start=True),
    "item7":  _pair(*_ITEM7,  line_start=True),
}
ANCHORS_LOOSE = {
    "item1a": _pair(*_ITEM1A, line_start=False),
    "item7":  _pair(*_ITEM7,  line_start=False),
}

# Lower plausibility bound.
#
# Originally 1,000, which was far too generous. On the real panel genuine
# Item 1A sections have a 10th percentile of ~29,000 characters and a median
# of ~75,000. A 1,083-character span cleared the old floor and went on to
# produce the largest year-over-year dissimilarity in the dataset — it was a
# scrap of forward-looking-statements boilerplate, not a section.
#
# 5,000 is still permissive against the observed distribution (well below
# the 10th percentile, so genuinely brief sections from small filers
# survive) while excluding fragments by an order of magnitude.
MIN_SECTION_CHARS = 5_000

# Upper plausibility bound. Risk-factor sections are long but not unbounded;
# the largest genuine ones in this panel run to roughly 150k characters.
# Anything past this is a missed end anchor reading on into the filing.
MAX_SECTION_CHARS = 250_000

# Phrases that mark a section as legitimately omitted rather than missed.
OMITTED = re.compile(
    r"not\s+required\s+for\s+smaller\s+reporting\s+companies"
    r"|smaller\s+reporting\s+compan(?:y|ies).{0,80}not\s+required"
    r"|incorporated\s+herein\s+by\s+reference",
    re.IGNORECASE)


@dataclass
class Section:
    name: str
    text: str
    status: str          # ok | absent | tooshort | noanchor
    n_chars: int
    n_candidates: int    # how many anchors were seen — diagnostic

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "ok_loose")


def _starts_cleanly(text: str, pos: int, window: int = 40) -> bool:
    """
    Does the text at `pos` look like the beginning of a section rather than
    the middle of a sentence?

    Skips leading whitespace, then requires the first meaningful character
    to be a capital letter, an opening quote, or a digit. A comma, a closing
    quote, or a lowercase word means the anchor landed inside a sentence.
    """
    seg = text[pos:pos + window].lstrip()
    if not seg:
        return False
    c = seg[0]
    return c.isupper() or c.isdigit() or c in "\"'“‘("


def extract(text: str, which: str) -> Section:
    """
    Extract one section, preferring line-start anchors.

    Strict first. If that yields nothing usable, retry with anchors that may
    appear anywhere, and mark the result `ok_loose` so the weaker evidence
    stays visible in the status counts rather than being laundered into
    `ok`. Both tiers apply the same length floor, longest-span rule and
    run-on guard.
    """
    strict = _extract_one(text, *ANCHORS[which], which)
    if strict.ok:
        return strict

    loose = _extract_one(text, *ANCHORS_LOOSE[which], which)
    if loose.status == "ok":
        loose.status = "ok_loose"
        return loose

    # Neither tier worked. Report the strict diagnosis — it is the more
    # informative one, since loose failing too means the section really is
    # not locatable rather than merely oddly formatted.
    return strict


def _extract_one(text: str, start_pat, end_pats, which: str) -> Section:
    starts = [m.end() for m in start_pat.finditer(text)]
    # Pool every acceptable end anchor. Which heading a filer used does not
    # matter — only that the section is bounded by one of them.
    ends = sorted(m.start() for p in end_pats for m in p.finditer(text))

    if not starts:
        return Section(which, "", "noanchor", 0, 0)

    # Pair each start with the first end anchor that follows it.
    #
    # Spans WITHOUT a following end anchor are held separately, and this
    # separation is the whole correctness argument. An earlier version gave
    # unanchored starts a fixed 200,000-character window so a final section
    # would still be recoverable. Combined with "longest span wins", that
    # was a trap: a stray late cross-reference ("see Item 1A") sitting after
    # the real Item 1B has no end anchor after it, inherits the full window,
    # and beats every genuine span. The result is a section of exactly
    # 200,000 characters — a constant masquerading as a measurement, and one
    # that showed up as a spike of ~3,800 identical lengths on real data.
    #
    # So: if ANY start has a real end anchor after it, only those compete.
    # The unanchored fallback is used only when nothing else exists, and it
    # is reported as `unbounded` rather than `ok` so it can never silently
    # enter the year-over-year comparison.
    anchored = [(s, min(e for e in ends if e > s))
                for s in starts if any(e > s for e in ends)]

    # A section does not begin in the middle of a sentence.
    #
    # This filter exists because of a systematic 2024 artifact. The SEC's
    # 2023 cybersecurity rule added Item 1C, and Item 1C routinely contains
    # a cross-reference of the form:
    #
    #     ...see Item 1A, Risk Factors, for more information on our
    #     cybersecurity-related risks.
    #
    # The anchor matches inside that sentence, and the span it opens runs to
    # Item 2 — capturing Item 1C's governance discussion and presenting it
    # as risk factors. On the real panel the five largest year-over-year
    # dissimilarities were ALL 2024 filings, all of them this.
    #
    # The tell is unambiguous: the captured text starts with a comma or a
    # lowercase word, because it is a sentence fragment. Genuine sections
    # start with a capital letter, a quote, or a digit. Spans failing that
    # are dropped, and only if none survive do we fall back to the full set.
    sentence_start = [se for se in anchored if _starts_cleanly(text, se[0])]
    if sentence_start:
        anchored = sentence_start

    if anchored:
        best_start, best_end = max(anchored, key=lambda se: se[1] - se[0])
        bounded = True
    else:
        best_start = starts[-1]
        best_end = min(best_start + MAX_SECTION_CHARS, len(text))
        bounded = False

    body = text[best_start:best_end].strip()

    if not bounded:
        status = "unbounded" if len(body) >= MIN_SECTION_CHARS else "tooshort"
        return Section(which, body if status == "unbounded" else "",
                       status, len(body), len(starts))

    # A section longer than any real one is a missed end anchor, not a
    # verbose company. The longest Item 1A in a 10-K runs to roughly 150k
    # characters; past MAX_SECTION_CHARS we are certainly reading on into
    # the rest of the filing.
    if len(body) > MAX_SECTION_CHARS:
        return Section(which, "", "runon", len(body), len(starts))

    if len(body) < MIN_SECTION_CHARS:
        # Distinguish "the company omitted this section" from "we failed to
        # find it". The two mean different things and must not be pooled.
        window = text[best_start:best_start + 1200]
        status = "absent" if OMITTED.search(window) else "tooshort"
        return Section(which, body, status, len(body), len(starts))

    return Section(which, body, "ok", len(body), len(starts))


def extract_all(text: str) -> dict[str, Section]:
    return {k: extract(text, k) for k in ANCHORS}
