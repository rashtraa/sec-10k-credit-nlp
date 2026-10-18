"""
Offline tests for section extraction.

The case that matters most is the table of contents. A naive extractor takes
the FIRST "Item 1A. Risk Factors" it sees, which is the contents line, and
returns the few hundred characters between it and the contents line for Item
1B. That output is not empty and not obviously wrong — it is a short string
of plausible-looking text — so nothing downstream complains. The pipeline
runs green and every number it produces is meaningless.

These tests exist so that failure mode cannot ship.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sections import extract, extract_all, MIN_SECTION_CHARS

REAL_RISK_TEXT = (
    "Our business is subject to numerous risks. Competition in our markets "
    "is intense and increasing. We depend on a small number of suppliers. "
    "Adverse macroeconomic conditions could reduce demand for our products. "
) * 40           # ~9,000 chars — realistic order of magnitude

REAL_MDA_TEXT = (
    "Revenue for fiscal 2023 increased 12% to $4.2 billion, driven by volume "
    "growth in our North America segment. Gross margin declined 140 basis "
    "points due to input cost inflation. Liquidity remains adequate. "
) * 40

FILING = f"""ANNUAL REPORT PURSUANT TO SECTION 13

TABLE OF CONTENTS

PART I
Item 1. Business 3
Item 1A. Risk Factors 12
Item 1B. Unresolved Staff Comments 41
Item 7. Management's Discussion and Analysis of Financial Condition 55
Item 7A. Quantitative and Qualitative Disclosures About Market Risk 78

PART I

Item 1. Business

We design and manufacture industrial components. For additional detail on
the risks affecting our business, see Item 1A. Risk Factors below.

Item 1A. Risk Factors

{REAL_RISK_TEXT}

Item 1B. Unresolved Staff Comments

None.

Item 7. Management's Discussion and Analysis of Financial Condition and
Results of Operations

{REAL_MDA_TEXT}

Item 7A. Quantitative and Qualitative Disclosures About Market Risk

We are exposed to interest rate risk.
"""

# A smaller reporting company that legitimately omits Item 1A.
OMITTED_FILING = """TABLE OF CONTENTS
Item 1A. Risk Factors 8
Item 1B. Unresolved Staff Comments 8

PART I

Item 1A. Risk Factors

Not required for smaller reporting companies.

Item 1B. Unresolved Staff Comments

None.
"""


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    return cond


def test_toc_not_captured():
    print("table-of-contents discrimination:")
    s = extract(FILING, "item1a")
    ok = True
    ok &= check(f"status is ok (got {s.status!r})", s.ok)
    # Exactly two line-start anchors exist here: the contents entry and the
    # real heading. The third mention — "see Item 1A. Risk Factors below."
    # — sits mid-sentence and is correctly not counted, which is the point
    # of anchoring to line starts.
    ok &= check(f"saw both line-start anchors and chose between them "
                f"({s.n_candidates})", s.n_candidates == 2)
    ok &= check(f"length is realistic ({s.n_chars:,} chars)",
                s.n_chars > 5000)
    ok &= check("captured the real prose",
                "Adverse macroeconomic conditions" in s.text)
    ok &= check("did NOT capture the contents page-number line",
                "Unresolved Staff Comments 41" not in s.text)
    ok &= check("did NOT capture the cross-reference sentence",
                "see Item 1A" not in s.text)
    ok &= check("did not bleed into Item 1B",
                "Unresolved Staff Comments" not in s.text)
    return ok


def test_mda():
    print("\nItem 7 (MD&A):")
    s = extract(FILING, "item7")
    ok = True
    ok &= check(f"status is ok (got {s.status!r})", s.ok)
    ok &= check("captured MD&A prose", "Gross margin declined" in s.text)
    ok &= check("did not bleed into Item 7A",
                "exposed to interest rate risk" not in s.text)
    return ok


def test_omitted():
    print("\nlegitimately omitted section:")
    s = extract(OMITTED_FILING, "item1a")
    ok = True
    ok &= check(f"flagged 'absent', not 'ok' (got {s.status!r})",
                s.status == "absent")
    ok &= check("absent sections are not marked ok", not s.ok)
    return ok


def test_missing_anchor():
    print("\nno anchor at all:")
    s = extract("This document contains no item headings whatsoever.", "item1a")
    ok = check(f"flagged 'noanchor' (got {s.status!r})", s.status == "noanchor")
    return ok


def test_late_crossreference_does_not_win():
    """
    Regression test for the 200,000-character spike.

    A stray "Item 1A" reference AFTER the real Item 1B — an exhibit index, a
    signature block, a forward-looking-statements note — has no end anchor
    following it. An earlier version gave such starts a fixed 200k window,
    which then beat every genuine span under "longest wins", producing ~3,800
    filings all exactly 200,000 characters long.

    The real section here is ~9k chars. The trailing reference sits 300k
    characters later with nothing after it. The extractor must return the
    9k section, not a 300k one.
    """
    print("\nlate cross-reference after the real section:")
    doc = (FILING
           + "\n\n" + ("Financial statements and supplementary data. " * 6000)
           + "\n\nEXHIBIT INDEX\n\nRisks are discussed under Item 1A. "
             "Risk Factors of this Annual Report.\n")
    s = extract(doc, "item1a")
    ok = True
    ok &= check(f"status is ok (got {s.status!r})", s.ok)
    ok &= check(f"length is the real section, not a window "
                f"({s.n_chars:,} chars)", 5_000 < s.n_chars < 20_000)
    ok &= check("did not swallow the rest of the filing",
                "supplementary data" not in s.text)
    ok &= check("length is not the old fixed 200,000 window",
                s.n_chars != 200_000)
    return ok


def test_unbounded_is_not_ok():
    """A section with no end anchor anywhere must not be reported usable."""
    print("\nno end anchor anywhere:")
    doc = "PART I\n\nItem 1A. Risk Factors\n\n" + REAL_RISK_TEXT
    s = extract(doc, "item1a")
    ok = True
    ok &= check(f"flagged 'unbounded' (got {s.status!r})",
                s.status == "unbounded")
    ok &= check("unbounded does not count as ok", not s.ok)
    return ok


def test_runon_rejected():
    """A span past the plausibility bound is a parse failure, not a section."""
    print("\nimplausibly long span:")
    doc = ("Item 1A. Risk Factors\n" + ("risk " * 60_000)
           + "\nItem 1B. Unresolved Staff Comments\nNone.")
    s = extract(doc, "item1a")
    ok = check(f"flagged 'runon' (got {s.status!r}, {s.n_chars:,} chars)",
               s.status == "runon")
    ok &= check("runon does not count as ok", not s.ok)
    return ok


def test_end_anchor_alternatives():
    """
    Item 1B is often omitted, and filings after Dec 2023 add Item 1C
    (Cybersecurity). The section must still be bounded when the filer used a
    different next heading. This gap cost 731 filings on the first real run.
    """
    print("\nalternative end anchors:")
    ok = True
    variants = {
        "Item 1B (standard)":
            "Item 1A. Risk Factors\n\n{r}\n\nItem 1B. Unresolved Staff Comments\nNone.\n"
            "\nItem 2. Properties\nWe lease offices.",
        "Item 1C (post-2023 rule)":
            "Item 1A. Risk Factors\n\n{r}\n\nItem 1C. Cybersecurity\n"
            "We maintain a security program.\n\nItem 2. Properties\nWe lease offices.",
        "Item 2 only (1B and 1C both absent)":
            "Item 1A. Risk Factors\n\n{r}\n\nItem 2. Properties\nWe lease offices.",
    }
    for name, tmpl in variants.items():
        s = extract(tmpl.format(r=REAL_RISK_TEXT), "item1a")
        good = s.ok and 5_000 < s.n_chars < 20_000
        ok &= check(f"{name}  ->  {s.status}, {s.n_chars:,} chars", good)
        ok &= check(f"    did not swallow the next section",
                    "lease offices" not in s.text and "security program" not in s.text)
    return ok


def test_midsentence_crossreference_ignored():
    """
    Regression test for EPR 2020.

    The forward-looking-statements disclaimer in a large share of filings
    contains a phrase like:

        ...described under Item 1A, "Risk Factors" in this Annual Report...

    Before anchors were required to start a line, that mid-sentence mention
    matched, opened a span inside the disclaimer, and produced an 1,083
    character "risk factors section" of safe-harbour boilerplate. It went on
    to score the highest year-over-year dissimilarity in the entire panel
    (0.951), because it was compared against a real section from the year
    before.

    A fabricated extreme value at the top of the distribution is worse than
    a missing one: it is exactly what a reader inspects first.
    """
    print("\nmid-sentence cross-reference in the disclaimer:")
    doc = f"""FORWARD-LOOKING STATEMENTS

Our actual results could differ materially from those described under
Item 1A, "Risk Factors" in this Annual Report on Form 10-K. For these
statements we claim the safe harbor of the Reform Act of 1995.

Item 1B. Unresolved Staff Comments

None.

PART I

Item 1A. Risk Factors

{REAL_RISK_TEXT}

Item 2. Properties

We lease offices.
"""
    s = extract(doc, "item1a")
    ok = True
    ok &= check(f"status is ok (got {s.status!r})", s.ok)
    ok &= check(f"got the real section ({s.n_chars:,} chars)",
                s.n_chars > 5_000)
    ok &= check("did not anchor inside the disclaimer",
                "safe harbor" not in s.text)
    ok &= check("captured the real prose",
                "Adverse macroeconomic conditions" in s.text)
    return ok


def test_fragment_rejected():
    """A span far below any genuine section length is not a section."""
    print("\nfragment below the floor:")
    doc = ('Item 1A. Risk Factors\n' + ("boilerplate " * 60)
           + "\nItem 1B. Unresolved Staff Comments\nNone.")
    s = extract(doc, "item1a")
    ok = check(f"flagged 'tooshort' (got {s.status!r}, {s.n_chars:,} chars "
               f"vs floor {MIN_SECTION_CHARS:,})", s.status == "tooshort")
    ok &= check("fragment does not count as ok", not s.ok)
    return ok


def test_item1c_crossreference_2024():
    """
    Regression test for the 2024 cybersecurity artifact.

    The SEC's 2023 rule added Item 1C, and Item 1C routinely cross-references
    Item 1A mid-sentence. That match opens a span running to Item 2, which
    captures Item 1C's governance text and labels it risk factors. On the
    real panel the five largest year-over-year dissimilarities were ALL 2024
    filings hitting exactly this.
    """
    print("\nItem 1C cross-reference (2024 filings):")
    doc = f"""PART I

Item 1A. Risk Factors

{REAL_RISK_TEXT}

Item 1B. Unresolved Staff Comments

None.

Item 1C. Cybersecurity

Risk Management and Strategy. See Item 1A, Risk Factors, for more
information on our cybersecurity-related risks. Governance. On behalf of
the Board of Directors, the Audit Committee provides oversight of the
Company's management of cybersecurity risk. They periodically review and
discuss with management the Company's information security posture.

Item 2. Properties

We lease offices.
"""
    s = extract(doc, "item1a")
    ok = True
    ok &= check(f"status is ok (got {s.status!r})", s.ok)
    ok &= check("captured the real risk prose",
                "Adverse macroeconomic conditions" in s.text)
    ok &= check("did NOT capture Item 1C governance text",
                "Audit Committee" not in s.text)
    ok &= check("does not begin mid-sentence",
                not s.text.lstrip().startswith(","))
    return ok


def test_both():
    print("\nextract_all:")
    d = extract_all(FILING)
    return check("both sections extracted ok",
                 d["item1a"].ok and d["item7"].ok)


if __name__ == "__main__":
    results = [test_toc_not_captured(), test_mda(), test_omitted(),
               test_missing_anchor(), test_late_crossreference_does_not_win(),
               test_unbounded_is_not_ok(), test_runon_rejected(),
               test_end_anchor_alternatives(),
               test_midsentence_crossreference_ignored(),
               test_fragment_rejected(), test_item1c_crossreference_2024(),
               test_both()]
    print("\nALL PASS" if all(results) else "\nFAILURES ABOVE")
    sys.exit(0 if all(results) else 1)
