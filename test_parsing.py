"""Offline sanity checks for the two parsers that everything else depends on."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from build_universe import IDX_LINE
from download_filings import html_to_text

# ---------------------------------------------------------------- form.idx
# Real lines, copied from published quarterly indexes across several years,
# including the ragged-spacing cases that break fixed-offset parsers.
IDX_SAMPLES = [
    ("10-K        Apple Inc.                                                    320193     2024-11-01  edgar/data/320193/0000320193-24-000123.txt",
     dict(cik="320193", date="2024-11-01")),
    ("10-K        1 800 FLOWERS COM INC                                         1084869    2023-09-14  edgar/data/1084869/0001084869-23-000038.txt",
     dict(cik="1084869", date="2023-09-14")),
    ("10-K        BANK OF AMERICA CORP /DE/                                     70858      2024-02-20  edgar/data/70858/0000070858-24-000027.txt",
     dict(cik="70858", date="2024-02-20")),
    # Must NOT match — amendment and legacy variants
    ("10-K/A      SOME COMPANY INC                                              999999     2024-03-01  edgar/data/999999/0000999999-24-000001.txt",
     None),
]

def test_idx():
    ok = True
    for line, want in IDX_SAMPLES:
        m = IDX_LINE.match(line)
        form = m.group("form").strip() if m else None
        accepted = bool(m) and form == "10-K"
        if want is None:
            if accepted:
                print(f"  FAIL: should have been rejected -> {form!r}")
                ok = False
            else:
                print(f"  ok   rejected {form!r}")
        else:
            if not accepted:
                print(f"  FAIL: not matched: {line[:60]}")
                ok = False
            elif m.group("cik") != want["cik"] or m.group("date") != want["date"]:
                print(f"  FAIL: wrong fields cik={m.group('cik')} date={m.group('date')}")
                ok = False
            else:
                print(f"  ok   cik={m.group('cik'):>8}  {m.group('name')[:35]}")
    return ok


# ---------------------------------------------------------------- html
HTML = """<html><head><style>.x{color:red}</style><script>var a=1;</script></head>
<body>
<ix:header><ix:hidden><ix:nonNumeric>SHOULD NOT APPEAR</ix:nonNumeric></ix:hidden></ix:header>
<div>PART I</div>
<table><tr><td>Item&nbsp;1A.</td><td>Risk Factors</td></tr></table>
<p>Our business faces   substantial   risks, including&nbsp;competition.</p>
<div style="display:none">HIDDEN BOILERPLATE</div>
<p>Item 7. Management&#8217;s Discussion and Analysis</p>
<p>Revenue increased 12%.</p>
</body></html>"""

def test_html():
    txt = html_to_text(HTML)
    print("\n--- extracted ---")
    print(txt)
    print("--- end ---\n")
    checks = {
        "script content removed":      "var a=1" not in txt,
        "style content removed":       "color:red" not in txt,
        "ix:header payload removed":   "SHOULD NOT APPEAR" not in txt,
        "nbsp normalised":             "\xa0" not in txt,
        "whitespace collapsed":        "substantial   risks" not in txt,
        "Item 1A on its own line":     any(l.strip().startswith("Item 1A")
                                           for l in txt.split("\n")),
        "hidden div removed":          "HIDDEN BOILERPLATE" not in txt,
        "td cells space-separated":    "Item 1A. Risk Factors" in txt,
        "Item 7 anchor present":       "Item 7." in txt,
        "prose preserved":             "Revenue increased 12%." in txt,
    }
    ok = True
    for name, passed in checks.items():
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
        ok &= passed
    return ok


if __name__ == "__main__":
    print("form.idx parser:")
    a = test_idx()
    print("\nhtml_to_text:")
    b = test_html()
    print("\nALL PASS" if (a and b) else "\nFAILURES ABOVE")
    sys.exit(0 if (a and b) else 1)
