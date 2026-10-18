# edgar-credit-signal

Does the language a company uses in its 10-K risk factors carry credit
signal its financial ratios do not already contain?

Built on 8,801 SEC annual reports — 1,499 firms, 2019–2024.

**Full write-up: [WRITEUP.md](WRITEUP.md)** ·
**Interview defense pack: [INTERVIEW_DEFENSE.md](INTERVIEW_DEFENSE.md)**

---

## The finding in 60 seconds

**No.** Risk-factor text adds nothing to predicting corporate deterioration
once financial ratios are known.

```
text block      delta AUC  -0.0022   z +0.48   p 0.313   NO SIGNAL
risk_change     delta AUC  -0.0006   z -0.09   p 0.687   NO SIGNAL
```

against an out-of-time ratio baseline of **0.7612 AUC**.

That negative result is only worth something because the test had to earn
the right to report it. Two controls run **before** the verdict and gate it:

```
noise x50   promoted 4 of 50 (8.0%) vs alpha 5%, binomial p 0.240   CALIBRATED
roa         delta AUC +0.0445   z +10.30   p 0.005                  SIGNAL
```

A test that cannot reject noise certifies nothing. A test that cannot detect
a variable known to drive the label has no power. This one does both.

---

## The number that mattered most

The first extraction run reported **91.0% success**. The length distribution
had a plausible mean and a sensible left tail.

About **3,800 of those "sections" were exactly 200,000 characters long** — a
fallback window in my own code, fabricating text that beat every genuine
extraction under a longest-span rule.

It was caught by counting the **most common exact length**. Real companies
do not write sections of identical length; a constant that precise comes
from code, not from data. That check is now a permanent cell in
`02_sections.ipynb`, and the most common exact length today is 5 filings
(0.07%).

Four more artifacts were found the same way — each producing output that was
not empty, not obviously wrong, and would have passed silently into the
result. [WRITEUP.md §6](WRITEUP.md) has all five.

---

## Results

### Panel

| | |
|---|---|
| 10-K filings scanned (24 quarterly EDGAR indexes) | 42,909 |
| Panel after ticker + balanced-year filters | **1,499 firms · 8,878 filings** |
| Downloaded | 8,801 (99.1%), 1,007 MB gzipped |
| Item 1A extracted usably | **7,027 (80.8%)** |
| Year-over-year observations | **5,310** across 1,352 firms |
| Modelling panel (label + features complete) | 2,097 firm-years, 759 firms |

### Signal test

| block | AUC base | AUC + block | Δ | z | p | verdict |
|---|---|---|---|---|---|---|
| noise ×50 (control) | — | — | −0.0066 | — | 8% vs 5% | **CALIBRATED** |
| roa (control) | 0.7167 | 0.7612 | **+0.0445** | **+10.30** | 0.005 | **SIGNAL** |
| leverage (diagnostic) | 0.7589 | 0.7612 | +0.0023 | +0.85 | 0.095 | no signal |
| **text block** | 0.7612 | 0.7589 | **−0.0022** | +0.48 | 0.313 | **NO SIGNAL** |
| **risk_change alone** | 0.7612 | 0.7606 | **−0.0006** | −0.09 | 0.687 | **NO SIGNAL** |

### Univariate out-of-time AUC

| feature | AUC | feature | AUC |
|---|---|---|---|
| log_assets | **0.7191** | lm_litigious | 0.5466 |
| roa | **0.6963** | lm_negative | 0.5267 |
| leverage | **0.6598** | lm_uncertainty | 0.5205 |
| cash_ratio | 0.5893 | risk_change | 0.5138 |
| | | lm_constraining | 0.4721 |

Firm size is the strongest single predictor. Every text feature is
near-random on its own, and nothing joint emerges.

---

## Method, briefly

**Label** — an operating loss next year among firms profitable now, so the
model predicts *transitions into* deterioration rather than persistence of a
state it can already see.

**Split** — out-of-time (train 2020–22, test 2023), not random: two
firm-years from the same year share macro conditions, so a random holdout
leaks.

**Null** — the text block is shuffled **across firms within each year**.
That destroys the link between text and firm while preserving every marginal
distribution, the correlations within the block, the year mix, the model
form and the parameter count. A null draw differs from the real data in
exactly one respect: which firm each text profile describes.

**Why a null at all** — adding any column moves the AUC, and adding five
columns of anything moves it more. "AUC improved when I added text" is not
evidence.

---

## Repo

```
01_ingest.ipynb           universe, rate-limited download    ~60 min
02_sections.ipynb         Item 1A / 7 extraction             ~4 min
03_text_features.ipynb    LM rates, risk_change              ~3 min
04_signal_test.ipynb      label, controls, verdict           ~2 min

sections.py               extraction: two anchor tiers, length audit
text_features.py          Loughran-McDonald + TF-IDF YoY distance
fundamentals.py           SEC XBRL frames -> ratios          ~2 min
signal_test.py            panel, permutation null, controls
viz.py                    chart style (CVD-validated palette)
status.py                 what is built, what runs next

test_sections.py          8 extraction failure modes, offline, 10 s
test_signal_controls.py   proves the null rejects noise and detects signal
```

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-stage1.txt -r requirements-stage2.txt
python status.py
```

Every stage writes to disk and is resumable. Data comes from SEC EDGAR
(free, no key — a User-Agent with a real email is required) and is not in
the repo.

---

## Caveats

A null result is not proof of absence: 601 test rows and 56 events exclude a
large effect, not a small one. One label, one horizon, one measure of
textual change — TF-IDF cosine distance counts vocabulary overlap, so a firm
that adds one devastating sentence scores as almost no change. The balanced-
panel requirement excludes firms that delisted or failed, plausibly the most
distressed ones. Full discussion in
[WRITEUP.md §7](WRITEUP.md).
