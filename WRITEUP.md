# Does 10-K risk-factor language carry credit signal the balance sheet does not?

**A falsification study on 8,801 SEC annual reports, 1,499 firms, 2019–2024.**

---

## Summary

Financial ratios predict corporate distress. The interesting question is
whether the *text* of a 10-K adds anything once those ratios are known —
and the honest way to ask it is to build a test that has to prove it can
detect a real effect and reject a fake one before its verdict is worth
anything.

The answer here is **no**, and the test earned the right to say so.

| | |
|---|---|
| **Risk-factor text adds no incremental signal** | ΔAUC **−0.0022** (z +0.48, p 0.313) over a 0.7612 ratio baseline, out-of-time |
| **Nor does year-over-year rewriting on its own** | ΔAUC **−0.0006** (z −0.09, p 0.687) |
| **The test can detect a real driver** | Profitability, held out and added back: **+0.0445, z +10.30** |
| **The test rejects noise at the nominal rate** | 50 independent noise blocks promoted **4 times (8%)** against α = 5%, binomial p 0.240 |

Four extraction and methodology artifacts were found and killed on the way,
each by a diagnostic built to catch it. The largest fabricated **~3,800
identical 200,000-character "sections"** and was invisible in every
aggregate.

---

## 1 · The question, and why it needs a null

Adding columns to a model moves its AUC. Adding five columns of anything
moves it more. So "AUC went up when I added text" is not evidence, and the
usual presentation of exactly that number is the reason this project exists.

The statistic here is the out-of-time AUC difference between a logistic
regression on financial ratios and the same model with a text block
appended. That delta is then judged against a **permutation null**: the text
block is shuffled **across firms within each year**, destroying the
firm-specific link while leaving untouched

- every marginal distribution in the block,
- the correlations *within* the block (rows move together),
- the year mix,
- the model form, and
- the parameter count.

A null draw therefore differs from the real data in exactly one respect:
which firm each text profile belongs to. Two hundred draws give the
distribution of deltas produced by text that describes the wrong company.

---

## 2 · Data

| | |
|---|---|
| 10-K filings scanned (24 quarterly EDGAR indexes) | 42,909 |
| Firms with a traded ticker | 4,606 |
| Firms filing in all six years (balanced panel) | 3,190 |
| **Panel** | **1,499 firms · 8,878 filings** |
| Downloaded | 8,801 (99.1%), 1,007 MB gzipped text |

The balanced-panel requirement is not tidiness. The headline text feature
compares a firm's risk factors to *its own* prior year, so a filing gap
produces a discontinuity rather than an observation.

Filings are stripped to text on ingestion. The raw HTML for this panel is
50–150 GB; nothing downstream reads the markup.

### Section extraction

| status | filings | share |
|---|---|---|
| `ok` | 5,639 | 64.8% |
| `ok_loose` | 1,388 | 16.0% |
| `noanchor` | 1,242 | 14.3% |
| `tooshort` | 300 | 3.4% |
| `absent` | 99 | 1.1% |
| `unbounded` | 16 | 0.2% |
| `runon` | 15 | 0.2% |

**7,027 usable (80.8%)**, yielding **5,310 year-over-year observations
across 1,352 firms**. Statuses are kept separate rather than pooled: a
company that legitimately omitted Item 1A and a filing we failed to parse
mean different things, and merging them would put parsing noise into the
comparison the project rests on.

---

## 3 · Features

**Loughran-McDonald category rates** — Negative, Uncertainty, Litigious and
Constraining word counts as a share of Item 1A. The finance-specific
dictionary is used for a citable reason: in the general-purpose Harvard-IV
list, *liability*, *tax*, *cost*, *capital* and *board* all score negative,
which in a 10-K is ordinary accounting vocabulary.

**Year-over-year risk-factor dissimilarity** (`risk_change`) — cosine
distance between a firm's Item 1A and its own prior year, on TF-IDF vectors
fitted over the corpus. Risk sections are overwhelmingly boilerplate copied
forward, so substantial rewriting means something happened.

| | |
|---|---|
| observations | 5,310 |
| mean / median | 0.1233 / 0.0851 |
| p10 / p90 | 0.0320 / 0.2729 |
| copied near-verbatim (< 0.05) | 1,236 |

The feature is **within-firm** by construction, so stable differences in
house drafting style difference out, and strictly **backward-looking** —
year *t* is compared only to *t−1*.

---

## 4 · The test

**Label.** An operating loss next year (EBIT < 0 at *t+1*) among firms
profitable now (EBIT ≥ 0 at *t*), so the model predicts **transitions into**
deterioration rather than persistence of a state it can already observe.

**Split.** Out-of-time, not random: train 2020–2022 (1,496 rows, 8.29%
distress), test 2023 (601 rows, 9.32%). Two firm-years from the same year
share macro conditions, so a random holdout leaks.

**Panel.** 2,097 firm-years, 759 firms, 8.58% distress rate.

### The controls run first, and they gate everything

**Negative control — noise must be promoted at roughly α, not never.**
Fifty independent Gaussian blocks of the same width as the text block, each
put through the full procedure. The question is not whether any single draw
is promoted — at α = 0.05 one draw is promoted 5% of the time *by
definition* — but whether the promotion rate matches α, judged by a
one-sided binomial test.

```
promoted at alpha=0.05     4 of 50  (8.0%)
expected under H0          2.5      (5%)
binomial p (one-sided)     0.2396
mean delta across draws    -0.0066     <- junk columns correctly COST accuracy
-> CALIBRATED
```

**Positive control — a variable known to drive *this* label.** Current
profitability, held out of the baseline and offered back as a candidate.

```
AUC without roa   0.7167
AUC with roa      0.7612      delta +0.0445
null mean         -0.0012 +/- 0.0044
z +10.30    p 0.0050    -> SIGNAL
```

Only with both in hand is any verdict below meaningful.

### Result

| block | AUC base | AUC + block | Δ | z | p | verdict |
|---|---|---|---|---|---|---|
| noise ×50 (control) | — | — | −0.0066 | — | 8% vs 5% | **CALIBRATED** |
| roa (control) | 0.7167 | 0.7612 | **+0.0445** | **+10.30** | 0.005 | **SIGNAL** |
| leverage (diagnostic) | 0.7589 | 0.7612 | +0.0023 | +0.85 | 0.095 | no signal |
| **text block** | 0.7612 | 0.7589 | **−0.0022** | **+0.48** | **0.313** | **NO SIGNAL** |
| **risk_change alone** | 0.7612 | 0.7606 | **−0.0006** | −0.09 | 0.687 | **NO SIGNAL** |

Univariate out-of-time AUCs say the same from the other direction:

| feature | AUC | | feature | AUC |
|---|---|---|---|---|
| log_assets | **0.7191** | | lm_litigious | 0.5466 |
| roa | **0.6963** | | lm_negative | 0.5267 |
| leverage | **0.6598** | | lm_uncertainty | 0.5205 |
| cash_ratio | 0.5893 | | risk_change | 0.5138 |
| | | | lm_constraining | 0.4721 |

Every text feature sits between 0.47 and 0.55 — near-random alone, and
nothing joint emerges.

**Two secondary findings worth stating.** Firm size is the single strongest
predictor of an operating-loss transition. And *leverage does not predict
operating losses either* (z +0.85) — it predicts **default** and coverage
breaches, a different event, through a different channel.

---

## 5 · What the data killed

Five beliefs, each ended by a specific calculation rather than by argument.
Three were positions this project held at the time.

| Hypothesis | Killed by |
|---|---|
| Risk-factor rewriting predicts deterioration beyond ratios | ΔAUC −0.0006, p 0.687, against a null that detects ROA at z 10.3 |
| The LM category rates add what rewriting does not | Block ΔAUC −0.0022, p 0.313; every univariate AUC in [0.47, 0.55] |
| Leverage is a valid positive control for this label | z +0.85 — leverage predicts default, not operating losses |
| A single noise draw is a sufficient negative control | It is promoted α of the time by construction; 3 of 20 passed a 3×-α threshold that certified nothing |
| An extraction reporting 91% success is working | ~3,800 of those "sections" were exactly 200,000 characters long |

---

## 6 · The artifacts, and the diagnostics that caught them

Each of these produced output that was not empty, not obviously wrong, and
would have passed unnoticed into the result.

### 6.1 The 200,000-character window

The extractor gave a start anchor with no following end anchor a fixed
200,000-character window, so a final section would still be recoverable.
Combined with *longest span wins*, this was a trap: a stray late
cross-reference (`"see Item 1A. Risk Factors"` in an exhibit index, sitting
after the real Item 1B) has no end anchor after it, inherited the full
window, and **beat every genuine span**.

The first run reported 91.0% success. The mean was plausible, the
distribution had a sensible left tail, and roughly **half of the "usable"
extractions were fabricated**.

**Caught by counting the most common *exact* length.** Real companies do not
write sections of identical length; a constant that precise comes from code.
After the fix the most common exact length is 5 filings (0.07%).

### 6.2 The mid-sentence anchor

The forward-looking-statements disclaimer of a large share of filings
contains `...described under Item 1A, "Risk Factors" in this Annual
Report...`. The anchor matched inside that sentence and returned a
1,083-character fragment of safe-harbour boilerplate — which then scored the
**single highest year-over-year dissimilarity in the entire dataset
(0.951)**, because it was compared against a genuine section from the prior
year.

A fabricated value at the *top* of a distribution is worse than a missing
one: it is the first thing anyone inspects. **Caught by reading the extreme
tail by hand.** Fixed by requiring anchors to begin a line, and by raising
the minimum section length from 1,000 to 5,000 characters — the observed
10th percentile of genuine sections is 25,166.

### 6.3 The Item 1C cross-reference (2024 only)

The SEC's 2023 cybersecurity rule added Item 1C, which routinely says *"see
Item 1A, Risk Factors, for more information on our cybersecurity-related
risks."* Anchoring there captured Item 1C's governance discussion and
labelled it risk factors. **All five of the largest dissimilarities became
2024 filings.** The tell was that the captured text began with a comma:
sections do not start mid-sentence.

### 6.4 The crushed ratio

Winsorising Assets and EBIT separately does not bound their quotient. A firm
with near-zero assets yields an ROA in the hundreds, and after
standardisation that single row compresses every other value toward zero.

The symptom was a positive control reporting that **current profitability
adds +0.0001 AUC to predicting next-year operating losses** — close to
mechanically impossible. Isolated, 15 such rows in 1,500 moved ROA's
univariate AUC from 0.80 to **0.20**: the feature does not weaken, it
inverts. Ratios are now winsorised at 1/99 alongside their inputs.

### 6.5 The label that ate the panel

The first label used interest coverage below 1, the better credit metric.
It required `InterestExpense` tagged at both *t* and *t+1*, and most filers
report interest only in the notes. The panel came out at **408 firm-years
from 197 firms** — a 48-row test set with about five events.

The positive control refused to certify it, correctly: no test has power at
that size. **An attrition table** now reports survivors after each filter,
so the expensive constraint is visible rather than inferred.

---

## 7 · Limitations

**A null result is not proof of absence.** With 601 test rows and 56 events,
this study can exclude a large effect, not a small one. The confidence
interval around −0.0022 comfortably contains modest positive values.

**One label, one horizon.** An operating-loss transition is a coarse event.
Text might carry signal for default, for rating downgrades, or at horizons
other than one year — none of which is tested here. Coverage below 1 would
have been the better metric and was abandoned for data availability, which
is a real limitation and not a preference: only **68.7%** of the final panel
has a tagged `InterestExpense`.

**Calendar frames, not fiscal years.** The XBRL frames API assigns filings to
calendar periods, so a June year-end firm does not map cleanly onto "fiscal
2023". This is immaterial for within-firm comparisons and was chosen over
per-firm `companyfacts` requests, which would have cost ~5 GB of JSON to
extract a dozen numbers per firm-year.

**Bag-of-words, not meaning.** TF-IDF cosine distance measures vocabulary
overlap. A firm that rewrites its risk factors in different words without
changing the substance scores as a large change; one that adds a single
devastating sentence scores as almost none. A sentence-embedding measure
would test a different and probably better hypothesis.

**16% of usable sections come from the fallback anchor tier** (`ok_loose`),
which rests on weaker evidence than a line-start heading. These are counted
separately but are not excluded, and the result has not been re-run on the
strict subset alone.

**Survivorship.** The balanced-panel requirement keeps only firms that filed
in all six years, which excludes firms that were acquired, delisted or
failed — plausibly the most distressed ones. This biases against finding
signal precisely where text would be most informative.

---

## 8 · Reproducing

```
01_ingest.ipynb          universe, download           ~60 min (rate-limited)
02_sections.ipynb        Item 1A / Item 7 extraction  ~4 min
03_text_features.ipynb   LM rates, risk_change        ~3 min
04_signal_test.ipynb     label, controls, verdict     ~2 min

fundamentals.py          XBRL frames -> ratios        ~2 min
test_sections.py         8 extraction failure modes   10 s, offline
test_signal_controls.py  validates the permutation machinery
```

Stack: `pandas`, `numpy`, `scikit-learn`, `scipy`, `pysentiment2`, `lxml`,
`matplotlib`. Data comes from SEC EDGAR and is not in the repo.

Every stage writes to disk and is resumable. `python status.py` reports what
exists and what runs next.
