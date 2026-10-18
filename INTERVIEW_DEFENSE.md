# Interview defense pack

Every claim in this project, the question it invites, and the answer.
Ordered by how likely you are to be asked.

Two rules before the questions:

1. **Volunteer the limitation before they find it.** Your strongest move in
   an interview is the one you already use in your writeups — say what the
   method cannot do before anyone asks. Nobody dents that.
2. **If you do not know, say so and say what you would run.** "I did not
   test that; the way to test it would be X" is a good answer. Guessing is
   not.

---

## Tier 1 — you will almost certainly be asked these

### "Walk me through the project."

Sixty seconds, in this order. Do not start with the pipeline.

> Financial ratios predict corporate distress. I wanted to know whether the
> *text* of a 10-K adds anything once you already know the ratios — whether
> the way a company rewrites its risk factors carries information the
> balance sheet does not.
>
> I built the panel from SEC EDGAR: 8,801 annual reports, 1,499 firms, six
> years. Extracted the risk-factors section from each, then built two kinds
> of feature — Loughran-McDonald finance-sentiment rates, and the
> year-over-year cosine distance between a firm's risk factors and its own
> prior year.
>
> The answer is no. The text block moves out-of-time AUC by −0.0022 against
> a 0.7612 ratio baseline, p = 0.31. Risk-factor rewriting alone is −0.0006.
>
> That negative result only means something because the test had to prove
> itself first. I ran a permutation null and gated it on two controls: noise
> had to be promoted at roughly the nominal rate, and a variable known to
> drive the label — current profitability — had to be detected. Noise came
> in at 8% against α of 5%, and profitability at z = 10.3.

### "Your result is negative. Isn't that a failed project?"

> The failure mode I was guarding against is the opposite one. Adding any
> column to a model moves its AUC; adding five columns of anything moves it
> more. If I had reported "AUC went up when I added text" without a null,
> that number would have been meaningless — and that is how this analysis
> is usually presented.
>
> What I can say is stronger: text adds nothing, measured against a test
> that provably detects a real driver at z = 10.3 and promotes pure noise at
> the nominal rate. The finding is that a plausible hypothesis is false,
> and I can show the distribution that establishes it.

If pressed: **a null result costs you nothing if the test has power, and
demonstrating power is most of the work.**

### "How do you know your test actually works?"

This is the question the whole project is built to answer. Have both
numbers ready.

> Two controls, and they run before the verdict — the notebook prints VOID
> over every result if either misbehaves.
>
> **Negative:** fifty independent Gaussian blocks the same width as the text
> block, each through the full procedure. Four of fifty were promoted, 8%
> against a nominal 5%, binomial p = 0.24. And the mean delta across draws
> is −0.0066 — junk columns correctly *cost* accuracy.
>
> **Positive:** current profitability, held out of the baseline and offered
> back as a candidate. +0.0445 AUC, z = +10.3.
>
> A test that cannot reject noise certifies nothing. A test that cannot
> detect a known driver has no power.

### "Why a permutation null? Why not just a t-test on the coefficient?"

> Because the thing I want a null for is not a coefficient, it is the
> out-of-sample AUC gain from adding a *block* of five correlated features.
> There is no clean analytic distribution for that.
>
> The permutation gives it directly: shuffle the text block across firms
> within each year. That destroys the link between text and firm while
> preserving every marginal distribution, the correlations within the block,
> the year mix, the model form and the parameter count. A null draw differs
> from the real data in exactly one respect — which firm each text profile
> describes. Two hundred draws is the distribution of deltas produced by
> text that describes the wrong company.

### "Why shuffle *within* year rather than across the whole panel?"

> To hold the year mix fixed. Distress rates differ by year — 6.8% in 2020,
> 9.3% in 2023 — and shuffling across the whole panel would let a null draw
> differ from the real data in year composition as well as in the
> firm-to-text link. Within-year permutation leaves exactly one thing
> varying.

---

## Tier 2 — likely, and where the depth shows

### "What was the hardest bug?"

Tell the 200,000-character story. It is the best thing in the project.

> My first extraction run reported 91% success. The length distribution had
> a plausible mean and a sensible left tail.
>
> About 3,800 of those "sections" were exactly 200,000 characters long. I had
> written a fallback: if a section's start anchor has no end anchor after
> it, take a 200,000-character window so a final section is still
> recoverable. Combined with a longest-span-wins rule, that was a trap — a
> stray late cross-reference like "see Item 1A. Risk Factors" in an exhibit
> index has no end anchor after it, inherited the full window, and beat
> every genuine span.
>
> I found it by counting the most common *exact* length. Real companies do
> not write sections of identical length, so a constant that precise is
> coming from code, not data. That check is now a permanent cell, and the
> most common exact length today is five filings — 0.07%.

If they want more, you have four others: the mid-sentence anchor that
produced the highest dissimilarity in the dataset from safe-harbour
boilerplate; the Item 1C cross-reference that made all five top values 2024
filings; the crushed ratio; the label that cut the panel to 408 rows.

### "Why Loughran-McDonald rather than a general sentiment model?"

> Because general-purpose dictionaries misread financial text. In
> Harvard-IV, *liability*, *tax*, *cost*, *capital* and *board* all score
> negative — in a 10-K those are ordinary accounting vocabulary. Loughran
> and McDonald built the finance-specific lists for exactly that reason.
>
> It is also a choice I can justify, which matters more to me than a
> fancier model I could not.

### "Why interest coverage first, and why did you abandon it?"

> Coverage below 1 is the better credit metric — it says the firm cannot
> service its interest out of operating earnings. But `InterestExpense` is
> tagged by only 69% of my panel; most filers disclose interest in the
> notes rather than in XBRL. Requiring it at both *t* and *t+1*, plus
> coverage ≥ 1 at *t*, cut the panel to 408 firm-years from 197 firms — a
> 48-row test set with about five events.
>
> The positive control refused to certify it, correctly: no test has power
> at that size. So I moved to an operating-loss transition, which uses
> `OperatingIncomeLoss` — tagged by nearly everyone. It is a coarser event,
> but it is observable. A feature you cannot observe is not a feature.
>
> I report the 69% coverage figure in the notebook so the trade-off is
> auditable rather than hidden.

### "Your positive control failed the first time. What happened?"

This one shows judgment. Do not hide it.

> I used leverage, on the reasoning that leverage drives credit distress. It
> came back at +0.0005 AUC, z = 1.36 — no signal.
>
> That failure was correct. Leverage predicts *default* and coverage
> breaches, because leverage drives interest expense which drives coverage.
> It says very little about whether a firm's *operations* swing to a loss.
> I had changed the label from a coverage breach to an operating-loss
> transition and left the control matched to the old one.
>
> A positive control has to be a variable known to drive **the label you
> actually chose**, not credit risk in general. I switched to current
> profitability and kept leverage as a reported diagnostic, because
> "leverage does not predict operating losses" is a real result.

### "You said noise got promoted once. Did you just rerun until it passed?"

> No, and that would have been the worst possible response.
>
> A single noise draw tested at α = 0.05 is promoted 5% of the time by
> construction — that is what a 5% false-positive rate means. Read as a
> pass/fail gate on one draw, the negative control is a coin flip that voids
> the analysis once in twenty runs.
>
> So I changed the question from "was this draw promoted?" to "is noise
> promoted at roughly the nominal rate?" — fifty draws, judged by a one-
> sided binomial test. My first attempt at that tolerated "up to 3× nominal"
> and duly passed a run at exactly 3×, which is an arbitrary threshold
> dressed up as a criterion. The binomial test replaced it.

### "Why out-of-time rather than a random split?"

> Because two firm-years from the same year share macro conditions and, for
> a panel like this, largely share a shock. A randomly held-out row is not
> independent of its training neighbours, so a random split leaks and
> reports an optimistic AUC. Train is 2020–2022, test is 2023, strictly
> forward.

---

## Tier 3 — the questions designed to break it

### "Your test set is 601 rows with 56 events. Can you detect anything?"

Concede this cleanly. It is the strongest objection.

> That is the real limitation, and it is why I state the result as "no
> detectable effect" rather than "no effect". With 56 events I can exclude a
> large effect, not a small one — the interval around −0.0022 comfortably
> contains modest positive values.
>
> What the positive control establishes is that the test is not simply
> blind: it detects profitability at z = 10.3 on the same 601 rows. So the
> power exists for an effect of that size. Text is not near that size.

### "Isn't TF-IDF cosine distance a crude measure of textual change?"

> Yes, and it tests a specific hypothesis rather than the general one. It
> measures vocabulary overlap, so a firm that rewrites its risk factors in
> different words without changing the substance scores as a large change,
> and one that adds a single devastating sentence scores as almost none.
>
> A sentence-embedding measure would test a better hypothesis. I did not run
> it — it needs an embedding model over 7,000 sections, and I wanted the
> falsification machinery right before I made the feature more elaborate.
> That ordering was deliberate: a better feature judged by a test I had not
> validated would have been worth less.

### "Could survivorship bias explain the null?"

> It could push in that direction, yes. My panel requires a 10-K in all six
> years, which excludes firms that were acquired, delisted or failed —
> plausibly the most distressed ones, and precisely where risk-factor text
> would be most informative. So the design biases *against* finding signal.
>
> The balanced-panel requirement is not arbitrary — the year-over-year
> feature compares a firm to its own prior year, so a gap yields a
> discontinuity rather than an observation. But relaxing it and allowing
> unbalanced firms is the obvious robustness check and I have not run it.

### "16% of your sections came from a 'loose' anchor tier. Doesn't that contaminate the result?"

> It might, and I track it separately rather than folding it into the good
> bucket for exactly that reason. Those are cases where the heading matched
> mid-line rather than at the start of a line — usually because the filer
> put the heading in a layout table.
>
> I have not re-run the verdict on the strict subset alone. That is the
> first robustness check I would run, and it is in the limitations.

### "You winsorise at 1/99. Isn't that throwing away the extreme firms — the ones most likely to fail?"

> It bounds them rather than removes them; the rows stay, the values are
> clipped. And the alternative is worse than losing tail information.
>
> I winsorised the raw inputs but not the ratios built from them, and a firm
> with near-zero assets produced an ROA in the hundreds. After
> standardisation that single row compresses every other value toward zero.
> Isolated, 15 such rows in 1,500 moved ROA's univariate AUC from 0.80 to
> 0.20 — the feature does not weaken, it inverts.
>
> The symptom was my positive control reporting that current profitability
> adds +0.0001 to predicting next-year operating losses, which is close to
> mechanically impossible. That is what sent me looking.

### "Why calendar frames rather than each firm's fiscal year?"

> Cost. The XBRL frames API returns one concept for every filer in one
> period per request — about 60 requests for the whole panel. Per-firm
> `companyfacts` would be 1,500 requests returning roughly 5 GB of JSON to
> extract a dozen numbers each.
>
> The trade-off is that a June year-end firm does not map cleanly onto
> "fiscal 2023". For ratios compared within-firm across years that is
> immaterial, and I state it in the limitations rather than leaving it
> implicit.

### "What would you do next?"

Have three, ordered.

> First, sentence embeddings instead of TF-IDF — that tests whether the
> *meaning* changed rather than the vocabulary, and it is the version of the
> hypothesis I actually believe.
>
> Second, a better label. Rating downgrades or actual defaults, which need
> data I did not have free access to. Coverage below 1 if I could source
> interest expense properly.
>
> Third, relax the balanced-panel requirement and re-run, to see whether the
> null survives when delisted and failed firms are allowed back in.

---

## Numbers to have cold

| | |
|---|---|
| Filings downloaded | 8,801 of 8,878 (99.1%), 1,007 MB |
| Firms · years | 1,499 · 2019–2024 |
| Item 1A usable | 7,027 (80.8%) |
| Year-over-year observations | 5,310 across 1,352 firms |
| Modelling panel | 2,097 firm-years, 759 firms, 8.58% distress |
| Train / test | 2020–22 (1,496) / 2023 (601, 56 events) |
| Baseline AUC | 0.7612 |
| Text block | Δ −0.0022, z +0.48, p 0.313 |
| risk_change alone | Δ −0.0006, z −0.09, p 0.687 |
| Positive control (roa) | Δ +0.0445, z +10.30 |
| Negative control | 4 of 50 promoted (8%) vs α 5%, binomial p 0.240 |
| Strongest single feature | log_assets, univariate AUC 0.7191 |

---

## The one-sentence version

> I tested whether 10-K risk-factor language predicts corporate
> deterioration beyond financial ratios, built a permutation test that had
> to prove it could reject noise and detect a known driver before I would
> believe it, and the answer was no — with the null distribution to show it.
