"""
Stage 4b — does risk-factor text predict credit deterioration beyond the
balance sheet?

THE QUESTION
------------
Financial ratios already predict distress. The question is not whether text
predicts anything — a model with text in it will always score at least as
well in-sample. It is whether text adds information the ratios do not
already contain, on data the model has not seen.

THE LABEL
---------
An OPERATING LOSS next year, among firms operating at a profit now:
`distress` is EBIT < 0 in year t+1, predicted from year-t information, with
the sample restricted to EBIT >= 0 at t. So the model predicts TRANSITIONS
into deterioration, not persistence of a state it can already see.

Interest coverage below 1 is the better credit metric and was the first
choice. It was abandoned because InterestExpense is tagged by only a
minority of filers, and requiring it at both t and t+1 cut the panel to 408
firm-years — a 48-row test set that the positive control correctly refused
to certify.

THE TEST
--------
Fit a logistic regression on ratios alone, then on ratios plus text, on an
out-of-time split. The statistic is the AUC difference.

A raw improvement proves nothing: adding any column to a model moves the
AUC, and adding several columns of anything moves it more. So the delta is
judged against a PERMUTATION NULL. The text block is shuffled ACROSS FIRMS
WITHIN EACH YEAR, which destroys the firm-specific information while leaving
every marginal distribution, the year mix, the model form and the number of
parameters exactly as they were. Refit, recompute, repeat. The resulting
distribution is what this delta looks like when the text carries no
information at all.

Two controls run alongside, and they are not optional:

  NEGATIVE CONTROL — many independent blocks of pure Gaussian noise. Noise
  must be promoted at roughly the nominal rate, judged by a binomial test.
  Not "was this draw promoted?" — a single draw is promoted alpha of the
  time by construction, which makes a one-draw gate a coin flip.

  POSITIVE CONTROL — current profitability (ROA), held out of the baseline
  and added back as a candidate. It must PASS. A test that cannot detect a
  variable known to drive THIS label has no power. Leverage was the first
  choice and failed, correctly: leverage predicts default and coverage
  breaches, not operating losses. The control must match the label.

If either control misbehaves, every verdict in this file is void.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("signal")

# Ratio features. Every one is built from Assets, Liabilities, Cash and
# OperatingIncomeLoss — the four XBRL concepts with near-universal coverage.
#
# `coverage` (EBIT/interest) and `rev_growth` were in this list originally
# and were removed. Not because they are uninformative — coverage is the
# best single credit ratio there is — but because requiring them shrank the
# panel from ~8,800 firm-years to 408. InterestExpense is frequently
# reported only in the notes rather than tagged, and every additional
# required column multiplies the attrition. A feature you cannot observe is
# not a feature.
RATIO_COLS = ["log_assets", "leverage", "roa", "cash_ratio"]
TEXT_COLS = ["risk_change", "lm_negative", "lm_uncertainty",
             "lm_litigious", "lm_constraining"]

# Columns needed to BUILD the panel but not fed to the model.
_LABEL_INPUTS = ["ebit", "assets"]


# ────────────────────────────────────────────────────────── panel assembly
def build_panel(fund: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    """Merge fundamentals with text features and construct ratios + label."""
    f = fund.rename(columns={"year": "filing_year"}).copy()

    # Winsorise the raw inputs before ratios are formed. XBRL contains
    # genuine reporting errors (unit slips of 1e3, sign flips), and a single
    # one can dominate a logistic fit.
    for c in ["assets", "liabilities", "equity", "cash", "ebit",
              "interest", "revenue", "cfo", "net_income"]:
        if c in f:
            lo, hi = f[c].quantile([0.001, 0.999])
            f[c] = f[c].clip(lo, hi)

    f = f.sort_values(["cik", "filing_year"])
    f["log_assets"] = np.log(f["assets"].where(f["assets"] > 0))
    f["leverage"] = f["liabilities"] / f["assets"].where(f["assets"] > 0)
    f["roa"] = f["ebit"] / f["assets"].where(f["assets"] > 0)
    f["cash_ratio"] = f["cash"] / f["assets"].where(f["assets"] > 0)

    # Interest expense is reported as a positive magnitude by most filers and
    # negative by some. Absolute value, and require it to be material —
    # a firm with essentially no debt has no meaningful coverage ratio and
    # cannot transition into interest-coverage distress.
    interest = f["interest"].abs()
    interest = interest.where(interest > 0.001 * f["assets"].abs())
    f["coverage"] = f["ebit"] / interest

    f["rev_growth"] = f.groupby("cik")["revenue"].pct_change()

    # ---- the label -------------------------------------------------------
    # An OPERATING LOSS next year, among firms operating at a profit now.
    #
    # The first version of this used interest coverage below 1, which is the
    # better credit metric on paper. In practice InterestExpense is tagged
    # by only a minority of filers, and requiring it at BOTH t and t+1 —
    # plus coverage >= 1 at t — cut the panel to 408 firm-years from 197
    # firms. A 48-row test set cannot detect leverage, let alone text, and
    # the positive control correctly refused to certify it.
    #
    # OperatingIncomeLoss is tagged by nearly everyone. "A profitable firm
    # posts an operating loss next year" is a coarser event than a coverage
    # breach, but it is observable, it is unambiguous, and it is a genuine
    # deterioration. Coverage survives as a reported diagnostic below.
    f["ebit_next"] = f.groupby("cik")["ebit"].shift(-1)
    f["year_next"] = f.groupby("cik")["filing_year"].shift(-1)
    valid_next = f["year_next"] == f["filing_year"] + 1      # consecutive only

    f["distress"] = np.where(valid_next & f["ebit_next"].notna(),
                             (f["ebit_next"] < 0).astype(float), np.nan)

    panel = f.merge(
        text[["cik", "filing_year", "ticker"] + TEXT_COLS],
        on=["cik", "filing_year"], how="inner")

    # Predict TRANSITIONS, not persistence. A firm already loss-making is
    # trivially likely to still be loss-making, and keeping those inflates
    # every AUC here without anyone learning anything.
    panel = panel[panel["ebit"] >= 0]

    need = RATIO_COLS + TEXT_COLS + ["distress"]
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=need)

    # Winsorise the RATIOS, not just the inputs they were built from.
    #
    # Clipping assets and EBIT separately does not bound their quotient. A
    # firm with near-zero assets produces an ROA in the hundreds, and after
    # StandardScaler that single row compresses every other value toward
    # zero — the feature still exists but carries almost no usable variation.
    #
    # This was not hypothetical. Without it, ROA added +0.0001 AUC as a
    # positive control on the real panel: current profitability apparently
    # failing to predict next-year operating losses, which is close to
    # mechanically impossible. The feature was not weak, it was crushed.
    for c in RATIO_COLS:
        lo, hi = panel[c].quantile([0.01, 0.99])
        panel[c] = panel[c].clip(lo, hi)

    return panel.reset_index(drop=True)


def attrition(fund: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    """
    Where do the rows go?

    A panel that arrives 95% smaller than its inputs is the single most
    common way an analysis quietly loses its power, and an aggregate row
    count will not tell you which filter did it. This reports the survivors
    after each step so the expensive constraint is visible rather than
    inferred.
    """
    f = fund.rename(columns={"year": "filing_year"}).copy()
    f = f.sort_values(["cik", "filing_year"])
    f["ebit_next"] = f.groupby("cik")["ebit"].shift(-1)
    f["year_next"] = f.groupby("cik")["filing_year"].shift(-1)

    steps = [("fundamentals rows", f)]
    a = f[f["assets"].notna() & (f["assets"] > 0)]
    steps.append(("has Assets > 0", a))
    b = a[a["ebit"].notna()]
    steps.append(("has OperatingIncomeLoss", b))
    c = b[b["liabilities"].notna() & b["cash"].notna()]
    steps.append(("has Liabilities and Cash", c))
    d = c[c["year_next"] == c["filing_year"] + 1]
    steps.append(("has a consecutive next year", d))
    e = d[d["ebit_next"].notna()]
    steps.append(("has EBIT next year (label)", e))
    g = e[e["ebit"] >= 0]
    steps.append(("profitable now (transition only)", g))
    h = g.merge(text[["cik", "filing_year"] + TEXT_COLS],
                on=["cik", "filing_year"], how="inner")
    steps.append(("merged with text features", h))
    i = h.dropna(subset=TEXT_COLS)
    steps.append(("text features all present", i))

    rows = []
    prev = None
    for name, d_ in steps:
        n = len(d_)
        rows.append({"step": name, "rows": n,
                     "firms": d_["cik"].nunique(),
                     "lost": "" if prev is None else f"-{prev - n:,}",
                     "kept": "" if prev is None else f"{n/prev:.0%}"})
        prev = n
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────── the model
def _fit_auc(Xtr, ytr, Xte, yte, seed=0) -> float:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
    )
    model.fit(Xtr, ytr)
    return roc_auc_score(yte, model.predict_proba(Xte)[:, 1])


def delta_auc(panel: pd.DataFrame, base_cols: list[str],
              add_cols: list[str], test_year: int,
              add_override: np.ndarray | None = None) -> tuple[float, float, float]:
    """
    Returns (auc_base, auc_augmented, delta) on an out-of-time split.

    `add_override` substitutes values for the added block — that is how the
    permutation null and the noise control are injected without touching any
    other part of the fit.
    """
    tr = panel["filing_year"] < test_year
    te = panel["filing_year"] == test_year
    y_tr, y_te = panel.loc[tr, "distress"], panel.loc[te, "distress"]

    Xb_tr = panel.loc[tr, base_cols].to_numpy()
    Xb_te = panel.loc[te, base_cols].to_numpy()

    add = (panel[add_cols].to_numpy() if add_override is None else add_override)
    Xa_tr = np.hstack([Xb_tr, add[tr.to_numpy()]])
    Xa_te = np.hstack([Xb_te, add[te.to_numpy()]])

    a_base = _fit_auc(Xb_tr, y_tr, Xb_te, y_te)
    a_aug = _fit_auc(Xa_tr, y_tr, Xa_te, y_te)
    return a_base, a_aug, a_aug - a_base


def permute_within_year(values: np.ndarray, years: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Shuffle the block across firms, independently within each year.

    Rows move together, so correlations WITHIN the block survive; only the
    link between the block and the firm is destroyed. Doing it within year
    also preserves the year mix, so a null draw differs from the real data
    in exactly one respect: which firm each text profile belongs to.
    """
    out = values.copy()
    for y in np.unique(years):
        idx = np.flatnonzero(years == y)
        out[idx] = values[rng.permutation(idx)]
    return out


def permutation_test(panel: pd.DataFrame, base_cols: list[str],
                     add_cols: list[str], test_year: int,
                     n_perm: int = 200, seed: int = 20260821,
                     block: np.ndarray | None = None) -> dict:
    """Observed delta plus its permutation null. Returns a verdict dict."""
    rng = np.random.default_rng(seed)
    values = panel[add_cols].to_numpy() if block is None else block
    years = panel["filing_year"].to_numpy()

    a_base, a_aug, observed = delta_auc(
        panel, base_cols, add_cols, test_year, add_override=values)

    # The baseline never changes across permutations — only the added block
    # is shuffled. Refitting it n_perm times doubles the work for an
    # identical number, so it is computed once above and reused.
    tr = (panel["filing_year"] < test_year).to_numpy()
    te = (panel["filing_year"] == test_year).to_numpy()
    y_tr, y_te = panel["distress"].to_numpy()[tr], panel["distress"].to_numpy()[te]
    Xb = panel[base_cols].to_numpy()

    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = permute_within_year(values, years, rng)
        aug = _fit_auc(np.hstack([Xb[tr], shuffled[tr]]), y_tr,
                       np.hstack([Xb[te], shuffled[te]]), y_te)
        null[i] = aug - a_base

    mu, sd = null.mean(), null.std(ddof=1)
    z = (observed - mu) / sd if sd > 0 else np.nan
    # One-sided: we only care about improvement.
    p = float((null >= observed).sum() + 1) / (n_perm + 1)
    return {
        "auc_base": a_base, "auc_augmented": a_aug, "delta": observed,
        "null_mean": mu, "null_sd": sd, "z": z, "p_value": p,
        "n_perm": n_perm, "null": null,
        "verdict": "SIGNAL" if p < 0.05 else "NO SIGNAL",
    }


def noise_control(panel: pd.DataFrame, base_cols: list[str], width: int,
                  test_year: int, n_draws: int = 50, n_perm: int = 100,
                  alpha: float = 0.05, seed: int = 20260821) -> dict:
    """
    The negative control, done properly: MANY noise draws, not one.

    Testing a single noise realisation at alpha = 0.05 promotes it about 5%
    of the time by construction. That is not a broken test, it is the
    definition of a 5% false-positive rate — but read as a pass/fail gate on
    one draw it is a coin flip that voids the notebook once in twenty runs.

    That is exactly what happened here: one draw came in at z = +2.24,
    p = 0.0100, and the gate declared the whole analysis void. The null mean
    was -0.0076, correctly showing that five junk columns normally COST
    accuracy; the draw simply landed in the upper tail.

    The right question is not "was this noise draw promoted?" but "is noise
    promoted at roughly the nominal rate?" A promotion rate near alpha means
    the test is calibrated. Materially above it means the test is broken and
    every verdict is void.
    """
    rng = np.random.default_rng(seed)
    n = len(panel)
    cols = [f"_noise{i}" for i in range(width)]

    p_values, deltas = [], []
    for _ in range(n_draws):
        block = rng.normal(size=(n, width))
        r = permutation_test(panel, base_cols, cols, test_year,
                             n_perm=n_perm, seed=int(rng.integers(1 << 31)),
                             block=block)
        p_values.append(r["p_value"])
        deltas.append(r["delta"])

    from scipy.stats import binomtest

    p_values = np.array(p_values)
    n_prom = int((p_values < alpha).sum())
    rate = n_prom / n_draws

    # Judge the promotion count with a BINOMIAL TEST, not an eyeballed
    # multiple of alpha.
    #
    # An earlier version tolerated "up to 3x nominal", which is an arbitrary
    # threshold dressed up as a criterion — and it duly passed a run showing
    # 3 of 20 (15%) promoted, purely because 0.15 <= 0.15. Under H0 that
    # outcome has probability 0.075: uncomfortable, not damning, and a
    # threshold that cannot tell the difference is not a test.
    #
    # The honest question is whether the observed count is surprising if the
    # true false-positive rate really is alpha. That is exactly a one-sided
    # binomial test, and it also makes the sample-size problem explicit:
    # at 20 draws the standard error on a 5% rate is ~4.9%, so 5% and 15%
    # are barely distinguishable. Hence the default is now 50 draws.
    bt = binomtest(n_prom, n_draws, alpha, alternative="greater")
    calib_p = float(bt.pvalue)
    calibrated = calib_p >= 0.05

    return {
        "n_draws": n_draws, "alpha": alpha,
        "promotion_rate": rate, "n_promoted": n_prom,
        "expected_promoted": alpha * n_draws,
        "calibration_p": calib_p,
        "mean_delta": float(np.mean(deltas)),
        "p_values": p_values,
        "calibrated": calibrated,
        "verdict": ("CALIBRATED" if calibrated
                    else "BROKEN — noise promoted more often than alpha"),
    }


def report_noise(r: dict) -> None:
    print(f"\nNEGATIVE CONTROL — pure noise, {r['n_draws']} independent draws")
    print(f"    promoted at alpha={r['alpha']}   "
          f"{r['n_promoted']} of {r['n_draws']}  ({r['promotion_rate']:.1%})")
    print(f"    expected under H0          "
          f"{r['expected_promoted']:.1f}  ({r['alpha']:.0%})")
    print(f"    binomial p (one-sided)     {r['calibration_p']:.4f}"
          "   <- is that count surprising if the rate really is alpha?")
    print(f"    mean delta across draws    {r['mean_delta']:+.4f}"
          "   (should be <= 0: junk columns cost accuracy)")
    print(f"    -> {r['verdict']}")


def report(name: str, r: dict) -> None:
    print(f"\n{name}")
    print(f"    AUC ratios only        {r['auc_base']:.4f}")
    print(f"    AUC ratios + block     {r['auc_augmented']:.4f}")
    print(f"    delta                  {r['delta']:+.4f}")
    print(f"    null mean +/- sd       {r['null_mean']:+.4f} +/- {r['null_sd']:.4f}"
          f"   ({r['n_perm']} permutations)")
    print(f"    z                      {r['z']:+.2f}")
    print(f"    p (one-sided)          {r['p_value']:.4f}")
    print(f"    -> {r['verdict']}")
