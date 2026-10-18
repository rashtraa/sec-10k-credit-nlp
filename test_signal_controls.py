"""
Validate the permutation machinery on SYNTHETIC data where the answer is known.

Three scenarios must produce three different verdicts. If they don't, the
test cannot distinguish signal from noise and every real verdict is void.
"""
import sys; sys.path.insert(0, '/home/claude/edgar-credit-signal')
import numpy as np, pandas as pd
from signal_test import (delta_auc, permutation_test, report,
                         permute_within_year)

rng = np.random.default_rng(7)
N_FIRMS, YEARS = 900, [2019, 2020, 2021, 2022, 2023]
rows = []
for cik in range(N_FIRMS):
    lev = rng.beta(2, 3)
    for y in YEARS:
        rows.append(dict(cik=cik, filing_year=y, leverage=lev + rng.normal(0, .03),
                         log_assets=rng.normal(20, 2), roa=rng.normal(.05, .08),
                         cash_ratio=rng.beta(2, 8), coverage=rng.lognormal(1.5, 1),
                         rev_growth=rng.normal(.05, .2)))
d = pd.DataFrame(rows)

# TRUE risk: driven by leverage and roa only.
lin = 3.0*d.leverage - 8.0*d.roa - 1.5
d["distress"] = rng.binomial(1, 1/(1+np.exp(-lin))).astype(float)

BASE = ["log_assets","roa","cash_ratio","coverage","rev_growth"]   # leverage held out
TEXT = ["risk_change","lm_negative","lm_uncertainty","lm_litigious","lm_constraining"]

print(f"panel: {len(d):,} firm-years, distress rate {d.distress.mean():.1%}")

# 1 · NEGATIVE CONTROL — pure noise. Must FAIL.
noise = rng.normal(size=(len(d), len(TEXT)))
for i,c in enumerate(TEXT): d[c] = noise[:,i]
r_noise = permutation_test(d, BASE, TEXT, 2023, n_perm=150)
report("NEGATIVE CONTROL (pure noise) — must be NO SIGNAL", r_noise)

# 2 · POSITIVE CONTROL — leverage, which truly drives the label. Must PASS.
lev_block = d[["leverage"]].to_numpy()
r_lev = permutation_test(d, BASE, ["leverage"], 2023, n_perm=150, block=lev_block)
report("POSITIVE CONTROL (leverage) — must be SIGNAL", r_lev)

# 3 · MEAN-SHIFT TRAP — a constant offset carries no member-specific info.
#     This is the AmEx NB09 failure mode; it must NOT be promoted.
shift = np.tile(np.array([5.0]*len(TEXT)), (len(d),1)) + rng.normal(0,1e-6,(len(d),len(TEXT)))
r_shift = permutation_test(d, BASE, TEXT, 2023, n_perm=150, block=shift)
report("MEAN-SHIFT TRAP (constant block) — must be NO SIGNAL", r_shift)

ok = (r_noise["verdict"]=="NO SIGNAL" and r_lev["verdict"]=="SIGNAL"
      and r_shift["verdict"]=="NO SIGNAL")
print("\n" + ("CONTROLS BEHAVE — the test can tell signal from noise"
              if ok else "CONTROLS FAILED — verdicts would be meaningless"))
sys.exit(0 if ok else 1)
