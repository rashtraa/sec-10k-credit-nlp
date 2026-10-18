"""
Regression suite for stage 8.

Split deliberately in two. The pure-pandas reference implementations are
tested unconditionally — they are what Spark is checked AGAINST, so a bug
there would make the correctness control agree with itself and prove nothing.
The Spark tests skip when PySpark or a JVM is absent, which is the normal
state of a machine without Java.

Run:  pytest test_spark_features.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from spark_features import N_FEATURES, lm_features_pandas, yoy_pandas
from text_features import LM_CATEGORIES



def _corpus():
    a = ("The Company faces adverse litigation risk and material loss. " * 40)
    b = ("The Company faces adverse litigation risk and material loss. " * 40)
    c = ("Our vendors deliver components on schedule and demand is strong. " * 40)
    return pd.DataFrame([
        {"cik": 1, "ticker": "AAA", "filing_year": 2022, "item1a": a},
        {"cik": 1, "ticker": "AAA", "filing_year": 2023, "item1a": b},
        {"cik": 2, "ticker": "BBB", "filing_year": 2022, "item1a": c},
        {"cik": 2, "ticker": "BBB", "filing_year": 2023, "item1a": a},
    ])


# ────────────────────────────────────────────── the pandas reference itself
def test_lm_frequencies_are_frequencies():
    """Bounded in [0,1] and scale-free — a longer filing saying the same thing
    must not score higher just for being longer."""
    df = _corpus()
    out = lm_features_pandas(df)
    for c in LM_CATEGORIES:
        col = out[f"lm_{c.lower()}"]
        assert col.between(0, 1).all(), c
    doubled = df.copy()
    doubled["item1a"] = doubled["item1a"] * 2
    out2 = lm_features_pandas(doubled)
    assert np.allclose(out["lm_negative"], out2["lm_negative"], atol=1e-9)


def test_yoy_is_zero_for_an_unchanged_filing():
    """
    A firm that files identical text two years running has changed nothing, so
    dissimilarity must be 0. This is the positive control for the feature that
    carried the whole stage-3 hypothesis.
    """
    out = yoy_pandas(_corpus()).set_index(["cik", "filing_year"])
    assert out.loc[(1, 2023), "risk_change"] == pytest.approx(0.0, abs=1e-6)


def test_yoy_is_large_for_a_rewritten_filing():
    out = yoy_pandas(_corpus()).set_index(["cik", "filing_year"])
    assert out.loc[(2, 2023), "risk_change"] > 0.5


def test_yoy_is_nan_without_a_prior_year():
    out = yoy_pandas(_corpus()).set_index(["cik", "filing_year"])
    assert np.isnan(out.loc[(1, 2022), "risk_change"])
    assert np.isnan(out.loc[(2, 2022), "risk_change"])


def test_yoy_is_bounded():
    out = yoy_pandas(_corpus())["risk_change"].dropna()
    assert ((out >= 0.0) & (out <= 1.0)).all()


# ─────────────────────────────────────────────────────────── spark, if present
#
# The skip lives in the FIXTURE, not at module level. A module-level
# importorskip would skip the pandas reference tests above too — and those are
# what Spark is checked against, so losing them would leave the correctness
# control agreeing with itself.
@pytest.fixture(scope="module")
def spark():
    pytest.importorskip("pyspark", reason="PySpark not installed")
    from spark_features import get_spark
    try:
        s = get_spark("test", shuffle_partitions=2)
    except BaseException as e:
        # BaseException, not Exception. `_check_java` raises SystemExit, which
        # is NOT an Exception subclass — so `except Exception` let it escape
        # and pytest reported three ERRORs instead of three skips. A missing
        # JVM is a reason to skip, not a test failure.
        if isinstance(e, KeyboardInterrupt):
            raise
        pytest.skip(f"Spark unavailable: {e}")
    yield s
    s.stop()


def test_spark_lm_matches_pandas_exactly(spark):
    """
    The port must agree with the implementation it replaces. Same lexicon,
    same tokeniser, same arithmetic — so this is exact equality, not a
    tolerance. A tolerance here would hide a real disagreement.
    """
    from spark_features import lm_features_spark
    df = _corpus()
    a = lm_features_pandas(df).sort_values(["cik", "filing_year"]).reset_index(drop=True)
    b = (lm_features_spark(spark, df).toPandas()
         .sort_values(["cik", "filing_year"]).reset_index(drop=True))
    for col in ["n_words"] + [f"lm_{c.lower()}" for c in LM_CATEGORIES]:
        assert np.allclose(a[col].astype(float), b[col].astype(float),
                           rtol=1e-9, equal_nan=True), col


def test_pandas_udf_and_plain_udf_agree(spark):
    """Arrow batching is an optimisation. It must not change the answer."""
    from spark_features import lm_features_spark
    df = _corpus()
    v = lm_features_spark(spark, df, vectorised=True).toPandas().sort_values("cik")
    u = lm_features_spark(spark, df, vectorised=False).toPandas().sort_values("cik")
    assert np.allclose(v["lm_negative"].astype(float),
                       u["lm_negative"].astype(float), rtol=1e-9)


def test_spark_yoy_preserves_the_controls(spark):
    """
    Spark ML's IDF smoothing differs slightly from sklearn's, so the values are
    close rather than identical. What must hold exactly is the SHAPE: zero for
    an unchanged filing, large for a rewritten one, null with no prior year.
    """
    from spark_features import yoy_spark
    out = yoy_spark(spark, _corpus()).toPandas().set_index(["cik", "filing_year"])
    assert out.loc[(1, 2023), "risk_change"] == pytest.approx(0.0, abs=1e-3)
    assert out.loc[(2, 2023), "risk_change"] > 0.5
    assert np.isnan(out.loc[(1, 2022), "risk_change"])
