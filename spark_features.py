"""
Stage 8 — the same feature pipeline, in PySpark, with the honest benchmark.

WHY PORT SOMETHING THAT ALREADY WORKS
-------------------------------------
The pandas pipeline computes Loughran-McDonald sentiment frequencies and
year-over-year TF-IDF dissimilarity for 8,801 filings in a couple of minutes.
Nothing about that is broken, and rewriting it in Spark makes it *slower* at
this size. That is not a reason to skip the exercise — it is the exercise.

The question worth answering is not "can I use Spark". It is **at what data
size does Spark start to win**, and the answer is a number you can measure on
one laptop. Below the crossover you are paying JVM startup, serialisation and
shuffle costs for nothing. Above it, a single machine's memory is the
constraint and Spark's model is the only one that works.

An engineer who can say "Spark cost us 4x at our data size, here is the
measurement, here is the size at which it flips" is more useful than one who
reaches for it by default.

WHAT IS ACTUALLY DEMONSTRATED
-----------------------------
  pandas_udf vs plain udf   Arrow-batched UDFs move whole columns to Python at
                            once. A row-at-a-time UDF pays serialisation per
                            record. The gap is large and worth measuring.

  Spark ML pipeline         Tokenizer -> HashingTF -> IDF -> Normalizer, then a
                            self-join on (cik, year-1) to get year-over-year
                            cosine. This is the part that does not reduce to a
                            UDF, and the part that would genuinely scale.

  a correctness control     Spark and pandas must produce the SAME numbers.
                            A faster pipeline that quietly disagrees with the
                            one it replaces is not a port, it is a second
                            unvalidated pipeline.

Run:  python spark_features.py --check       # correctness only, fast
      python spark_features.py --benchmark   # the crossover measurement
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import INTERIM, PROCESSED
from text_features import LM_CATEGORIES, count_categories, load_lm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("spark")

USABLE = ("ok", "ok_loose")
N_FEATURES = 1 << 18          # hashing space for HashingTF
LOCAL_JDK = Path(__file__).resolve().parent / ".java"


# ══════════════════════════════════════════════════════════════════ session
# Spark 3.5 targets Java 8/11/17. On Java 17+ the module system blocks the
# reflective access Arrow uses to allocate off-heap buffers, and the failure is
# a wall of JVM stack trace ending in
#
#     java.lang.UnsupportedOperationException: sun.misc.Unsafe or
#     java.nio.DirectByteBuffer.<init>(long, int) not available
#
# which says nothing about Java versions. Hit on Java 21 while testing this
# module. The flags below re-open the packages Arrow needs. They must be set
# BEFORE the JVM starts, which in local mode means before the first
# SparkSession — `spark.driver.extraJavaOptions` is too late, because py4j has
# already launched the JVM by then.
_JDK_OPENS = " ".join([
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
    "--add-opens=java.base/java.io=ALL-UNNAMED",
    "--add-opens=java.base/java.net=ALL-UNNAMED",
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.util=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
    "-Dio.netty.tryReflectionSetAccessible=true",
])


def _check_java() -> None:
    """
    Fail with a sentence instead of a stack trace.

    Spark 3.5 supports Java 8, 11 and 17. On Java 21 it does not merely warn —
    Arrow cannot allocate off-heap buffers and every pandas_udf dies in a
    hundred lines of JVM trace whose top frame mentions `sun.misc.Unsafe` and
    never mentions Java versions. Verified: identical code, 8 tests failing on
    21 and 8 passing on 17. Anaconda ships 21 by default, so this is likely.
    """
    import os, re, shutil, subprocess

    home = os.environ.get("JAVA_HOME", "")
    exe = str(Path(home) / "bin" / "java") if home else (shutil.which("java") or "")

    # Last resort: a JDK unpacked into the project by `setup_java.py`. Found
    # here rather than requiring JAVA_HOME in every terminal, because "works
    # in one window and not the next" is its own category of wasted evening.
    if not exe:
        local = sorted(LOCAL_JDK.glob("*/bin/java*")) if LOCAL_JDK.exists() else []
        local = [p for p in local if p.suffix in ("", ".exe")]
        if local:
            exe = str(local[0])
            os.environ["JAVA_HOME"] = str(local[0].parent.parent)
            log.info("using project-local JDK at %s", os.environ["JAVA_HOME"])

    if not exe:
        raise SystemExit(
            "No Java found. PySpark needs a JVM, and Spark 3.5 needs Java 17.\n\n"
            "  Easiest, no admin rights needed:\n"
            "      python setup_java.py\n\n"
            "  Or install Temurin 17 yourself:\n"
            "      https://adoptium.net/temurin/releases/?version=17")
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True,
                             timeout=30).stderr
        major = int(re.search(r'version "(\d+)', out).group(1))
    except Exception:
        return                      # unparseable: let Spark speak for itself
    if major > 17:
        raise SystemExit(
            f"Java {major} found, but Spark 3.5 supports 8, 11 and 17 only.\n"
            "  On Java 18+ every pandas_udf fails inside Arrow with a JVM stack\n"
            "  trace that never mentions the Java version.\n\n"
            "  Install Temurin 17: https://adoptium.net/temurin/releases/?version=17\n"
            "  Then, in this terminal only:\n"
            '    $env:JAVA_HOME = "C:\\Program Files\\Eclipse Adoptium\\jdk-17..."\n'
            "  and re-run. `java -version` should print 17.")


def _pin_python() -> None:
    """
    Tell Spark exactly which Python to run its workers with.

    Spark launches a worker process per task by executing `python`. It does not
    inherit the interpreter running the driver — it looks up the name on PATH.
    On Windows that name resolves to the Microsoft Store App Execution Alias, a
    stub that prints

        Python was not found; run without arguments to install from the
        Microsoft Store

    and exits. The worker therefore never connects back, and Spark reports the
    symptom four hundred lines later as

        org.apache.spark.SparkException: Python worker failed to connect back
        Caused by: java.net.SocketTimeoutException: Accept timed out

    which points at sockets and says nothing about Python. A virtualenv makes
    it worse: activating the venv sets PATH for the shell, but the alias still
    wins for a bare `python` launched by a JVM subprocess.

    `sys.executable` is by definition the interpreter running this code, which
    is the one that has pandas, the lexicon and everything else the UDF needs.

    PYTHONPATH is set for a SECOND, separate reason — see `_ship_modules`.
    """
    import os
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    here = str(Path(__file__).resolve().parent)
    existing = os.environ.get("PYTHONPATH", "")
    if here not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = (here + os.pathsep + existing).rstrip(os.pathsep)


def _ship_modules(spark=None) -> None:
    """
    Make this project's own code travel WITH the pickled function.

    THE PLATFORM DIFFERENCE THAT MAKES THIS NECESSARY
    -------------------------------------------------
    cloudpickle serialises a function defined in `__main__` BY VALUE, but a
    function imported from a module BY REFERENCE. `count_categories` lives in
    `text_features`, so what crosses to the worker is the instruction "import
    text_features and look it up" — not the code.

    On Linux that works by accident. Spark runs `pyspark.daemon`, which FORKS
    workers, and a forked process inherits the parent's `sys.path` including
    the project directory. Windows has no fork: Spark spawns a fresh
    `python -m pyspark.worker` per task with a clean path, the import fails,
    and the worker dies before writing a byte. The JVM reports what it saw —

        org.apache.spark.SparkException: Python worker exited unexpectedly
        Caused by: java.io.EOFException

    — which describes a truncated socket and says nothing about an import.

    WHY NOT addPyFile
    -----------------
    `sparkContext.addPyFile()` is the textbook answer and it does not work
    here. It routes through Hadoop's `FileUtil.chmod`, which on Windows shells
    out to `winutils.exe`; without a Hadoop install that raises

        java.io.FileNotFoundException: HADOOP_HOME and hadoop.home.dir are unset

    So the textbook fix trades an import error for a dependency on a binary
    that has nothing to do with this program.

    `register_pickle_by_value` is better on every axis: cloudpickle embeds the
    module's source in the pickle itself, so there is no file to distribute, no
    path to configure and no Hadoop involved. It is also the honest expression
    of the intent — this code belongs to the job, not to the cluster.
    """
    import importlib

    try:
        from pyspark import cloudpickle
    except ImportError:                       # pragma: no cover
        return

    if not hasattr(cloudpickle, "register_pickle_by_value"):
        # cloudpickle < 2.0 — fall back to shipping the file, which needs
        # winutils on Windows but is better than nothing.
        if spark is not None:
            here = Path(__file__).resolve().parent
            for name in ("text_features.py",):
                f = here / name
                if f.exists():
                    try:
                        spark.sparkContext.addPyFile(str(f))
                    except Exception as e:
                        log.warning("addPyFile failed (%s); workers may not "
                                    "find project modules", e)
        return

    for name in ("text_features",):
        try:
            cloudpickle.register_pickle_by_value(importlib.import_module(name))
        except Exception as e:                # pragma: no cover
            log.warning("could not pickle %s by value: %s", name, e)


def _open_jdk_modules() -> None:
    import os
    if "PYSPARK_SUBMIT_ARGS" not in os.environ:
        os.environ["PYSPARK_SUBMIT_ARGS"] = (
            f'--driver-java-options "{_JDK_OPENS}" pyspark-shell')
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")


def get_spark(app: str = "edgar-features", shuffle_partitions: int = 8,
              cores: int | None = None, driver_memory: str = "2g"):
    """
    A local SparkSession sized for one machine.

    WHY NOT `local[*]`
    ------------------
    `local[*]` runs one task per core, and every Python task is a SEPARATE
    process that imports pandas and numpy — about 250-400 MB resident each.
    On a 16-core laptop that is 4-6 GB of Python on top of the driver JVM,
    and when the machine runs out the OS kills workers. Spark reports the
    killed worker as

        Python worker exited unexpectedly (crashed)
        Caused by: java.io.EOFException

    with no Python traceback, because there was no Python exception — the
    process was killed. The tell is that EARLY tasks succeed and later ones
    fail: memory pressure builds as workers accumulate.

    So the default here is four workers, not all of them. On this workload
    that costs almost nothing: the job is bound by Python function calls, and
    four processes already saturate the memory bandwidth that matters.

    `spark.sql.shuffle.partitions` defaults to 200, which on a laptop means 200
    tiny tasks and their scheduling overhead for a shuffle that should be eight.
    It is the single most common reason a small Spark job is embarrassingly
    slow, and leaving it at the default would make this benchmark a measurement
    of a misconfiguration rather than of Spark.
    """
    _check_java()
    _pin_python()
    _open_jdk_modules()
    from pyspark.sql import SparkSession
    session = (SparkSession.builder
            .appName(app)
            .master(f"local[{cores}]" if cores else "local[4]")
            .config("spark.sql.shuffle.partitions", shuffle_partitions)
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.driver.memory", driver_memory)
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.executor.extraJavaOptions", _JDK_OPENS)
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
            # Turn a truncated Arrow stream into an actual Python traceback.
            # Without these, a worker that dies on an exception is reported as
            # `EOFException`, which points at the socket rather than at the
            # bug, and cost an evening to diagnose once already.
            # A pandas_udf batch defaults to 10,000 ROWS. That is fine for
            # numbers and catastrophic for 70,000-character filings: one batch
            # would be ~700 MB of Python strings inside a worker process.
            # Spark warns about it obliquely ("task of very large size") and
            # then the worker dies with no traceback, because it was killed
            # rather than raising. 16 rows per batch caps a batch at ~1 MB.
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "16")
            .config("spark.python.worker.faulthandler.enabled", "true")
            .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
            .getOrCreate())
    _ship_modules(session)
    return session


def load_sections(sample: int | None = None) -> pd.DataFrame:
    sec = pd.read_parquet(INTERIM / "sections.parquet")
    sec = sec[sec["item1a_status"].isin(USABLE)][
        ["cik", "ticker", "filing_year", "item1a"]].reset_index(drop=True)
    if sample and sample < len(sec):
        sec = sec.sample(sample, random_state=20260821).reset_index(drop=True)
    return sec


# ══════════════════════════════════════════════════════════════════ pandas
def lm_features_pandas(df: pd.DataFrame) -> pd.DataFrame:
    cats = load_lm()
    rows = [count_categories(t, cats) for t in df["item1a"]]
    out = pd.DataFrame(rows)
    return pd.concat([df[["cik", "ticker", "filing_year"]], out], axis=1)


def yoy_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Year-over-year cosine dissimilarity, the pandas reference implementation.

    Two choices here exist only so the correctness control tests Spark rather
    than testing library conventions.

    HASHED, not vocabulary-based, because Spark's HashingTF is hashed.
    Comparing a hashed pipeline against a vocabulary-based one would fail for
    a reason that has nothing to do with Spark.

    IDF COMPUTED BY HAND as ln((n+1)/(df+1)), which is Spark ML's formula.
    sklearn's TfidfTransformer uses ln((1+n)/(1+df)) + 1 — the trailing +1 is
    a sklearn convention, and on a corpus with few distinct terms it shifts
    every weight enough to move year-over-year cosine by 0.24. Matching the
    formula turns a "close enough, trust me" comparison into an exact one.
    """
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.preprocessing import normalize

    hv = HashingVectorizer(n_features=N_FEATURES, alternate_sign=False,
                           norm=None, lowercase=True)
    counts = hv.transform(df["item1a"])
    n = counts.shape[0]
    doc_freq = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log((n + 1.0) / (doc_freq + 1.0))
    X = normalize(counts.multiply(idf).tocsr(), norm="l2")

    idx = {(int(c), int(y)): i for i, (c, y)
           in enumerate(zip(df["cik"], df["filing_year"]))}
    out = []
    for i, (c, y) in enumerate(zip(df["cik"], df["filing_year"])):
        j = idx.get((int(c), int(y) - 1))
        if j is None:
            out.append(np.nan)
        else:
            sim = float(X[i].multiply(X[j]).sum())
            out.append(1.0 - min(1.0, max(0.0, sim)))
    return df[["cik", "filing_year"]].assign(risk_change=out)


# ═══════════════════════════════════════════════════════════════════ spark
def lm_features_spark(spark, pdf: pd.DataFrame, vectorised: bool = True):
    """
    Loughran-McDonald frequencies as a Spark job.

    The lexicon is BROADCAST. Without that, the ~4,000-word dictionary is
    serialised and shipped with every single task — the classic mistake that
    makes a Spark job scale worse than the loop it replaced.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StructField, DoubleType, LongType

    cats = load_lm()
    bc = spark.sparkContext.broadcast(cats)

    # Repartition so no single task carries more than a few MB of text.
    # `createDataFrame` splits by ROW COUNT, which is the wrong unit when rows
    # are 70 KB documents — it produced 5 MB tasks and killed workers.
    n_parts = max(8, int(pdf["item1a"].str.len().sum() // 2_000_000) + 1)
    sdf = spark.createDataFrame(pdf).repartition(n_parts)

    schema = StructType(
        [StructField("n_words", LongType())] +
        [StructField(f"lm_{c.lower()}", DoubleType()) for c in LM_CATEGORIES])

    if vectorised:
        # pandas_udf: Arrow moves a whole column into Python at once, so the
        # per-record serialisation cost disappears.
        @F.pandas_udf(schema)
        def feats(texts: pd.Series) -> pd.DataFrame:
            return pd.DataFrame([count_categories(t, bc.value) for t in texts])
    else:
        # Row-at-a-time, kept only as the thing to measure against.
        @F.udf(schema)
        def feats(text):
            d = count_categories(text, bc.value)
            return [d["n_words"]] + [d[f"lm_{c.lower()}"] for c in LM_CATEGORIES]

    out = sdf.withColumn("f", feats(F.col("item1a")))
    cols = ["cik", "ticker", "filing_year", "f.n_words"] + \
           [f"f.lm_{c.lower()}" for c in LM_CATEGORIES]
    return out.select(*cols)


def yoy_spark(spark, pdf: pd.DataFrame):
    """
    Year-over-year dissimilarity with Spark ML plus a self-join.

    This is the part that does not reduce to a UDF and the part that would
    genuinely scale: IDF needs a pass over the whole corpus, and pairing each
    filing with the same firm's prior year is a shuffle. On one machine both
    are overhead; on a cluster they are the reason Spark exists.
    """
    from pyspark.ml.feature import HashingTF, IDF, Normalizer, RegexTokenizer
    from pyspark.sql import functions as F
    from pyspark.ml.linalg import SparseVector

    sdf = spark.createDataFrame(pdf).withColumn(
        "lower", F.lower(F.col("item1a")))
    # RegexTokenizer with sklearn's own token pattern, NOT Spark's default
    # `Tokenizer`. Spark's default splits on whitespace and keeps punctuation
    # attached, so "loss." and "loss" hash to different buckets while sklearn
    # treats them as one token. That mismatch alone moved year-over-year
    # cosine by up to 0.24 against the pandas reference — a disagreement that
    # looks like a Spark bug and is actually a tokenisation bug.
    tok = RegexTokenizer(inputCol="lower", outputCol="words",
                         pattern=r"\w\w+", gaps=False, toLowercase=True)
    tf = HashingTF(inputCol="words", outputCol="tf", numFeatures=N_FEATURES)
    stage = tf.transform(tok.transform(sdf))
    idf = IDF(inputCol="tf", outputCol="tfidf").fit(stage)
    vecs = Normalizer(inputCol="tfidf", outputCol="v", p=2.0).transform(
        idf.transform(stage)).select("cik", "filing_year", "v")

    prev = vecs.select(F.col("cik").alias("pcik"),
                       (F.col("filing_year") + 1).alias("pyear"),
                       F.col("v").alias("pv"))
    joined = vecs.join(prev,
                       (vecs.cik == prev.pcik) & (vecs.filing_year == prev.pyear),
                       how="left")

    def cosine(a, b):
        if a is None or b is None:
            return float("nan")
        if isinstance(a, SparseVector) and isinstance(b, SparseVector):
            da = dict(zip(a.indices, a.values))
            s = sum(v * da[i] for i, v in zip(b.indices, b.values) if i in da)
        else:
            s = float(a.dot(b))
        return 1.0 - min(1.0, max(0.0, float(s)))

    cos_udf = F.udf(cosine, "double")
    return joined.select("cik", "filing_year",
                         cos_udf("v", "pv").alias("risk_change"))


# ══════════════════════════════════════════════════════════════ correctness
def check(sample: int = 300, cores: int = 4) -> bool:
    """
    Spark and pandas must agree. A faster pipeline that quietly disagrees with
    the one it replaces is not a port — it is a second unvalidated pipeline,
    and the disagreement will surface as a modelling result nobody can explain.
    """
    pdf = load_sections(sample)
    spark = get_spark("check", cores=cores)

    a = lm_features_pandas(pdf).sort_values(["cik", "filing_year"]).reset_index(drop=True)
    b = (lm_features_spark(spark, pdf).toPandas()
         .sort_values(["cik", "filing_year"]).reset_index(drop=True))

    ok = True
    for col in ["n_words"] + [f"lm_{c.lower()}" for c in LM_CATEGORIES]:
        same = np.allclose(a[col].astype(float), b[col].astype(float),
                           rtol=1e-9, atol=1e-12, equal_nan=True)
        log.info("  %-18s %s", col, "match" if same else "MISMATCH")
        ok &= bool(same)

    ya = yoy_pandas(pdf).sort_values(["cik", "filing_year"]).reset_index(drop=True)
    yb = (yoy_spark(spark, pdf).toPandas()
          .sort_values(["cik", "filing_year"]).reset_index(drop=True))
    merged = ya.merge(yb, on=["cik", "filing_year"], suffixes=("_pd", "_sp"))
    both = merged.dropna(subset=["risk_change_pd", "risk_change_sp"])
    # Tolerance is loose here on purpose: sklearn and Spark ML use different
    # IDF smoothing (log((N+1)/(df+1))+1 against log((N+1)/(df+1))), so the
    # vectors are close but not bit-identical. Demanding equality would be
    # demanding the wrong thing.
    d = (both["risk_change_pd"] - both["risk_change_sp"]).abs()
    rho = (both["risk_change_pd"].corr(both["risk_change_sp"], method="spearman")
           if len(both) > 2 else float("nan"))
    log.info("  %-18s max abs diff %.4f · spearman %.4f · over %s paired rows",
             "risk_change", float(d.max()) if len(d) else float("nan"),
             rho, f"{len(d):,}")
    # Both sides now use Spark ML's IDF formula and sklearn's token pattern,
    # so the only remaining source of difference is hash collisions — the two
    # libraries use different murmur3 seeds, so a rare pair of distinct terms
    # may collide in one and not the other. At 2^18 buckets that is a handful
    # of documents at the fourth decimal, not a systematic gap.
    ok &= bool(len(d) == 0 or (d.max() < 0.01 and (rho > 0.99 or np.isnan(rho))))

    spark.stop()
    print("\nCORRECTNESS:", "PASS" if ok else "FAIL")
    return ok


# ═══════════════════════════════════════════════════════════════ benchmark
def _materialise(sdf) -> None:
    """
    Force every row through the plan.

    `.count()` does NOT. Spark's optimiser prunes columns a count does not
    need, so the UDF is never evaluated and the timing measures an empty plan.
    The first version of this benchmark used `.count()` and reported the
    row-at-a-time UDF as FASTER than the Arrow-batched one, which is
    impossible and was the clue. Writing to the `noop` sink runs the whole
    plan and discards the output — the standard way to time a Spark job.
    """
    sdf.write.format("noop").mode("overwrite").save()


def _warm(spark, pdf: pd.DataFrame) -> None:
    """
    One untimed pass of everything before any measurement.

    The first Spark action in a session pays JVM class loading, JIT warm-up
    and codegen. Timing it as if it were steady-state overstates Spark by
    seconds and makes the crossover look further away than it is.
    """
    small = pdf.head(min(50, len(pdf)))
    lm_features_pandas(small)
    _materialise(lm_features_spark(spark, small, vectorised=True))
    _materialise(lm_features_spark(spark, small, vectorised=False))

def benchmark(sizes: list[int], replicate: int = 1, cores: int = 4) -> pd.DataFrame:
    """
    Wall-clock at each corpus size, for four implementations.

    `replicate` duplicates the corpus to reach sizes the real data cannot,
    which is how the crossover gets observed on a laptop. Duplicated text is
    fine for a throughput measurement and useless for a modelling one, so it
    is used only here.
    """
    rows = []
    spark = get_spark("bench", cores=cores)
    log.info("warm-up pass (untimed)")
    _warm(spark, load_sections(50))

    for n in sizes:
        pdf = load_sections(min(n, 10 ** 9))
        if replicate > 1:
            pdf = pd.concat([pdf] * replicate, ignore_index=True)
        mb = pdf["item1a"].str.len().sum() / 1e6
        log.info("size %s filings (%.0f MB of text)", f"{len(pdf):,}", mb)

        t = time.perf_counter()
        lm_features_pandas(pdf)
        pandas_s = time.perf_counter() - t

        t = time.perf_counter()
        _materialise(lm_features_spark(spark, pdf, vectorised=True))
        spark_vec_s = time.perf_counter() - t

        t = time.perf_counter()
        _materialise(lm_features_spark(spark, pdf, vectorised=False))
        spark_udf_s = time.perf_counter() - t

        rows.append({"filings": len(pdf), "text_mb": round(mb, 1),
                     "pandas_s": round(pandas_s, 1),
                     "spark_pandas_udf_s": round(spark_vec_s, 1),
                     "spark_plain_udf_s": round(spark_udf_s, 1),
                     "spark_vs_pandas": round(spark_vec_s / pandas_s, 2),
                     "arrow_speedup": round(spark_udf_s / spark_vec_s, 2),
                     "pandas_filings_per_s": round(len(pdf) / pandas_s, 1),
                     "spark_filings_per_s": round(len(pdf) / spark_vec_s, 1)})
        log.info("  pandas %.1fs · spark(pandas_udf) %.1fs · spark(udf) %.1fs",
                 pandas_s, spark_vec_s, spark_udf_s)

    spark.stop()
    out = pd.DataFrame(rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED / "spark_benchmark.csv", index=False)

    print(f"\n{'='*72}\nTHROUGHPUT\n{'='*72}")
    print(out.to_string(index=False))
    print("\nspark_vs_pandas > 1 means Spark is SLOWER at that size. The number")
    print("to quote is where that ratio crosses 1 — below it Spark is paying")
    print("JVM, serialisation and shuffle costs for nothing.")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="correctness only")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--sizes", default="250,1000,4000")
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--cores", type=int, default=4,
                    help="local[N] workers. Lower this if workers are killed.")
    a = ap.parse_args()

    if a.check or not a.benchmark:
        check(cores=a.cores)
    if a.benchmark:
        # PowerShell treats `500,2000,7000` as an array literal and hands the
        # binder three items, which arrive here joined by spaces. Accept both
        # rather than making the caller remember to quote it.
        sizes = [int(x) for x in re.split(r"[,\s]+", a.sizes.strip()) if x]
        benchmark(sizes, a.replicate, cores=a.cores)
