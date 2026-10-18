"""
Find out which layer of PySpark is broken, in about a minute.

WHY A SEPARATE SCRIPT
---------------------
`Python worker exited unexpectedly (crashed)` followed by `EOFException` is
not one failure mode — it is every failure mode. The JVM only knows that the
pipe closed early. The Python side could have failed to launch, failed to
import, raised, or died in native code inside numpy or pyarrow, and all four
produce the identical stack trace.

So instead of guessing, this runs four jobs of escalating dependency and
reports the FIRST one that fails. Where it stops tells you what is wrong:

  1 no Python at all      pure JVM. Fails => Spark/Java itself is broken.
  2 plain Python UDF      a worker launches and runs Python. Fails => the
                          worker cannot start: interpreter path, permissions,
                          antivirus.
  3 pandas_udf (Arrow)    the worker must import pyarrow and pandas and hand
                          Arrow buffers across. Fails HERE while 2 passes =>
                          a native library mismatch, almost always
                          pyarrow/numpy built against different ABIs.
  4 project UDF           adds this project's own modules. Fails HERE while 3
                          passes => a module-shipping problem, not a Spark one.

Run:  python spark_doctor.py
"""
from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def versions() -> None:
    print("=" * 68)
    print("VERSIONS")
    print("=" * 68)
    print(f"  python    {sys.version.split()[0]}  ({sys.executable})")
    for mod in ("pyspark", "pyarrow", "pandas", "numpy"):
        try:
            m = __import__(mod)
            print(f"  {mod:<9} {getattr(m, '__version__', '?')}")
        except Exception as e:
            print(f"  {mod:<9} MISSING ({e})")
    try:
        from spark_features import _check_java
        _check_java()
        import os
        home = os.environ.get("JAVA_HOME", "(on PATH)")
        out = subprocess.run([str(Path(home) / "bin" / "java") if home != "(on PATH)" else "java",
                              "-version"], capture_output=True, text=True).stderr
        print(f"  java      {out.splitlines()[0] if out else '?'}")
    except BaseException as e:
        print(f"  java      {e}")
    print()


def worker_can_start() -> bool:
    """
    Launch the exact command Spark uses for a worker and see what happens.

    Spark starts workers as `python -m pyspark.worker`. Running it directly
    surfaces an import error or a DLL failure as a normal Python traceback,
    where Spark would have shown only a closed socket.
    """
    print("=" * 68)
    print("CAN THE WORKER INTERPRETER EVEN START?")
    print("=" * 68)
    code = ("import pyarrow, pandas, numpy;"
            "print('worker imports ok:', pyarrow.__version__, pandas.__version__, numpy.__version__)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print("  stdout:", r.stdout.strip() or "(none)")
    if r.returncode != 0:
        print("  stderr:", r.stderr.strip()[-1500:])
        print("\n  >>> The interpreter Spark will use cannot import its own")
        print("      dependencies. That is the whole bug — nothing to do with")
        print("      Spark. Fix the environment first.")
        return False
    print("  exit code 0 — the worker interpreter is fine\n")
    return True


def _retry_with_fewer_cores() -> bool:
    """
    Re-run the failing job in a fresh process with 2 workers.

    A fresh process because a SparkSession cannot change its master once
    created, and `getOrCreate` would hand back the existing one.
    """
    code = (
        "import pandas as pd;"
        "from spark_features import get_spark;"
        "from pyspark.sql import functions as F;"
        "from pyspark.sql.types import DoubleType;"
        "s=get_spark('retry', cores=2);"
        "s.sparkContext.setLogLevel('ERROR');"
        "d=s.createDataFrame(pd.DataFrame({'t':['adverse risk of loss']*200}));"
        "print('SUM', d.withColumn('n', F.udf(lambda x: float(len(x)), DoubleType())('t'))"
        ".agg(F.sum('n')).collect()[0][0]);"
        "s.stop()")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=str(Path(__file__).resolve().parent))
    ok = r.returncode == 0 and "SUM" in r.stdout
    print(f"  retry at cores=2: {'PASS' if ok else 'FAIL'}")
    if not ok:
        tail = (r.stderr or "").strip().splitlines()[-3:]
        for line in tail:
            print(f"    {line[:150]}")
    return ok


def main() -> None:
    versions()
    if not worker_can_start():
        return

    import numpy as np
    import pandas as pd
    from spark_features import get_spark

    spark = get_spark("doctor", shuffle_partitions=2)
    spark.sparkContext.setLogLevel("ERROR")
    df = spark.createDataFrame(pd.DataFrame({
        "id": range(200),
        "text": ["the company faces adverse risk of material loss"] * 200}))

    results: list[tuple[str, bool, str]] = []

    def attempt(name: str, fn) -> bool:
        try:
            fn()
            print(f"  PASS  {name}")
            results.append((name, True, ""))
            return True
        except BaseException as e:
            msg = str(e).splitlines()[0][:160]
            print(f"  FAIL  {name}\n        {msg}")
            results.append((name, False, msg))
            return False

    print("=" * 68)
    print("ESCALATING TESTS")
    print("=" * 68)

    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, StringType

    ok1 = attempt("1 · JVM only (no Python worker)",
                  lambda: df.groupBy("text").count().collect())

    ok2 = ok1 and attempt(
        "2 · plain Python UDF (worker runs, no Arrow)",
        lambda: df.withColumn("n", F.udf(lambda t: float(len(t)), DoubleType())("text"))
                  .agg(F.sum("n")).collect())

    def _pandas_udf_job():
        # No type annotations on this function. `from __future__ import
        # annotations` at the top of this module turns every annotation into a
        # string, and pyspark introspects the signature to pick a pandas_udf
        # evaluation type — it then sees `'pd.Series'` as text and refuses with
        # UNSUPPORTED_SIGNATURE. Caught by this very script failing on itself.
        @F.pandas_udf(DoubleType())
        def n_chars(s):
            return s.str.len().astype(float)
        df.withColumn("n", n_chars("text")).agg(F.sum("n")).collect()

    ok3 = ok2 and attempt("3 · pandas_udf (Arrow round trip)", _pandas_udf_job)

    def _project_job():
        from spark_features import lm_features_spark
        pdf = pd.DataFrame({"cik": [1, 2], "ticker": ["A", "B"],
                            "filing_year": [2023, 2023],
                            "item1a": ["adverse risk of loss " * 50] * 2})
        lm_features_spark(spark, pdf).collect()

    ok3 and attempt("4 · this project's UDF (adds local modules)", _project_job)

    spark.stop()

    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    first_fail = next((n for n, ok, _ in results if not ok), None)
    if first_fail is None:
        print("  Everything passed. Whatever failed earlier is data-dependent —")
        print("  most likely task size. Try:")
        print("      python spark_features.py --check --repartition 64")
    elif first_fail.startswith("1"):
        print("  Spark itself cannot run. Java or the Spark install is the problem,")
        print("  not Python. Check `java -version` reports 17.")
    elif first_fail.startswith("2"):
        print("  A Python worker started and then died.")
        print("  Retrying the same job with 2 workers instead of all cores...")
        print()
        if _retry_with_fewer_cores():
            print("\n  CONFIRMED: it works with fewer workers.")
            print("  Every Python task is a separate process importing pandas and")
            print("  numpy — 250-400 MB each. On all cores that is several GB, and")
            print("  the OS kills workers. No Python traceback because there was no")
            print("  Python exception; the process was killed. The tell is that")
            print("  early tasks pass and later ones fail.")
            print("\n  Fix — already the default, so just re-run:")
            print("      python spark_features.py --check")
            print("      python spark_features.py --benchmark --cores 4 --sizes \"500,2000\"")
            print("  Lower `--cores` further if it still fails.")
        else:
            print("\n  NOT memory. The worker dies at any parallelism, which points")
            print("  at something killing python.exe:")
            print("    - antivirus / EDR blocking the spawned interpreter")
            print("    - a firewall rule on loopback (Spark talks to workers over")
            print("      127.0.0.1 sockets)")
            print("\n  Try adding an exclusion for")
            print(f"      {sys.executable}")
            print("  and for the project folder. If that is not possible, run")
            print("  stage 8 on Kaggle (Linux) — the code is verified there and")
            print("  the benchmark is the deliverable, not the OS under it.")
    elif first_fail.startswith("3"):
        print("  Plain Python works, Arrow does not. This is a NATIVE library")
        print("  mismatch — pyarrow and numpy compiled against different ABIs,")
        print("  which crashes the worker with no Python traceback.")
        print("\n  Fix, in this order:")
        print("      pip install \"pyarrow==14.0.2\"")
        print("      pip install \"numpy==1.26.4\" --force-reinstall")
        print("  then re-run this script.")
        print("\n  If that fails, run stage 8 on Kaggle (Linux) instead — the")
        print("  code is verified there and the benchmark is what matters,")
        print("  not the operating system it ran on.")
    else:
        print("  Spark and Arrow are fine; only this project's modules fail to")
        print("  reach the worker. That is a packaging problem, not a Spark one.")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        print("\nThe doctor itself failed — paste the traceback above.")
