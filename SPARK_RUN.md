# Stage 8 — PySpark, and the honest benchmark

Earns **PySpark**, **Spark SQL**, **Spark ML**, **pandas_udf / Arrow** and
**distributed-computing judgement** on your CV. That last one is the real
deliverable: the point is not that you used Spark, it is that you **measured
when Spark is the wrong tool** and can say so with a number.

---

## 1 · Java first — this is where it goes wrong

**Spark 3.5 supports Java 8, 11 and 17. Not 21.** Anaconda ships 21, so this
is probably you.

```powershell
java -version
```

If that says 21, every `pandas_udf` will die inside Arrow with a hundred lines
of JVM stack trace whose top frame mentions `sun.misc.Unsafe` and never
mentions Java versions. I hit exactly this while building the module — same
code, 8 tests failing on 21 and 8 passing on 17.

### Getting Java 17 without admin rights

```powershell
python setup_java.py
```

Unpacks a Temurin 17 JDK into `.java/` inside the project. No installer, no
administrator prompt, no system PATH edits — and `spark_features.py` finds it
automatically, so **nothing needs `JAVA_HOME` set and no new terminal behaves
differently**. That last part matters: "works in one window and not the next"
is its own category of wasted evening.

If the download is blocked, do it by hand — still no admin needed:

1. <https://adoptium.net/temurin/releases/?version=17>, Windows x64,
   package type **.zip** (not `.msi`)
2. extract so you end up with `.java\jdk-17...\bin\java.exe`
3. `python setup_java.py` to confirm

`.java/` is 300 MB of binaries and is already in `.gitignore`.

If you would rather use a system-wide install, set `JAVA_HOME` to a Java 17
and that takes precedence. `spark_features.py` checks the version at startup
and prints a sentence instead of the stack trace either way.

```powershell
pip install -r requirements-stage8.txt
pytest test_spark_features.py -q
```

**8 passed.** Five of them test the pandas reference implementations
unconditionally — those are what Spark gets checked *against*, so if PySpark
is missing they must still run, or the correctness control would be agreeing
with itself.

If Windows complains about `HADOOP_HOME` or `winutils.exe`, it is a warning,
not an error — nothing here writes through Hadoop's filesystem layer.

### The other Windows trap, already handled

Spark launches a worker process per task by executing `python` — it does not
inherit the interpreter running the driver, it looks the name up on PATH. On
Windows that resolves to the **Microsoft Store App Execution Alias**, a stub
that prints *"Python was not found; run without arguments to install from the
Microsoft Store"* and exits. The worker never connects back, and Spark reports
it four hundred lines later as

```
org.apache.spark.SparkException: Python worker failed to connect back
Caused by: java.net.SocketTimeoutException: Accept timed out
```

which points at sockets and never mentions Python. Activating a venv does not
save you: it sets PATH for your shell, but the alias still wins for a bare
`python` launched by a JVM subprocess.

`spark_features.py` now pins `PYSPARK_PYTHON` to `sys.executable` — by
definition the interpreter running the code, and the one that actually has
pandas and the lexicon. Nothing for you to set. If you ever see that socket
timeout in another Spark project, this is the first thing to check.

### And the third one: no fork on Windows

Once the worker launches, it can still die immediately with

```
org.apache.spark.SparkException: Python worker exited unexpectedly (crashed)
Caused by: java.io.EOFException
```

which describes a truncated stream and says nothing about the cause.

The cause is that cloudpickle serialises a function defined in `__main__` **by
value**, but one imported from a module **by reference**. `count_categories`
lives in `text_features`, so what crosses to the worker is "import
text_features and look it up" — not the code.

On Linux that works by accident: Spark runs `pyspark.daemon`, which **forks**
workers, and a forked process inherits the parent's `sys.path` including the
project directory. **Windows has no fork.** Spark spawns a fresh
`python -m pyspark.worker` per task with a clean path, the import fails, and
the worker dies before writing a byte. Identical code, passes on Linux, crashes
on Windows.

The textbook fix is `sparkContext.addPyFile(...)`. **It does not work on
Windows.** It routes through Hadoop's `FileUtil.chmod`, which shells out to
`winutils.exe`, so without a Hadoop install you get

```
java.io.FileNotFoundException: HADOOP_HOME and hadoop.home.dir are unset
```

— trading an import error for a dependency on a binary that has nothing to do
with this program.

What is used instead is `cloudpickle.register_pickle_by_value(text_features)`.
cloudpickle then embeds the module's source **in the pickle itself**: no file
to distribute, no path to configure, no Hadoop involved. It is also the more
honest statement of intent — this code belongs to the job, not to the cluster.

`spark.python.worker.faulthandler.enabled` is on too, so the next worker crash
arrives as a Python traceback rather than an `EOFException`.

---

## 1b · If a worker dies

```
org.apache.spark.SparkException: Python worker exited unexpectedly (crashed)
Caused by: java.io.EOFException
```

That message is not one failure mode, it is every failure mode — the JVM only
knows the pipe closed early. Do not read the stack trace; run the diagnostic:

```powershell
.\.venv\Scripts\python.exe spark_doctor.py
```

Four jobs of escalating dependency, and the first one that fails tells you
what is wrong. Each verdict names its own fix.

**The most likely answer is memory.** Every Python task is a *separate
process* importing pandas and numpy — 250-400 MB resident each. `local[*]`
runs one per core, so on a 16-core laptop that is several GB on top of the
driver JVM, and the OS starts killing workers. There is no Python traceback
because there was no Python exception: the process was killed. **The tell is
that early tasks succeed and later ones fail**, as memory pressure builds.

The default is now `local[4]`, not `local[*]`. Lower it further if needed:

```powershell
python spark_features.py --check --cores 2
```

On this workload the cost is close to zero — the job is bound by Python
function calls, and four processes already saturate what matters.

---

## 2 · Correctness before speed

```powershell
python spark_features.py --check
```

Expect every LM feature to read `match` and `risk_change` to show **max abs
diff 0.0000**. A faster pipeline that quietly disagrees with the one it
replaces is not a port; it is a second unvalidated pipeline, and the
disagreement surfaces later as a modelling result nobody can explain.

Getting to exact agreement took two fixes worth knowing about, because both
are the kind of thing that silently produces "close enough" numbers:

- **Spark's default `Tokenizer` splits on whitespace and keeps punctuation
  attached**, so `loss.` and `loss` hash to different buckets while sklearn
  treats them as one token. Using `RegexTokenizer` with sklearn's own pattern
  fixes it.
- **The two libraries smooth IDF differently** — sklearn uses
  `ln((1+n)/(1+df)) + 1`, Spark ML uses `ln((n+1)/(df+1))`. That trailing `+1`
  moved year-over-year cosine by up to **0.24**. The pandas reference now
  computes IDF by hand with Spark's formula, which turns "close enough, trust
  me" into an exact comparison.

---

## 3 · The benchmark

```powershell
python spark_features.py --benchmark --sizes 500,2000,7000
```

Writes `data\processed\spark_benchmark.csv`.

To push past the size of the real corpus and find the crossover, duplicate it:

```powershell
python spark_features.py --benchmark --sizes 7000 --replicate 4
```

Duplicated text is fine for a throughput measurement and useless for a
modelling one, which is why it lives only here.

### Read `spark_vs_pandas`

Above 1, Spark is **slower** — you are paying JVM startup, serialisation and
shuffle costs for nothing. The number to quote is where that ratio crosses 1.
On synthetic data at this scale it sat around **2.5x slower**, which is the
expected and honest answer for a corpus that fits comfortably in memory.

### Two methodology fixes, because the first version lied

**`.count()` does not evaluate a UDF.** Spark's optimiser prunes columns a
count does not need, so the UDF never runs. My first benchmark used it and
reported the row-at-a-time UDF as *faster* than the Arrow-batched one — which
is impossible, and was the clue. The fix is to write to the `noop` sink, which
runs the whole plan and discards the output.

**The first Spark action pays JVM class loading, JIT and codegen.** Timing it
as steady-state overstates Spark by seconds. There is now an untimed warm-up
pass before any measurement.

---

## 4 · The three Spark decisions to be able to defend

**`spark.sql.shuffle.partitions = 8`, not the default 200.** On a laptop the
default means 200 tiny tasks and their scheduling overhead for a shuffle that
should be eight. It is the most common reason a small Spark job is
embarrassingly slow, and leaving it would have made this a benchmark of a
misconfiguration.

**The Loughran-McDonald lexicon is broadcast.** Without that, a 4,000-word
dictionary is serialised and shipped with every task — the classic mistake
that makes a Spark job scale *worse* than the loop it replaced.

**`pandas_udf`, not `udf`.** Arrow moves a whole column into Python at once;
a row-at-a-time UDF pays serialisation per record. Both are in the benchmark
so the gap is a number rather than an assertion.

---

## 5 · What to record

| what | why |
|---|---|
| `spark_vs_pandas` at the real corpus size | the honest "Spark cost us N×" number |
| the size where that ratio crosses 1 | the crossover — the actual finding |
| `arrow_speedup` | pandas_udf vs plain udf, measured |
| correctness result | that the port agrees exactly with what it replaced |

---

## 6 · Say this yourself before an interviewer does

One machine, `local[*]`. This measures Spark's *overhead* honestly and says
nothing about how it scales across a cluster, where the shuffle and the IDF
pass become network-bound rather than free.

The crossover depends on the work per row. This pipeline does a lot of Python
per document, which flatters Spark's parallelism; a cheap transform over the
same rows would favour pandas further.

And the useful conclusion is not "Spark is bad". It is that **at 8,801
filings, Spark is the wrong tool, and here is the measurement** — plus the
size at which that flips.
