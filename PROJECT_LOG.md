# PROJECT LOG — edgar-credit-signal

**Paste this whole file into a new chat to carry the full context forward.**
Everything below is state, not narrative: what exists, what it measures, what
broke and why, and what is still open.

---

## 0 · Who and what

**Rashtra Mitra Tripathi** — MA Applied Quantitative Finance, Madras School of
Economics, 2025–27. Summer intern at **Moody's, Structured Finance Group**
(June–Aug 2026).

**Target role:** Citi **AIM Data Science Trainee** (JD C09_DS_003). The JD's
highest-frequency asks, in order: **RAG / Gen AI** (named twice), **semantic
search** ("esp semantic search"), **agentic solutions (LangChain / Google Agent
Space)**, **NLP / LLM / prompt engineering**, **FastAPI / ADK / deep learning**,
**PySpark / big data**, then the standard preprocessing / feature engineering /
EDA / ROC-AUC / Python / SQL / Git stack.

**The project:** an end-to-end NLP, retrieval and RAG system over 8,801 SEC
10-K filings that reports a null result, a measured retrieval benchmark, and a
finding it withdrew — with ~100 regression tests encoding the specific ways each
stage was previously wrong.

---

## 1 · Hard constraints (do not violate)

- **Project lives on `X:\Projects\edgar-credit-signal`.** Never C: — only ~10 GB
  free there.
- **No LLM API key. Everything runs locally.** `all-MiniLM-L6-v2` for embeddings,
  `flan-t5-base` for the generative comparison. Do not propose OpenAI/Anthropic
  API solutions.
- **Never claim a JD keyword that was not actually built and run.** The standing
  rule: *"A keyword that clears the ATS and dies in the interview is worse than a
  missing one."* Code that exists but has not been executed end-to-end does not
  go on the CV.
- **CV is one page.** Three bullets per heading. Skills section fits ~3 physical
  lines.
- **Difficulty must be defensible in an interview.** Every number must have a
  story about how it could have been wrong.
- Windows machine, VS Code, `.venv` per-terminal. **VS Code has twice restored
  stale notebook copies** over committed files — always close and reopen notebook
  tabs after files are written, and use *Kernel → Restart and Run All*.

---

## 2 · Stage status

| # | Stage | Status |
|---|---|---|
| 1 | Ingestion (SEC EDGAR REST API) | **Done** |
| 2 | Section extraction (Item 1A) | **Done** |
| 3 | Text features (TF-IDF, Loughran-McDonald) | **Done** |
| 4 | Signal test (logistic regression + permutation) | **Done — null result** |
| 5 | Retrieval benchmark (BM25 / dense / RRF) | **Done — finding withdrawn** |
| 6 | RAG + evaluation | **Done — two runs, all defects fixed** |
| 7 | FastAPI service | **Written + unit-tested, NEVER RUN** |
| 8 | PySpark port | **Written, correct on Linux, dies on Windows** |

---

## 3 · Every headline number

### Stage 1–3 — corpus
- **8,801** SEC 10-K filings, **1,499** firms, 2019–24, **1.0 GB**
- REST API ingestion at **99.1%** yield
- Two-tier section extraction at **80.8%** recovery
- **5,310** firm-years of features

### Stage 4 — the null result
- Logistic regression, out-of-time split: **0.7612 ROC-AUC**
- Incremental text signal: **ΔAUC −0.0022, p 0.313** → **no signal**
- Negative control: injected noise promoted **8%** of the time vs 5% nominal,
  binomial **p 0.24** → gate is calibrated
- Positive control: known driver detected at **z +10.3** → test has power
- Diagnosed extraction defect fabricating **~3,800 identical records** behind a
  reported 91% success rate

### Stage 5 — retrieval benchmark
- **45,995** passages, **433** firms, index v5.7
- **384-dim** sentence embeddings (`all-MiniLM-L6-v2`), FAISS `IndexFlatIP`
- **2,000** queries; BM25 **MRR 0.2124** vs dense **0.1383**
- Stratified by query–passage vocabulary overlap: BM25's lead sits almost
  entirely in the high-overlap tercile
- Low-overlap tercile showed dense +108% — **withdrawn** after a hand audit found
  **4 of 10** pairs were junk (dividers, page furniture, fragments)
- Reported result: hybrid RRF wins cleanly in the **middle stratum**

### Stage 6 — RAG (final, run 2 + fixes)
- **grounded verbatim 1.0000** (extractive; 1.0 by construction, verified)
- **abstention ROC-AUC 0.8213** against **249** questions from held-out filings
- Operating point from a **10% false-answer budget**: threshold **0.77** →
  **51.5% coverage at 8.0% false answers**
- `DEFAULT_ABSTAIN_AT = 0.77` (was 0.45, which answered **99.2%** of
  unanswerable questions)
- retrieval recall@k **0.27** · attribution@1 **0.13** · citation ceiling **0.48**
- Of 327 wrong citations: **292 never had the gold passage retrieved**, only 35
  were the answerer quoting wrong from a set containing the right one
- median latency **186 ms** extractive, **~1,100 ms** generative
- `flan-t5-base`: 100% bigram overlap with source alongside **15.6%** verbatim
  sentence containment → it is copying spans, not paraphrasing

**The single best sentence in the project:** the abstention AUC moved from
**0.652 → 0.8213 without a line of the system changing** — the first number was
measuring a contaminated negative control.

---

## 4 · Bug catalogue (the intellectual core — do not lose this)

Each entry is a *control that failed*, which is the whole methodological point.

### Stage 5
1. **BM25 memory** — nested dicts needed GB at 200k passages. Rewritten onto
   scipy CSR; `test_bm25_matches_reference` proves identical arithmetic.
2. **`array('l')` portability** — 4 bytes on Windows, 8 on Linux. Switched to
   `'q'`.
3. **Heading leak into passage bodies** — eval queries *are* headings, so one
   leak lets BM25 win by string match and the comparison measures nothing. Found
   because the repair and the audit used **different matching rules**. 10 leaks →
   0 across v5.3–v5.7.
4. **Token truncation** — 2.1% of chunks exceeded the 256-token window. Also the
   *control was wrong* (counted "exactly full" as truncated).
5. **Junk eval queries** — 15.1% were page furniture, dividers, fragments. Fixed
   at build time via `is_usable_query`.
6. **`_substring_leak_pass` carved mid-sentence**, producing *"We are subject to
   the , including…"* — now drops the passage instead.
7. **HTML entities never decoded** (`&#x2022;` as literal text) — filings
   double-escape.
8. **The 108% "dense wins" finding was withdrawn** after a 10-pair hand audit.

### Stage 6 — run 1
9. **`grounded verbatim = 0.9850`** in a mode that *cannot* hallucinate. The
   checker recovered sentences by re-splitting joined text; join-then-split is
   not the identity (a sentence ending "U.S." glued to one starting lower-case
   returns as one string present in no passage). Now carries `spans` and matches
   each against **its own** passage.
10. **Contaminated negative control** — screen at 0.45, survivors topped out at
    0.447. *"Our indebtedness could adversely affect our financial **health**"*
    answered from another firm's *"…financial **condition**"* at 0.883. Screen
    tightened to **0.30**.
11. **`grounding overlap = 1.0000`** — saturated, not good. Bigram version added.
12. **`attribution@k` was three questions in one** — split into retrieval
    recall@k / attribution@1 / citation ceiling, and renamed
    **attribution@cited** (extractive cites 1–2 of 5 retrieved; "@5" implied
    chances it never had).

### Stage 6 — run 2
13. **My own "is the screen working" warning could only ever fire.** It warned
    when the retained max sat near the threshold — but on any distribution whose
    density rises toward the cut, survivors *always* bunch at the ceiling.
    Replaced with **share of pool rejected** + a hand-audit cell.
14. **Bigram boundary artefact** — the pair straddling two joined answer
    sentences was written by nobody (counted a miss); the pair straddling two
    concatenated source passages was equally fictional (counted a hit). Both
    vanish once each span is scored against its own passage.
15. **flan-t5 graded on text it never read** — `546 > 512` token warning meant
    the third retrieved passage was dropped from the prompt **while still being
    credited as a citation**. Context now fitted to the window; only passages
    that fit are cited.

### Stage 6 — run 3
16. **`NameError: text is not defined`** — an edit anchored on
    `text = gen(prompt)` deleted it. **22 tests stayed green because generative
    mode had zero coverage** (every test needed a 250 MB download, so none ran).
    Added a stub-generator fixture; writing it immediately found bug 17.
17. **The "nothing fits" fallback cut at a hardcoded 1,200 chars**,
    re-introducing the exact silent overflow the fitting loop existed to prevent.
    Now truncates to budget; an impossible window raises.

### Stage 8
18. **`.count()` prunes UDFs** — first benchmark reported the plain UDF *faster*
    than Arrow, which is impossible. Fixed with `noop` sink + warm-up.
19. **Spark `Tokenizer` keeps punctuation; IDF smoothing differs from sklearn** —
    moved max abs diff 0.2444 → 0.0000.
20. **Java 21 vs Spark 3.5** — Arrow fails with `sun.misc.Unsafe not available`.
    8 tests fail on 21, 8 pass on 17.

---

## 5 · Repo inventory (`X:\Projects\edgar-credit-signal`)

**Core modules:** `config.py` `sec_client.py` `download_filings.py`
`build_universe.py` `extract_sections.py` `sections.py` `chunking.py` (v5.7)
`text_features.py` `fundamentals.py` `signal_test.py` `retrieval.py`
`build_index.py` `rag.py` `rag_eval.py` `api.py` `loadtest.py`
`spark_features.py` `spark_doctor.py` `setup_java.py` `viz.py` `status.py`

**Notebooks:** `01_ingest` `02_sections` `03_text_features` `04_signal_test`
`05_retrieval_benchmark` `06_retrieval_analysis` `07_rag`
`kaggle_spark_benchmark`

**Tests (~100):** `test_parsing` `test_sections` `test_signal_controls`
`test_retrieval` (39) `test_rag` (**22**) `test_api` (15, unrun)
`test_spark_features`

**Docs:** `README.md` `WRITEUP.md` `STAGE5_FINDINGS.md` `STAGE6_FINDINGS.md`
`RAG_DEFENCE.md` `CV_ATS_FINAL.md` `CV_POINTERS.md` `CV_TIGHTENED.md`
`CV_FINAL.md` `INTERVIEW_DEFENSE.md` `RAG_RUN.md` `API_RUN.md` `SPARK_RUN.md`
`VSCODE.md`

---

## 6 · CV — current final state

### Moody's — Summer Intern, Structured Finance Group · June–Aug 2026

- Ran **exploratory data analysis (EDA)** on a regional structured finance
  portfolio and presented the findings: maturity-wall concentration in
  lower-rated issuers, CLO capacity covering two-thirds of refinancing need
- Automated data validation and the monthly **data quality** review in
  **Python** and **SQL Server**: 3.4M field-level comparisons for a legacy
  platform migration, rule-level **SQL** checks generated dynamically, runtime
  cut from hours to minutes
- **APAC Champions, Top 5 globally** (team of 6), Moody's AI Innovation World
  Cup: built a hallucination-resistant **LLM agent** explaining credit rating
  methodologies, using **prompt engineering** and grounded-response design

### Risk-Factor Text as a Credit Signal — NLP, Semantic Search and RAG (Gen AI) System · Aug 2026

- Turned **8,801 SEC 10-K filings** into an analysis-ready dataset: **REST API**
  ingestion at 99.1% yield, 80.8% section-extraction recovery, then **data
  preprocessing, tokenisation and feature engineering** (**TF-IDF**,
  Loughran-McDonald sentiment) across **5,310 firm-years**
- **No incremental signal** from the text beyond financial ratios: **logistic
  regression** on an **out-of-time split** (0.7612 **ROC-AUC**), controlled
  **permutation test** (ΔAUC −0.0022, p 0.313) — after tracing a defect that
  fabricated **~3,800 duplicate records** behind a reported 91% success rate
- Built **semantic search** over **45,995 passages** — **transformer sentence
  embeddings**, **FAISS vector database**, benchmarked against **BM25** — then a
  **RAG (Gen AI)** layer with **100% verbatim-grounded citations** that
  **refuses unanswerable questions**: **ROC-AUC 0.82** against **249 held-out
  filings**, 8.0% **false-answer rate**

### Consumer Credit Risk P&D Engine · Aug 2026

- Built a **WOE/IV** application scorecard on a 2.26M-loan consumer portfolio
  (454k train / 376k **out-of-time**): 0.7076 **AUC**, 0.4152 Gini, 0.2986 KS; a
  3.0pp default under-prediction priced at **$115m on $5.50bn** exposure
- Matched the lender's proprietary risk grade (0.7100) using public borrower
  attributes alone; a tuned **XGBoost** added only **+0.0073 AUC** once pricing
  leakage was removed
- Estimated lifetime **PD** with an **Aalen-Johansen** competing-risks
  estimator, correcting a **111% Kaplan-Meier overstatement**; mapped to a
  7-grade scale monotonic from 3.3% to 49.6% realised default

### Cardmember Profitability and Portfolio P&L Model — AmEx Campus Challenge · July 2026

- Built a **customer-level P&L model** across 500k card accounts — interchange,
  net interest income, reward and benefit cost, **ECL** — identifying the most
  profitable at **86.5% top-quintile accuracy**
- Ranked 11 parameters by their effect on the top-quintile boundary and targeted
  three changes at the strongest: **+0.093 (0.772→0.865)** at a 0.5c/point cost
  against 1–2c face value
- Diagnosed **MNAR missingness** across five data blocks with held-out
  **logistic regression** (AUC 0.62 to 1.00), encoding informative indicators
  instead of mean imputation under SHA-256 data-contract governance

### Skills (fits 3 lines)

```
Skills: Python (pandas, NumPy, scikit-learn), SQL, Git

NLP & Gen AI: semantic search, Retrieval-Augmented Generation (RAG),
Large Language Models (LLM), prompt engineering

ML & Statistics: classification, logistic regression, decision trees, feature
engineering, EDA, ROC-AUC, hypothesis testing
```

### CV structure decisions already made
- **Academic Achievements section: dropped** (quiz wins carried no number; the
  GPA line had no room in Education).
- **Chief Editor entry: dropped** (−2 lines).
- **Placement Coordinator sub-bullet: dropped**, title line kept (−1 line).
- **President + ₹37,000 budget to 13% surplus: KEPT** — the only leadership claim
  with a number.
- Project heading changed from *"NLP Falsification Study"* to *"NLP, Semantic
  Search and RAG (Gen AI) System"* — headings weigh more in parsers.
- The **withdrawn 108% finding** no longer fits on the page; it lives in the
  interview answers instead.

---

## 7 · What is still open

**1 · FastAPI — highest value remaining.** Listed under the JD's *Essential
Skills*; currently scores zero. Written and unit-tested, never run.

```powershell
.\.venv\Scripts\python.exe -m pytest test_api.py -q     # expect 15 passed
.\.venv\Scripts\python.exe -m uvicorn api:app --port 8000
.\.venv\Scripts\python.exe loadtest.py                  # second terminal
```
Then add, with real numbers from `loadtest_results.json`:
> Served the stack behind a **FastAPI** REST service (`/search`, `/answer`,
> `/score`) with request validation and per-route latency percentiles,
> load-tested to **p95 X ms at N concurrent requests**

**2 · Notebook 06** has not been re-run on the rebuilt v5.7 index.

**3 · PySpark** — a *preferred* qualification. Windows workers die (diagnosed as
far as: not memory, not Arrow ABI; suspect antivirus/EDR or loopback firewall).
`kaggle_spark_benchmark.ipynb` is ready — upload four files as a dataset, Run
All, ten minutes on Linux.

**4 · Next technical step, if any:** swap `all-MiniLM-L6-v2` for a
finance-domain embedder and re-run stages 5 and 6 unchanged. One-variable
experiment — every control, metric and table stays, only the vectors change.
Justified by: 292 of 327 wrong citations never had the gold passage retrieved.

---

## 8 · Never claim these

**LangChain · Google Agent Space · ADK** — in the JD, not used. The Moody's LLM
agent is the honest neighbour.
**Clustering · A/B testing · F1-score** — in the JD, never built.
**Deep learning · cloud (AWS/Azure/GCP) · Tableau · Power BI · R** — no.
**PySpark · FastAPI** — until actually executed.

---

## 9 · Principles this project established

- A control that can only come out one way is not a control. (Caught three
  times: the grounding check, the screen warning, the truncation control.)
- Make the repair and the audit use **different** matching rules.
- A metric is only as good as the set it is measured against.
- Screen a negative control with a measure **independent** of the score under
  test — TF-IDF for the screen, dense cosine for the confidence.
- Name the cost before choosing the operating point. Never optimise F1 where the
  two error types are not equally bad.
- An untested branch is untested whether the excuse is a model download or
  anything else.
- Withdraw findings that a hand audit does not support. Ten pairs read by hand
  beat any aggregate.
