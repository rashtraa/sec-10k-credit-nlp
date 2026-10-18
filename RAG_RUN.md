# Stage 6 — RAG with citations and refusals

Earns **RAG**, **grounded generation**, **hallucination evaluation** and
**abstention** on your CV. Reuses the stage-5 index untouched — nothing is
re-embedded, nothing is re-downloaded except optionally a 250 MB generator.

---

## 1 · Install

```powershell
pip install -r requirements-stage6.txt
```

One package: `sentencepiece`, which the T5 tokenizer needs and whose absence
produces an error message that does not mention sentencepiece. Skip it only if
you intend never to run generative mode.

---

## 2 · Tests first

```powershell
pytest test_rag.py -q
```

**11 passed**, under a second, no downloads. They run against a stub encoder
on purpose: the logic being checked — does it cite what it quoted, does it
refuse when nothing matches, can the grounding check actually fail — should
not depend on a model download or a torch version.

The two that matter most:

- `test_extractive_answer_is_always_verbatim` — the control the whole stage
  rests on
- `test_grounding_check_can_actually_fail` — because a control that cannot
  fail proves nothing

---

## 3 · Run the notebook

Open **`07_rag.ipynb`**, select the `.venv` kernel, Run All.

Roughly four minutes for the extractive path. Then set `RUN_GENERATIVE = True`
(it already is) if you want the flan-t5 comparison — that adds about four more
minutes and a 250 MB download on first run.

### What to check, in order

**Read the four sample answers first.** Two answerable, two not. If the two
unanswerable ones are not refused, stop and tell me before reading any table.

**`grounded verbatim` must be 1.0000.** In extractive mode the answer is a
span of a cited passage. Anything less is a pipeline bug, not a model
limitation. It is a control, not a result — never quote it as evidence the
system is good.

**`answered (unanswerable)`** is the false-answer rate: how often it
confidently answered about a company it has never seen. This is the number a
risk function asks about first.

**`abstention AUC`** says whether confidence separates the two question sets
at all. A system that abstains on everything has a perfect false-answer rate
and is useless — which is why refusal rate alone proves nothing.

**Read the failure cases.** The notebook prints wrong-citation answers and
answered-anyway questions. On this corpus most "false answers" are questions
another firm genuinely also asked, which is a limitation of the screening, not
a hallucination. Know that before an interviewer points it out.

---

## 4 · What is actually being measured

The unanswerable questions are **not invented**. Your index holds 500 filings;
thousands more were parsed and left out. Their headings are real
analyst-style questions about companies the index has never seen.

The catch, and the reason there is a screen: risk language repeats across
firms, so a question from an unindexed filing might still be answerable from
an indexed one. Every candidate is checked by TF-IDF similarity against the
whole index and anything with a plausible answer already present is dropped.
Without that, the "unanswerable" set would be partly answerable and the AUC
would measure something else entirely.

That screen is the part to describe in an interview. It is the difference
between a negative control and a decoration.

---

## 5 · Numbers to record

From `report_table`:

| what | where it goes |
|---|---|
| grounded verbatim (extractive) | the control — quote as 100%, call it a control |
| false-answer rate on unanswerable | the headline safety number |
| abstention AUC | proof the refusal is informed, not blanket |
| attribution@1 | how often the cited filing is the right one |
| generative grounding overlap | the honest "is RAG safe" number, if you ran it |
| median ms | latency, for the FastAPI stage |

---

## 6 · Known limitations, state them yourself

The extractive answer is a sentence lifted from a filing, not a synthesis
across filings. That is a deliberate trade — it buys a grounding guarantee no
generative system can offer — but it means "compare supply-chain risk at three
companies" is out of scope.

Confidence is cosine similarity to the best retrieved passage. It is a
retrieval score being used as an answerability score, which works because they
are correlated, not because they are the same thing. A trained calibrator over
retrieval features would be better and is the obvious improvement.

The corpus is 500 filings. A question about a company outside that sample is
unanswerable for a boring reason rather than an interesting one.
