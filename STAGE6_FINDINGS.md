# Stage 6 — the RAG evaluation, across two runs

Run 1 produced three wrong numbers. Run 2 fixed them and produced two more —
one of them in a diagnostic I had written specifically to catch this kind of
thing. Both runs are below, because the movement between them is the finding.

```
                       run 1 (screen 0.45)   run 2 (screen 0.30)
answered (unanswerable)             0.9975                0.9920   at thr 0.45
grounded verbatim                   0.9850                1.0000
grounding overlap 1gram             1.0000                1.0000   (saturated)
grounding overlap 2gram                  —                0.9855   -> now 1.0000
retrieval recall@k                       —                0.2700
attribution@1                       0.1300                0.1300
citation ceiling                         —                0.4815
abstention AUC                      0.6520                0.8213
median ms                            195.9                 186.2
```

---

## The headline

**The abstention AUC moved from 0.652 to 0.8213 without a single line of the
system changing.** The first number was measuring a contaminated negative
control. That is the most transferable thing in this stage: a metric is only as
good as the set it is measured against, and *absent from the corpus* is not the
same as *unanswerable* when the corpus is 10-K risk boilerplate.

**The operating point is the number to quote, not the AUC.** At the default
0.45 the system answered 99.2% of the questions it could not answer — because
0.45 was a constant nobody chose. Naming the budget instead:

| budget on false answers | threshold | coverage | actual false-answer rate |
|---|---|---|---|
| 10% | **0.77** | **51.5%** | **8.0%** |

Half the answerable questions refused is the price. Saying so is the point.

`DEFAULT_ABSTAIN_AT` is now **0.77**, with the derivation in the docstring.

---

## Fixed in run 2

**`grounded verbatim` 0.9850 → 1.0000.** The check recovered the answer's
sentences by re-splitting the joined text; join-then-split is not the identity.
A sentence ending in an abbreviation glued to one starting lower-case came back
as one string present in no passage. Spans are now carried, and each is matched
against *its own* passage rather than all citations concatenated — strictly
harder, since concatenation accepts a sentence straddling two filings.

**The negative control.** Screen 0.45 → 0.30, cutting the set from 400 to 249.

**`attribution@k` → `attribution@cited`**, plus `retrieval recall@k` and
`citation ceiling` split out of the single number.

---

## Found in run 2

### 1 · My "is the screen working" warning could only ever fire

It warned when the retained maximum sat within 0.01 of the threshold. It fired
at 0.45 (max 0.447) and fired again at 0.30 (max 0.299) — because on any
distribution whose density rises toward the cut, the survivors *always* bunch at
the ceiling. It is the exact sin this project keeps catching elsewhere: **a
control that can only come out one way is not a control.**

Replaced with the share of the candidate pool **rejected**, which can come out
either way, plus a hand-audit cell that prints the ten unanswerable questions
the system was most confident about. Stage 5 withdrew a finding on a
ten-item hand audit; the same discipline decides whether 0.30 was enough here.

### 2 · The bigram metric had a boundary artefact

`grounding overlap (2gram)` read 0.9855 for extractive mode, where it must be
1.0 by construction. Two artefacts, in opposite directions:

- the bigram straddling two joined *answer* sentences was written by nobody,
  exists in no filing, and was counted as a **miss**
- the bigram straddling two concatenated *source* passages is equally fictional
  and was counted as a **hit** — so a sentence stitched across two filings could
  pass as grounded

Both vanish once each span is scored against its own passage.

### 3 · flan-t5 was being graded on text it never read

The run printed `Token indices sequence length is longer than the specified
maximum sequence length for this model (546 > 512)`. Not cosmetic: the encoder
window is 512, so the tail of the prompt — the third retrieved passage — was
dropped, **while the evaluation still credited that passage as a citation.**
Context is now fitted to the window and only passages that fit are cited.

### 4 · flan-t5-base is barely generative

100% bigram overlap with the source alongside **15.6%** verbatim sentence
containment means it is copying spans and stopping mid-sentence, not
paraphrasing. So it is a weak test of hallucination. That is a limitation of a
250 MB model, stated honestly — not a clean bill of health for RAG.

### 5 · Retrieval, not answering, is the ceiling

Of 327 wrong citations, **292 never had the gold passage retrieved at all**;
only 35 were the answerer quoting the wrong thing from a set containing the
right one. Citation ceiling 0.4815 vs attribution@1 0.1300 says the same thing.
The next unit of work is the embedder — which is where stage 5 pointed too.

---

## Methodology change worth defending

The evaluation now runs with **abstention switched off** (`abstain_at = 0.0`)
and applies the operating point afterwards via `report_at_threshold`.

Measuring at the shipping threshold destroys the information needed to choose
it: every refused question has no citation and no grounding to grade, so the
sweep can only look upward from wherever the constant happened to sit. Grading
everything and thresholding after is exact in that direction — raising the bar
only removes answers, so the answered set is always a subset of one already
graded. Lowering it is **refused with an error** rather than approximated.

---

## The CV bullet — earned, with real numbers

> Extended it to a **RAG** layer returning **grounded answers with filing-level
> citations** — **100% verbatim grounding** verified as a falsifiable control,
> not assumed — and scored **abstention as a binary classifier** (**ROC-AUC
> 0.82**) against a negative control of **249 real questions** from filings held
> out of the index, choosing the operating point from a stated false-answer
> budget (**0.77 → 51.5% coverage at 8.0% false answers**); cleaning a
> contaminated control moved the AUC from **0.65 to 0.82** with no change to the
> system

If you need it shorter, cut the last clause — but it is the best part.

### Interview answers this earns

**"Why is your AUC 0.82 and not higher?"** Because 10-K risk language is
boilerplate. My negative control is real questions from filings held out of the
index, and some of them have genuine answers in other firms' filings. I screen
for that with TF-IDF and I keep the screen lexical on purpose — screening with
the same encoder that produces the confidence score would raise the AUC by
construction without the system improving at all.

**"You reported 51% coverage. Isn't that bad?"** It is the number I chose. The
constraint was at most 10% false answers, and 51% coverage is what that bought.
A confident wrong citation is worse than silence, because the citation is what
makes a reader stop checking — so I did not optimise F1, which would have
treated those two errors as equal.

---

## To re-run

```powershell
.\.venv\Scripts\python.exe -m pytest test_rag.py -q     # expect 17 passed
```
then `07_rag.ipynb` → **Kernel → Restart and Run All**.

Expect to change: extractive `grounding overlap (2gram)` should now read exactly
**1.0000**, and the generative row should no longer print the 546 > 512 warning.
Everything else should reproduce. Then read the hand-audit cell — if more than
two or three of those ten have real answers in the corpus, drop `SCREEN` to 0.25
and re-run from cell 5, and quote the audited count alongside the AUC.
