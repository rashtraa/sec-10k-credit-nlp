# The RAG bullet — made defensible

## The problem with the version you have

> Extended it to a **RAG (Gen AI)** system that answers with source citations and
> **refuses questions the filings cannot answer**: **ROC-AUC 0.82** on that
> refusal decision, 51.5% coverage at an 8.0% wrong-answer rate

Three things an interviewer breaks in under a minute.

**"0.82 against what?"** An AUC with no stated control set is not a measurement.
This is the first question anyone technical asks, and right now the bullet has
no answer on the page.

**"8.0% wrong-answer rate" reads as "8% of my answers are wrong."** It is not.
It is *8% of the questions the corpus cannot answer get answered anyway*. If
they read it the first way and you explain it the second way, it looks like you
inflated it and got caught. **This is the single most dangerous phrase on your
CV** — it is the one place where a plain-English simplification changed the
meaning rather than clarifying it.

**"Citations" claims nothing checkable.** Every RAG demo on the internet has
citations. Yours are *verified* — you have the strongest faithfulness result in
the project and it is not on the page.

---

## Use this

**Two-line version (recommended):**

> Extended it to a **RAG (Gen AI)** system with **100% verbatim-grounded
> citations** that **refuses questions the corpus cannot answer** — refusal
> scored as a classifier at **ROC-AUC 0.82** against **249 held-out-filing
> questions**, at 51.5% coverage for an **8.0% false-answer rate**

**Three-line version, if the page allows it:**

> Extended it to a **RAG (Gen AI)** system with verifiable citations — **100%
> verbatim grounding**, checked rather than assumed — that **refuses questions
> the corpus cannot answer**: refusal scored as a classifier at **ROC-AUC 0.82**
> against **249 questions from filings held out of the index**, with the
> operating point set by a stated error budget (**51.5% coverage, 8.0% false
> answers**)

Three changes, each closing one hole:

- **"249 held-out-filing questions"** names the control set. The AUC becomes a
  measurement instead of a number.
- **"false-answer rate"** replaces "wrong-answer rate". Precise, and it is the
  term a model-risk reviewer already uses.
- **"100% verbatim-grounded"** promotes your best result onto the page.

---

## The eight questions, with answers

### 1 · "ROC-AUC 0.82 against what negative set?"

> 249 real questions taken from filings held out of the index — risk-factor
> headings from companies the corpus has never seen. Not invented, because
> invented questions are too easy: a made-up question about tungsten is trivially
> refused and would inflate the AUC.

### 2 · "But risk language is boilerplate. Aren't some of those answerable?"

**Volunteer this before they say it.** It is the most interesting thing in the
stage.

> Yes, and that is exactly what broke my first run. The AUC came out 0.65 and
> the system answered 399 of 400 unanswerable questions. I read the failures and
> most were not failures — "Our indebtedness could adversely affect our financial
> **health**" was answered from another firm's "…financial **condition**", cited
> correctly, at 0.88 confidence. The system was right; my control was dirty. I
> screen candidates by TF-IDF similarity against the whole index, tightened the
> screen from 0.45 to 0.30, and the AUC went to 0.82 — **without a line of the
> system changing.**

### 3 · "Why TF-IDF for the screen and not the embedding model?"

The question that separates people who understand evaluation from people who
have read about it.

> Because screening with the same encoder that produces the confidence score
> would rig it. Every question the encoder finds similar to the index leaves the
> negative set, so by construction the survivors score low confidence and the AUC
> rises without the system improving at all. The screen has to be independent of
> the score under test. TF-IDF is lexical, the confidence is dense, and keeping
> them different is what makes 0.82 an honest number.

### 4 · "How often is the citation actually the right document?"

**The landmine. The answer is 13%, and you must say it first.**

> Attribution@1 is 0.13, and I'd rather tell you than have you find it. But the
> number decomposes: of 327 wrong citations, **292 never had the right passage
> retrieved at all** — only 35 were the answering layer quoting the wrong thing
> from a set that contained the right one. So retrieval is the ceiling, not the
> answering step. I measure that separately as a "citation ceiling" — attribution
> restricted to questions retrieval already solved — and it is 0.48. The fix is a
> finance-domain embedder, and it is a one-variable experiment: every control,
> metric and table stays, only the vectors change.

Then the point that makes 13% survivable:

> And a wrong citation is *visible*. The reader clicks through and sees the
> mismatch. That is categorically different from a fluent hallucination with no
> citation at all, which is the failure mode that actually hurts you in credit.

### 5 · "100% grounding sounds too convenient."

> It is a control, not an achievement — in extractive mode the answer is a span
> of the cited passage, so it is 1.00 by construction. I quote it as proof the
> plumbing is right, never as proof the system is good. And I check it anyway,
> because a control that can only pass is worthless: it read **0.985** on the
> first run. Not hallucination — the checker was recovering the answer's
> sentences by re-splitting the joined text, and join-then-split isn't the
> identity. A sentence ending in "U.S." glued to one starting lower-case comes
> back as one string that's in no passage.

### 6 · "Why 0.77? Did you tune it?"

> No — I named the cost first. The constraint was answering at most 10% of the
> questions the corpus can't answer; 0.77 is what satisfies that, and 51.5%
> coverage is what it bought. I deliberately didn't optimise F1, because F1
> treats a wrong answer and a missed one as equal and they aren't. A confident
> wrong citation is worse than silence — the citation is what makes a reader stop
> checking.

### 7 · "51% coverage. Isn't that a system that mostly doesn't work?"

> It's a system that mostly doesn't *guess*. The alternative was already
> measured: at the default threshold it answered 99% of the unanswerable
> questions. For a credit analyst screening filings I'd move the bar down and
> tolerate noise; for anything feeding a published memo I'd move it up. That's a
> business input, not a hyperparameter, which is why it's a table in the notebook
> rather than a constant in the code.

### 8 · "Which LLM did you use?"

> The answering path that I report on is extractive — it selects sentences from
> the retrieved passages, so it *cannot* hallucinate, and that gives me a
> 100%-grounded reference point to measure anything else against. I also ran
> `flan-t5-base` locally as the generative comparison. It turned out to be barely
> generative: 100% bigram overlap with the source alongside 15.6% verbatim
> sentence containment means it's copying spans and stopping mid-sentence, not
> paraphrasing. So it's a weak hallucination test, and I say so rather than
> reporting it as a clean bill of health.

---

## The one-sentence version

If you get thirty seconds:

> It's retrieval-augmented answering over 8,801 filings where the interesting
> property isn't the answers, it's the refusals — I score "do I know this?" as a
> classifier against 249 questions from filings held out of the index, at
> ROC-AUC 0.82, and the biggest lesson was that cleaning the control moved it
> from 0.65 to 0.82 without changing the system at all.

---

## Two things not to say

**Don't say "state of the art" or "production-ready."** You built it on a laptop
with no API key, on a 433-firm sample, and attribution@1 is 0.13. The honesty is
the differentiator — most candidates cannot describe a weakness in their own
project without being cornered into it.

**Don't lead with the AUC.** Lead with the refusal *behaviour* — that a system
which says "I don't know" is the one you can point at a credit portfolio. The
number supports the idea; it isn't the idea.
