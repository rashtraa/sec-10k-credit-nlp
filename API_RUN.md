# Stage 7 — the service

Earns **FastAPI**, **REST API design**, **latency benchmarking under
concurrency** and **async/thread-pool reasoning**. Sits on top of stages 5 and
6 without duplicating any of their logic — that is deliberate, and it is the
first thing to say about it.

---

## 1 · Install and run

```powershell
pip install -r requirements-stage7.txt
pytest test_api.py -q
```

**15 passed**, about a second, no index and no model needed — the tests inject
a stub pipeline. Then, in one terminal:

```powershell
uvicorn api:app --port 8000
```

First start takes ~20 seconds: it loads 47,391 passages, a 70 MB embedding
matrix, the BM25 index and the model. **Once**, not per request. Open
<http://127.0.0.1:8000/docs> — FastAPI generates that page from the type hints,
so the documentation cannot drift from the code.

---

## 2 · Try it

```powershell
curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" `
  -d '{\"query\": \"supply chain disruption from a limited number of vendors\", \"k\": 3}'

curl -X POST http://127.0.0.1:8000/answer -H "Content-Type: application/json" `
  -d '{\"query\": \"What happens if a company loses a key supplier?\"}'

curl -X POST http://127.0.0.1:8000/answer -H "Content-Type: application/json" `
  -d '{\"query\": \"What is the melting point of tungsten?\"}'
```

The third should come back `"abstained": true` with an empty answer and a
**200**, not an error. A refusal is a successful response — the request was
well-formed and the service did its job. Returning 404 would tell a caller to
retry, which is exactly wrong.

| endpoint | what it does |
|---|---|
| `POST /search` | retrieve passages — `method` selects hybrid, bm25 or dense |
| `POST /answer` | grounded answer with citations, or an explicit refusal |
| `POST /score` | Loughran-McDonald sentiment profile for arbitrary filing text |
| `GET /health` | index size, firm count, device, abstention threshold |
| `GET /metrics` | p50 / p95 / p99 latency per route, from this process |

`/score` ships the stage-4 null result in its own response body. Without that,
a caller would build a risk score on a feature set measured not to carry one.

---

## 3 · Load test — where the real numbers come from

Leave uvicorn running. In a second terminal:

```powershell
python loadtest.py
```

Sweeps concurrency 1 → 16 against `/search` and `/answer`, 120 requests each,
and writes `loadtest_results.json`.

### Why this matters more than the notebook number

Stage 5 reported 3 ms median for BM25 and 5 ms for dense, measured one query at
a time with nothing else running. That answers *how long does the work take*
and says nothing about *what does a user experience*. Three effects only show
up under concurrency:

- **queueing** — requests arrive faster than they are served; the service is
  still "fast" and users still wait
- **contention** — the embedding model is CPU-bound, so eight concurrent
  requests do not run eight times faster on a four-core box
- **tail latency** — p95 and p99 diverge from p50, and the tail is what a user
  remembers

**The headline number for a CV is not the p50.** It is the concurrency at
which p95 stops being acceptable, because that is the capacity of one worker
process. When throughput plateaus while p95 keeps climbing, you have found it.
More workers (`uvicorn api:app --workers 4`) scale past that; more concurrency
against one worker does not.

---

## 4 · The design decisions to be able to defend

**The index loads once, in `lifespan`.** Per-request loading would add ~2 s
and 70 MB of churn to every call.

**Every handler is `async def`, but the work runs in a thread pool.** This is
the one worth rehearsing. Embedding a query is a matrix multiply — CPU-bound,
not IO. Left on the event loop it blocks *every other request* for its full
duration, and your p95 under load becomes a queue length rather than a
latency. `run_in_threadpool` is what stops that. It is the most common mistake
in FastAPI services and a very likely interview question.

**Request models validate at the boundary.** `k=500` and `method="magic"` are
rejected with a 422 before a handler runs. Without them, `k=500` would reach
the index and the bad method would fall through to a branch that does not
exist.

**The service reports its own percentiles.** A p50 describes the request
nobody complains about.

---

## 5 · Numbers to record

From `/metrics` and `loadtest_results.json`:

| what | why it goes on the CV |
|---|---|
| p50 / p95 / p99 for `/search` at concurrency 1 | the baseline |
| p95 at the concurrency where throughput plateaus | the capacity of one worker |
| peak throughput (rps) | what one process sustains |
| error count at every level | should be zero; if not, that is the finding |

---

## 6 · Known limitations, say them first

Single process, single machine, no authentication, no rate limiting, no
persistence. It is a demonstration of the retrieval stack over HTTP, not a
service anyone should deploy.

Latency is dominated by the embedding model on CPU. A GPU or an ONNX-exported
encoder would change the shape of every number here, and the load test would
need re-running before any of them could be quoted again.

The `/answer` endpoint in generative mode loads flan-t5 lazily on first use, so
the first generative request pays the model load. Fine for a demo, wrong for a
service — a real one would warm it at startup or refuse until it is ready.
