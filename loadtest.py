"""
Stage 7b — measure the service under concurrent load.

WHY A NOTEBOOK LATENCY IS NOT A LATENCY
---------------------------------------
Stage 5 reported 3 ms median for BM25 and 5 ms for dense. Both were measured
one query at a time, in a loop, with nothing else running. That number answers
"how long does the work take" and says nothing about "what does a user
experience", which is the only question that matters once more than one person
uses the thing.

Under concurrency three effects appear that a loop cannot show:

  QUEUEING       requests arrive faster than they are served and wait. The
                 service is still "fast" and users still wait.
  CONTENTION     the embedding model is CPU-bound. Two concurrent requests do
                 not run twice as fast; on a 4-core box, eight do not run
                 eight times as fast.
  TAIL LATENCY   the p95 and p99 diverge from the p50, and it is the tail a
                 user remembers.

So this sweeps concurrency levels and reports the whole distribution at each.
The headline number for a CV is not the p50 — it is the concurrency at which
p95 stops being acceptable, because that is the capacity of one process.

Run:  uvicorn api:app --port 8000        (in one terminal)
      python loadtest.py                 (in another)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx

QUERIES = [
    "supply chain disruption from a limited number of vendors",
    "cybersecurity breach exposing customer data",
    "rising interest rates increasing debt service costs",
    "loss of key personnel and succession risk",
    "regulatory change in international markets",
    "goodwill impairment following an acquisition",
    "competition from larger and better funded rivals",
    "foreign currency exposure on overseas revenue",
    "reliance on a single manufacturing facility",
    "litigation and regulatory penalties",
    "climate transition and emissions regulation",
    "inability to refinance maturing indebtedness",
]


async def _one(client: httpx.AsyncClient, url: str, payload: dict) -> tuple[float, int]:
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload, timeout=60.0)
        return (time.perf_counter() - t0) * 1000, r.status_code
    except Exception:
        return (time.perf_counter() - t0) * 1000, 0


async def run_level(base: str, route: str, concurrency: int, n: int,
                    payload_for) -> dict:
    """
    Fire `n` requests with at most `concurrency` in flight.

    A semaphore rather than batches: batching would idle the server between
    waves and flatter the tail, which is the opposite of what this measures.
    """
    sem = asyncio.Semaphore(concurrency)
    url = f"{base}{route}"

    async with httpx.AsyncClient() as client:
        async def guarded(i):
            async with sem:
                return await _one(client, url, payload_for(i))

        t0 = time.perf_counter()
        results = await asyncio.gather(*[guarded(i) for i in range(n)])
        wall = time.perf_counter() - t0

    lat = sorted(r[0] for r in results)
    errs = sum(1 for r in results if r[1] != 200)
    pct = lambda p: lat[min(int(len(lat) * p / 100), len(lat) - 1)]
    return {
        "route": route, "concurrency": concurrency, "n": n,
        "errors": errs,
        "throughput_rps": round(n / wall, 1),
        "p50_ms": round(statistics.median(lat), 1),
        "p95_ms": round(pct(95), 1),
        "p99_ms": round(pct(99), 1),
        "max_ms": round(lat[-1], 1),
    }


async def main(base: str, n: int, levels: list[int]) -> None:
    async with httpx.AsyncClient() as c:
        try:
            h = (await c.get(f"{base}/health", timeout=120.0)).json()
        except Exception:
            raise SystemExit(f"no service at {base} — start it with:\n"
                             f"  uvicorn api:app --port 8000")
    print(f"service: {h.get('n_chunks', '?'):,} passages · "
          f"{h.get('n_firms', '?')} firms · device {h.get('device')}\n")

    rows = []
    for route, payload_for in [
        ("/search", lambda i: {"query": QUERIES[i % len(QUERIES)], "k": 5}),
        ("/answer", lambda i: {"query": QUERIES[i % len(QUERIES)], "k": 5}),
    ]:
        for c in levels:
            r = await run_level(base, route, c, n, payload_for)
            rows.append(r)
            print(f"{r['route']:<9} c={r['concurrency']:<3} "
                  f"{r['throughput_rps']:>6.1f} rps   "
                  f"p50 {r['p50_ms']:>7.1f}   p95 {r['p95_ms']:>7.1f}   "
                  f"p99 {r['p99_ms']:>7.1f}   errors {r['errors']}")
        print()

    out = __import__("pathlib").Path("loadtest_results.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"written to {out}")

    print("\nHow to read this:")
    print("  Throughput plateaus once every core is busy — past that point,")
    print("  extra concurrency only grows the queue and the p95, which is why")
    print("  p95 rising while rps stays flat is the capacity limit of one")
    print("  worker. More workers (uvicorn --workers N) scale past it; more")
    print("  concurrency against one worker does not.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=120, help="requests per level")
    ap.add_argument("--levels", default="1,2,4,8,16")
    a = ap.parse_args()
    asyncio.run(main(a.base, a.n, [int(x) for x in a.levels.split(",")]))
