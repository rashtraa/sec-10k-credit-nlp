"""
A rate-limited, retrying HTTP client for SEC EDGAR.

One shared token bucket enforces the global request rate across every worker
thread, so raising THREADS does not raise the request rate. That distinction
matters: the SEC blocks on requests/second, not on connection count.
"""
from __future__ import annotations

import gzip
import logging
import threading
import time
from typing import Optional

import requests

from config import (SEC_MAX_RETRIES, SEC_RATE_LIMIT, SEC_TIMEOUT,
                    SEC_USER_AGENT)

log = logging.getLogger(__name__)


class TokenBucket:
    """Global rate limiter. Thread-safe, monotonic-clock based."""

    def __init__(self, rate_per_sec: float):
        self.min_interval = 1.0 / rate_per_sec
        self._lock = threading.Lock()
        self._next_free = time.monotonic()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_free - now
            if wait > 0:
                # Reserve our slot, then sleep outside contention would be
                # nicer, but holding the lock keeps ordering fair and the
                # sleeps here are ~0.125s.
                time.sleep(wait)
                self._next_free += self.min_interval
            else:
                self._next_free = now + self.min_interval


_bucket = TokenBucket(SEC_RATE_LIMIT)

_session_local = threading.local()


def _session() -> requests.Session:
    """One Session per thread — Sessions are not documented thread-safe."""
    s = getattr(_session_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        })
        _session_local.s = s
    return s


def get(url: str, *, binary: bool = False, host: Optional[str] = None):
    """
    Fetch a URL with rate limiting and exponential backoff.

    Returns str (decoded text) or bytes if binary=True. Returns None if the
    resource is genuinely absent (404) — callers treat that as a data gap,
    not an error.
    """
    headers = {"Host": host} if host else {}
    delay = 1.0
    for attempt in range(SEC_MAX_RETRIES):
        _bucket.acquire()
        try:
            r = _session().get(url, headers=headers, timeout=SEC_TIMEOUT)
        except requests.RequestException as e:
            log.warning("network error %s (attempt %d): %s", url, attempt + 1, e)
            time.sleep(delay)
            delay *= 2
            continue

        if r.status_code == 200:
            return r.content if binary else r.text
        if r.status_code == 404:
            return None
        if r.status_code in (403, 429) or r.status_code >= 500:
            # 403 from SEC usually means rate/UA trouble; back off hard.
            log.warning("HTTP %d on %s — backing off %.1fs", r.status_code, url, delay)
            time.sleep(delay)
            delay *= 2
            continue
        log.warning("HTTP %d on %s — not retrying", r.status_code, url)
        return None

    log.error("giving up on %s after %d attempts", url, SEC_MAX_RETRIES)
    return None


def write_gz(path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(text)


def read_gz(path) -> str:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        return f.read()
