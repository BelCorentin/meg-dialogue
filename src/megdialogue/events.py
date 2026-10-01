"""Session event log: one JSONL line per event, all on one monotonic clock.

Every event carries `t` (seconds since session start, time.perf_counter) and,
when it concerns audio, the sample index into the continuous session recording
(`mic_sample` / `out_sample`). Sample indices are the authoritative timing for
analysis; `t` is for latency accounting.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class EventLog:
    def __init__(self, path: Path | None = None):
        self.t0 = time.perf_counter()
        self.wall0 = time.time()
        self._lock = threading.Lock()
        self._fh = open(path, "a", buffering=1) if path else None
        self.events: list[dict] = []
        self.log("session_start", wall=self.wall0)

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def log(self, kind: str, **fields) -> dict:
        ev = {"t": round(self.now(), 6), "kind": kind, **fields}
        with self._lock:
            self.events.append(ev)
            if self._fh:
                self._fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
        return ev

    def close(self) -> None:
        if self._fh:
            self._fh.close()
