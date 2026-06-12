from __future__ import annotations

from threading import Lock

from grid_ops_backend.domain.run_state import RunState


class InMemoryRunRepository:
    def __init__(self) -> None:
        self._lock = Lock()
        self._runs: dict[str, RunState] = {}

    def create(self, run: RunState) -> RunState:
        with self._lock:
            self._runs[run.run_id] = run
            return run

    def update(self, run: RunState) -> RunState:
        with self._lock:
            self._runs[run.run_id] = run
            return run

    def get(self, run_id: str) -> RunState | None:
        with self._lock:
            return self._runs.get(run_id)
