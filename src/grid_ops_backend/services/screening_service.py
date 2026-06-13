from __future__ import annotations

from threading import Lock

from grid_ops_backend.domain.models import ScreeningResult
from grid_ops_backend.engine.grid_engine import GridEngine


class ScreeningService:
    def __init__(self, engine: GridEngine) -> None:
        self._engine = engine
        self._cache = {}
        self._cache_lock = Lock()

    def run(self, run_id: str, network_id: str, seed: int, top_k: int) -> ScreeningResult:
        cache_key = (network_id, seed)
        with self._cache_lock:
            contingencies = self._cache.get(cache_key)
            if contingencies is None:
                contingencies = self._engine.screen_n_minus_one(network_id=network_id, seed=seed)
                self._cache[cache_key] = contingencies
        dangerous = [cont for cont in contingencies if cont.violation_score >= 0.4]
        top = tuple(dangerous[:top_k])
        return ScreeningResult(
            run_id=run_id,
            network_id=network_id,
            total_contingencies=len(contingencies),
            dangerous_count=len(dangerous),
            top_contingencies=top,
        )
