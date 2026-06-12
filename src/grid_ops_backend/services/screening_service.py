from __future__ import annotations

from grid_ops_backend.domain.models import ScreeningResult
from grid_ops_backend.engine.grid_engine import GridEngine


class ScreeningService:
    def __init__(self, engine: GridEngine) -> None:
        self._engine = engine

    def run(self, run_id: str, network_id: str, seed: int, top_k: int) -> ScreeningResult:
        contingencies = self._engine.screen_n_minus_one(network_id=network_id, seed=seed)
        dangerous = [cont for cont in contingencies if cont.violation_score >= 0.4]
        top = tuple(dangerous[:top_k])
        return ScreeningResult(
            run_id=run_id,
            network_id=network_id,
            total_contingencies=len(contingencies),
            dangerous_count=len(dangerous),
            top_contingencies=top,
        )
