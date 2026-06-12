from __future__ import annotations

from grid_ops_backend.domain.models import ComparisonResult, RecommendationResult


class ComparisonService:
    def compare(
        self, run_id: str, baseline: RecommendationResult, llm_assisted: RecommendationResult
    ) -> ComparisonResult:
        baseline_score = baseline.safety_delta * 100.0 - baseline.total_cost
        llm_score = llm_assisted.safety_delta * 100.0 - llm_assisted.total_cost
        winner = "llm_assisted" if llm_score >= baseline_score else "baseline"
        return ComparisonResult(
            run_id=run_id,
            baseline_score=baseline_score,
            llm_score=llm_score,
            baseline_cost=baseline.total_cost,
            llm_cost=llm_assisted.total_cost,
            winner=winner,
        )
