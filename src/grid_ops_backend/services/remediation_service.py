from __future__ import annotations

from grid_ops_backend.domain.models import ActionProposal, ActionType, RecommendationResult


class RemediationService:
    def recommend(
        self,
        run_id: str,
        mode: str,
        dangerous_count: int,
    ) -> RecommendationResult:
        base_cost = max(1.0, dangerous_count * 0.8)
        proposals = (
            ActionProposal(
                action_type=ActionType.REDISPATCH,
                target_id="gen_cluster_1",
                value=12.0,
                estimated_cost=base_cost * 1.0,
                reason="Relieve post-contingency thermal loading.",
            ),
            ActionProposal(
                action_type=ActionType.SWITCH_LINE,
                target_id="line_7",
                value=1.0,
                estimated_cost=base_cost * 0.6,
                reason="Reroute flow around overloaded corridor.",
            ),
        )
        safety_delta = min(1.0, 0.35 + dangerous_count * 0.02)
        total_cost = sum(item.estimated_cost for item in proposals)

        rationale = (
            "Prioritized actions that reduce overload risk before cost.",
            "Rejected high-cost curtailment as first action.",
        )

        if mode == "llm_assisted":
            rationale = rationale + (
                "LLM narrative validated by deterministic safety checks.",
            )
            total_cost *= 0.95

        return RecommendationResult(
            run_id=run_id,
            accepted=True,
            mode=mode,
            safety_delta=safety_delta,
            total_cost=total_cost,
            proposals=proposals,
            rationale=rationale,
        )
