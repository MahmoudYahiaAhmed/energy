from grid_ops_backend.engine.grid_engine import GridEngine
from grid_ops_backend.services.comparison_service import ComparisonService
from grid_ops_backend.services.remediation_service import RemediationService
from grid_ops_backend.services.screening_service import ScreeningService


def test_screening_is_deterministic() -> None:
    service = ScreeningService(GridEngine())
    first = service.run(run_id="r1", network_id="ieee14", seed=42, top_k=5)
    second = service.run(run_id="r1", network_id="ieee14", seed=42, top_k=5)

    first_ids = [item.contingency_id for item in first.top_contingencies]
    second_ids = [item.contingency_id for item in second.top_contingencies]

    assert first_ids == second_ids
    assert first.dangerous_count == second.dangerous_count


def test_llm_assisted_recommendation_has_lower_cost() -> None:
    remediation = RemediationService()
    baseline = remediation.recommend("r1", "baseline", dangerous_count=6)
    llm = remediation.recommend("r1", "llm_assisted", dangerous_count=6)

    assert llm.total_cost < baseline.total_cost


def test_comparison_returns_valid_winner() -> None:
    remediation = RemediationService()
    comparison = ComparisonService()

    baseline = remediation.recommend("r1", "baseline", dangerous_count=5)
    llm = remediation.recommend("r1", "llm_assisted", dangerous_count=5)
    result = comparison.compare("r1", baseline, llm)

    assert result.winner in {"baseline", "llm_assisted"}
