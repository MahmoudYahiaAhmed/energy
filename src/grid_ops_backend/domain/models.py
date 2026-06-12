from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionType(str, Enum):
    SWITCH_LINE = "switch_line"
    REDISPATCH = "redispatch"
    CURTAIL = "curtail"


@dataclass(frozen=True)
class Contingency:
    contingency_id: str
    component_type: str
    component_id: str
    violation_score: float
    severity: Severity


@dataclass(frozen=True)
class ActionProposal:
    action_type: ActionType
    target_id: str
    value: float
    estimated_cost: float
    reason: str


@dataclass(frozen=True)
class ScreeningResult:
    run_id: str
    network_id: str
    total_contingencies: int
    dangerous_count: int
    top_contingencies: tuple[Contingency, ...]


@dataclass(frozen=True)
class RecommendationResult:
    run_id: str
    accepted: bool
    mode: str
    safety_delta: float
    total_cost: float
    proposals: tuple[ActionProposal, ...]
    rationale: tuple[str, ...]


@dataclass(frozen=True)
class ComparisonResult:
    run_id: str
    baseline_score: float
    llm_score: float
    baseline_cost: float
    llm_cost: float
    winner: str
