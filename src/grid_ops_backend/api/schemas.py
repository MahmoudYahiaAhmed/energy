from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RunCreateRequest(BaseModel):
    network_id: str = Field(default="ieee14")
    seed: int = Field(default=42)


class ScreeningRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)


class RecommendationMode(str, Enum):
    BASELINE = "baseline"
    LLM_ASSISTED = "llm_assisted"


class RecommendationRequest(BaseModel):
    mode: RecommendationMode = Field(default=RecommendationMode.BASELINE)


class ContingencyDTO(BaseModel):
    contingency_id: str
    component_type: str
    component_id: str
    violation_score: float
    severity: str


class ActionProposalDTO(BaseModel):
    action_type: str
    target_id: str
    value: float
    estimated_cost: float
    reason: str


class ApiEnvelope(BaseModel):
    success: bool
    data: dict[str, object] | None = None
    error: str | None = None
