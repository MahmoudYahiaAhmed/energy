from __future__ import annotations

import asyncio
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from grid_ops_backend.api.schemas import (
    ApiEnvelope,
    RecommendationRequest,
    RunCreateRequest,
    ScreeningRequest,
)
from grid_ops_backend.config import load_settings
from grid_ops_backend.domain.run_state import RunState
from grid_ops_backend.engine.grid_engine import GridEngine
from grid_ops_backend.services.comparison_service import ComparisonService
from grid_ops_backend.services.gridsfm_service import (
    GRIDSFM_CASES,
    PANDAPOWER_AND_GRIDSFM,
    GridSFMService,
)
from grid_ops_backend.services.remediation_service import RemediationService
from grid_ops_backend.services.screening_service import ScreeningService
from grid_ops_backend.storage.memory_repo import InMemoryRunRepository

# Pandapower-only built-in cases (small IEEE standard networks)
_PANDAPOWER_ONLY_CASES: tuple[dict, ...] = (
    {"id": "ieee14", "label": "IEEE 14-bus", "engines": ["pandapower"]},
    {"id": "ieee39", "label": "IEEE 39-bus (NE)", "engines": ["pandapower"]},
    {"id": "ieee57", "label": "IEEE 57-bus", "engines": ["pandapower"]},
    {"id": "ieee118", "label": "IEEE 118-bus", "engines": ["pandapower"]},
    {"id": "case300", "label": "IEEE 300-bus", "engines": ["pandapower"]},
)


def _build_case_list() -> list[dict]:
    cases = list(_PANDAPOWER_ONLY_CASES)
    for case_id in GRIDSFM_CASES:
        engines = ["gridsfm"]
        if case_id in PANDAPOWER_AND_GRIDSFM:
            engines = ["pandapower", "gridsfm"]
        label = case_id.replace("_", " ").replace("case", "Case ")
        cases.append({"id": case_id, "label": label.strip(), "engines": engines})
    return cases


_KNOWN_CASES = _build_case_list()


def _latest_screening_counts(run: RunState) -> tuple[int, int] | None:
    for event in reversed(run.events):
        if event.get("type") != "screening_completed":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        dangerous_count = payload.get("dangerous_count")
        total = payload.get("total")
        if isinstance(dangerous_count, int) and isinstance(total, int):
            return dangerous_count, total
    return None


def create_app() -> FastAPI:
    settings = load_settings()
    app = FastAPI(title="Grid Ops Backend", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allow_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    engine = GridEngine()
    screening_service = ScreeningService(engine)
    remediation_service = RemediationService()
    comparison_service = ComparisonService()
    repo = InMemoryRunRepository()
    gridsfm_service = GridSFMService(
        samples_dir=settings.gridsfm_samples_dir,
    )

    @app.on_event("startup")
    @app.get("/health", response_model=ApiEnvelope)
    def health() -> ApiEnvelope:
        return ApiEnvelope(
            success=True,
            data={
                "status": "ok",
                "pandapower_available": engine.pandapower_available,
                    "gridsfm_available": gridsfm_service.available,
                    "gridsfm_case_count": len(gridsfm_service.list_available_cases()),
            },
        )

    @app.get("/api/v1/cases", response_model=ApiEnvelope)
    def list_cases() -> ApiEnvelope:
        return ApiEnvelope(success=True, data={"cases": _KNOWN_CASES})

    @app.post("/api/v1/runs", response_model=ApiEnvelope)
    def create_run(payload: RunCreateRequest) -> ApiEnvelope:
        run_id = str(uuid4())
        run = RunState(run_id=run_id, network_id=payload.network_id, seed=payload.seed)
        run = run.with_event("run_created", {"run_id": run_id, "network_id": payload.network_id})
        repo.create(run)
        return ApiEnvelope(success=True, data={"run_id": run_id})

    @app.get("/api/v1/runs/{run_id}", response_model=ApiEnvelope)
    def get_run(run_id: str) -> ApiEnvelope:
        run = repo.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return ApiEnvelope(
            success=True,
            data={
                "run_id": run.run_id,
                "network_id": run.network_id,
                "seed": run.seed,
                "event_count": len(run.events),
            },
        )

    @app.post("/api/v1/runs/{run_id}/screen", response_model=ApiEnvelope)
    def screen_run(run_id: str, payload: ScreeningRequest) -> ApiEnvelope:
        run = repo.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")

        result = screening_service.run(
            run_id=run_id,
            network_id=run.network_id,
            seed=run.seed,
            top_k=payload.top_k,
        )
        run = run.with_event(
            "screening_completed",
            {"dangerous_count": result.dangerous_count, "total": result.total_contingencies},
        )
        repo.update(run)

        return ApiEnvelope(
            success=True,
            data={
                "run_id": result.run_id,
                "network_id": result.network_id,
                "total_contingencies": result.total_contingencies,
                "dangerous_count": result.dangerous_count,
                "top_contingencies": [item.__dict__ for item in result.top_contingencies],
            },
        )

    @app.post("/api/v1/runs/{run_id}/recommend", response_model=ApiEnvelope)
    def recommend_run(run_id: str, payload: RecommendationRequest) -> ApiEnvelope:
        run = repo.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")

        cached_screening = _latest_screening_counts(run)
        if cached_screening is None:
            screening = screening_service.run(
                run_id=run_id, network_id=run.network_id, seed=run.seed, top_k=5
            )
            dangerous_count = screening.dangerous_count
            run = run.with_event(
                "screening_completed",
                {"dangerous_count": screening.dangerous_count, "total": screening.total_contingencies},
            )
        else:
            dangerous_count, _ = cached_screening
        recommendation = remediation_service.recommend(
            run_id=run_id, mode=payload.mode.value, dangerous_count=dangerous_count
        )
        run = run.with_event(
            "recommendation_generated",
            {"mode": recommendation.mode, "accepted": recommendation.accepted},
        )
        repo.update(run)

        return ApiEnvelope(
            success=True,
            data={
                "run_id": recommendation.run_id,
                "mode": recommendation.mode,
                "accepted": recommendation.accepted,
                "safety_delta": recommendation.safety_delta,
                "total_cost": recommendation.total_cost,
                "proposals": [item.__dict__ for item in recommendation.proposals],
                "rationale": list(recommendation.rationale),
            },
        )

    @app.post("/api/v1/runs/{run_id}/compare", response_model=ApiEnvelope)
    def compare_modes(run_id: str) -> ApiEnvelope:
        run = repo.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")

        cached_screening = _latest_screening_counts(run)
        if cached_screening is None:
            screening = screening_service.run(
                run_id=run_id, network_id=run.network_id, seed=run.seed, top_k=5
            )
            dangerous_count = screening.dangerous_count
            run = run.with_event(
                "screening_completed",
                {"dangerous_count": screening.dangerous_count, "total": screening.total_contingencies},
            )
        else:
            dangerous_count, _ = cached_screening
        baseline = remediation_service.recommend(
            run_id=run_id, mode="baseline", dangerous_count=dangerous_count
        )
        llm_assisted = remediation_service.recommend(
            run_id=run_id, mode="llm_assisted", dangerous_count=dangerous_count
        )
        comparison = comparison_service.compare(run_id, baseline, llm_assisted)
        run = run.with_event("comparison_completed", {"winner": comparison.winner})
        repo.update(run)

        return ApiEnvelope(success=True, data=comparison.__dict__)

    @app.get("/api/v1/runs/{run_id}/gridsfm", response_model=ApiEnvelope)
    def get_gridsfm(run_id: str) -> ApiEnvelope:
        run = repo.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")

        if not gridsfm_service.available:
            return ApiEnvelope(
                success=False,
                error="GridSFM not configured (set GRIDSFM_MODEL_DIR in .env)",
            )

        result = gridsfm_service.get_case(run.network_id)
        if result is None:
            return ApiEnvelope(
                success=False,
                    error=f"No GridSFM sample found for '{run.network_id}'.",
            )

        return ApiEnvelope(
            success=True,
            data={
                "case_name": result.case_name,
                    "bus_count": result.bus_count,
                    "gen_count": result.gen_count,
                    "line_count": result.line_count,
                    "has_solution": result.has_solution,
                "feasible": result.feasible,
                    "termination_status": result.termination_status,
                    "sample_path": result.sample_path,
            },
        )

    @app.get("/api/v1/runs/{run_id}/events")
    async def stream_run_events(run_id: str) -> StreamingResponse:
        run = repo.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")

        events_snapshot = list(run.events)

        async def event_generator() -> object:
            for event in events_snapshot:
                yield f"event: {event['type']}\ndata: {event['payload']}\n\n"
                await asyncio.sleep(0.01)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app
