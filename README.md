# Grid Ops Backend

Backend-first implementation for the Grid Operation Agents challenge.

This project exposes a frontend-ready API for:
- N-1 contingency screening
- deterministic remediation planning
- baseline vs LLM-assisted comparison metrics
- run progress streaming via Server-Sent Events (SSE)

The local Streamlit prototype can run either DC power flow (`pandapower.rundcpp`) or AC power flow
(`pandapower.runpp`). DC mode is faster and checks active-power thermal loading only. AC mode is
slower but also checks voltage magnitudes and enables voltage/reactive corrective actions. The app
includes IEEE 9, 14, 30, 57, 118, and 300-bus pandapower examples.

## Quick start

```bash
pip install -e .[dev]
uvicorn grid_ops_backend.api.app:create_app --factory --reload
```

Optional pandapower support:

```bash
pip install -e .[grid]
```

## API overview

- `GET /health`
- `POST /api/v1/runs`
- `GET /api/v1/runs/{run_id}`
- `GET /api/v1/runs/{run_id}/events` (SSE)
- `POST /api/v1/runs/{run_id}/screen`
- `POST /api/v1/runs/{run_id}/recommend`
- `POST /api/v1/runs/{run_id}/compare`

## Frontend integration notes

- Stable JSON envelopes for success and error responses.
- Every run has deterministic behavior using the provided seed.
- SSE events include run lifecycle and recommendation status.
