# Grid Ops Backend

Backend-first implementation for the Grid Operation Agents challenge.

This project includes:
- a FastAPI backend for run creation, N-1 screening, remediation, comparison, and SSE progress events
- a Streamlit prototype for interactive pandapower contingency studies
- a CLI optimizer for post-contingency corrective action experiments
- an optional Vite/TanStack frontend in `frontend/`

## Requirements

- Python 3.11+
- Node.js 20+ for the optional frontend

## Python Setup

From the repository root:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Run the FastAPI Backend

```bash
uvicorn grid_ops_backend.api.app:create_app --factory --reload
```

The API runs at `http://127.0.0.1:8000`.

Useful endpoints:
- `GET /health`
- `GET /api/v1/cases`
- `POST /api/v1/runs`
- `GET /api/v1/runs/{run_id}`
- `GET /api/v1/runs/{run_id}/events`
- `POST /api/v1/runs/{run_id}/screen`
- `POST /api/v1/runs/{run_id}/recommend`
- `POST /api/v1/runs/{run_id}/compare`

## Run the Streamlit Prototype

```bash
streamlit run app.py
```

The Streamlit app lets you choose a sample grid, run AC or DC power flow, apply an N-1 contingency, and inspect corrective actions.

## Run the CLI Optimizer

```bash
python run_optimizer.py --network case118 --contingency-index 0 --mode ac
```

Write the JSON report to a file:

```bash
python run_optimizer.py --network case118 --contingency-index 0 --mode ac --output-report report.json
```

Useful options:
- `--mode ac` or `--mode dc`
- `--max-greedy-steps 10`
- `--allow-line-switching`
- `--no-load-curtailment`
- `--use-gridsfm`
- `--gridsfm-checkpoint <path>`

## Run Tests

```bash
pytest
```

## Optional Frontend

Install and run the frontend separately:

```bash
cd frontend
npm install
npm run dev
```

The frontend dev server prints its local URL in the terminal. Keep the FastAPI backend running in another terminal for API-backed views.

## Optional Environment Variables

Create a `.env` file in the repository root if you need local configuration:

```bash
CORS_ALLOW_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
GRIDSFM_MODEL_DIR=C:\path\to\gridsfm
```

`GRIDSFM_MODEL_DIR` is only needed for GridSFM-backed sample loading or ranking. The pandapower workflows run without it.

## Notes

The local prototype can run either DC power flow (`pandapower.rundcpp`) or AC power flow (`pandapower.runpp`). DC mode is faster and checks active-power thermal loading only. AC mode is slower but also checks voltage magnitudes and enables voltage/reactive corrective actions.

This is an offline research and education prototype, not a production grid-control system.
