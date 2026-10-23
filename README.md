# Agentic Demand Forecasting & Inventory Optimization

Working FastAPI + Streamlit application that forecasts product demand, scores uncertainty, recommends replenishment, and explains the result through a LangGraph multi-agent workflow.

Forecasting, evaluation, and inventory math are deterministic and **do not require an LLM**. OpenRouter is used only for natural-language explanations when `LLM_API_KEY` is set.

## What the app answers

1. Expected demand for a product over the next seven days
2. How reliable that forecast looks under critic rules
3. Which SKUs look understocked given simulated inventory
4. Recommended reorder quantity, reorder point, and safety stock
5. Assumptions behind the recommendation
6. LightGBM versus a seasonal-naive baseline on a chronological holdout
7. Whether a forecast-driven policy reduces **simulated** inventory cost versus a naive policy

Do not treat simulated inventory improvements as measured business impact unless a backtest was run and the assumptions match your operation. The M5 files do not contain on-hand inventory, supplier lead times, or procurement costs.

## Project layout

- `backend/app` — FastAPI, SQLAlchemy, ML, agents
- `frontend` — Streamlit dashboard (talks to the API only)
- `data/raw` — place official M5 CSVs here
- `data/sample` — generated demonstration data
- `data/processed` — parquet extract and SQLite database
- `artifacts` — LightGBM model and feature pipeline
- `scripts` — database init, sample data, training
- `tests` — reserved for unit tests (not included in this drop)

## Setup (Windows PowerShell)

```powershell
cd "d:\futureness-agentic-ai\Demo_Projects\Agentic Demand Forecasting & Inventory Optimization"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Always run commands from the project root so `PYTHONPATH` imports and SQLite paths resolve correctly:

```powershell
$env:PYTHONPATH = "."
```

## Sample-data mode (default)

No Kaggle download is required.

```powershell
$env:PYTHONPATH = "."
python scripts/generate_sample_data.py
python scripts/init_db.py
python scripts/train_models.py
```

The generator writes M5-shaped CSVs plus `DATA_SOURCE.json` labeled **demonstration data**. API startup will also generate and ingest sample data if files are missing.

## Run the API

```powershell
$env:PYTHONPATH = "."
uvicorn backend.app.main:app --reload --port 8000
```

OpenAPI docs: http://127.0.0.1:8000/docs  
Health: http://127.0.0.1:8000/health

## Run the dashboard

In a second terminal:

```powershell
$env:PYTHONPATH = "."
$env:API_BASE_URL = "http://127.0.0.1:8000"
streamlit run frontend/app.py
```

The UI shows a clear error if the API is not running.

## OpenRouter (optional)

The default provider is OpenRouter. Add to `.env`:

```
LLM_PROVIDER=openrouter
LLM_MODEL=openai/gpt-4o-mini
LLM_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Leave `LLM_API_KEY` empty for fully offline explanations. Never commit `.env`.

## Official M5 dataset

1. Accept the rules and download from [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).
2. Place these files in `data/raw/`:
   - `sales_train_validation.csv`
   - `calendar.csv`
   - `sell_prices.csv`
   - `sales_train_evaluation.csv` (optional, unused by default)
3. In `.env` set:

```
SAMPLE_DATA_MODE=false
M5_STORE_FILTER=CA_1
M5_MAX_ITEMS=50
```

The ingest path converts the wide sales table to long format, joins calendar and prices, forward/back-fills missing prices, and writes `data/processed/daily_sales.parquet`. Full M5 is large; keep the store/item filters for local machines. The app does **not** download Kaggle files automatically.

Then:

```powershell
$env:PYTHONPATH = "."
python scripts/init_db.py
python scripts/train_models.py
```

## Agents

LangGraph order:

`START → Forecast → Critic → Inventory Planner → Explanation → Persist → END`

If the critic marks the forecast invalid, the planner is skipped and the run is stored as `review_required`. Numerical forecasts come from LightGBM or seasonal naive, not from the LLM. Reorder math is deterministic `(s, Q)` logic:

- Lead-time demand = sum of daily forecasts over lead time
- Safety stock = `z * residual_sigma * sqrt(lead_time)`
- Reorder point = lead-time demand + safety stock
- Order quantity = `max(0, ceil(ROP - on_hand - on_order))`

## Evaluation metrics

| Metric | Definition |
| --- | --- |
| MAE | Mean absolute error |
| RMSE | Root mean squared error |
| WAPE | `sum(\|y - yhat\|) / sum(\|y\|)`; if actuals sum to 0, result is 0 when predictions are also 0, otherwise 1 |
| Bias | Mean of `yhat - y` |

Inventory KPIs (stockout rate, fill rate, holding/ordering cost) are **simulated** with configurable lead time, starting stock, and cost rates.

## Tests

The `tests/` folder is reserved. Add pytest modules there later. Tests must use the synthetic generator and must not call Kaggle or OpenRouter.

## Docker

```powershell
copy .env.example .env
docker compose up --build
```

Volumes persist `data/` and `artifacts/`. Set `LLM_API_KEY` in `.env` if you want OpenRouter inside Compose.

## Model limitations

- Recursive multi-step forecasts feed predictions into lag features after the origin; error can compound.
- Prediction intervals use residual standard deviation, not calibrated quantiles.
- Unseen product IDs need a retrain because of label encoding.
- Inventory policy quality is only as good as the assumed lead time, costs, and starting stock.
