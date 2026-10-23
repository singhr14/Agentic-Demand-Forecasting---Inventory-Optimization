from __future__ import annotations

import json
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from backend.app.api.deps import db_session
from backend.app.ml.data_store import data_store
from backend.app.ml.evaluation import compare_inventory_policies, evaluate_point_forecasts, rolling_backtest
from backend.app.repositories.base import EvaluationRepository
from backend.app.schemas.api import EvaluationRequest
from backend.app.services.forecasting import ensure_models

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


def _run_eval(db: Session, horizon: int) -> dict:
    df = data_store.load()
    lgbm = ensure_models()
    accuracy = evaluate_point_forecasts(df, lgbm, horizon=horizon)
    backtest = rolling_backtest(df, lgbm, horizon=horizon)
    inventory = compare_inventory_policies(df, lgbm, horizon=horizon)
    payload = {
        "accuracy": accuracy,
        "backtest": backtest,
        "inventory_simulation": inventory,
        "limitations": [
            "Holdout metrics are measured on chronological test dates using the same products for baseline and LightGBM.",
            "Uncertainty estimates are residual-based and not fully calibrated prediction intervals.",
            "Inventory outcomes are simulated with assumed lead time, costs, and starting stock. M5 does not contain these fields.",
        ],
    }
    repo = EvaluationRepository(db)
    start = date.fromisoformat(accuracy["test_start"])
    end = date.fromisoformat(accuracy["test_end"])
    repo.add("lightgbm", start, end, payload)
    repo.add("seasonal_naive", start, end, payload)
    return payload


@router.post("/run")
async def run_evaluation(payload: EvaluationRequest, db: Session = Depends(db_session)) -> dict:
    try:
        return await run_in_threadpool(_run_eval, db, payload.horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/latest")
def latest_evaluation(db: Session = Depends(db_session)) -> dict:
    rows = EvaluationRepository(db).latest()
    if not rows:
        return {"available": False, "message": "No evaluation has been run yet."}
    latest = rows[0]
    return {
        "available": True,
        "id": latest.id,
        "model_name": latest.model_name,
        "test_start_date": str(latest.test_start_date),
        "test_end_date": str(latest.test_end_date),
        "created_at": latest.created_at.isoformat(),
        "metrics": json.loads(latest.metrics_json),
    }
