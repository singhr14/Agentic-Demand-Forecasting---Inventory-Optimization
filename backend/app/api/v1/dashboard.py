from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.api.deps import db_session
from backend.app.models.entities import AgentExecutionLog, EvaluationRun, ForecastRun, InventoryState, Product

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary")
def summary(db: Session = Depends(db_session)) -> dict:
    product_count = db.scalar(select(func.count()).select_from(Product)) or 0
    forecast_count = db.scalar(select(func.count()).select_from(ForecastRun)) or 0
    low_stock = db.scalar(select(func.count()).select_from(InventoryState).where(InventoryState.current_stock <= 10)) or 0
    latest_eval = db.scalar(select(EvaluationRun).order_by(EvaluationRun.created_at.desc()))
    recent_agents = list(
        db.scalars(select(AgentExecutionLog).order_by(AgentExecutionLog.started_at.desc()).limit(10)).all()
    )
    return {
        "product_count": product_count,
        "forecast_count": forecast_count,
        "low_stock_count": low_stock,
        "latest_evaluation": {
            "available": latest_eval is not None,
            "model_name": getattr(latest_eval, "model_name", None),
            "metrics": json.loads(latest_eval.metrics_json) if latest_eval else None,
        },
        "recent_agent_events": [
            {"run_id": r.run_id, "agent_name": r.agent_name, "status": r.status, "started_at": r.started_at.isoformat()}
            for r in recent_agents
        ],
    }
