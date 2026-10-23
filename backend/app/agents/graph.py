from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from backend.app.agents.critic import critique_forecast
from backend.app.agents.explain import explain
from backend.app.core.config import get_settings
from backend.app.database import SessionLocal
from backend.app.ml.inventory import InventoryInputs, plan_inventory, plan_to_dict
from backend.app.repositories.base import AgentLogRepository, InventoryRepository
from backend.app.services.forecasting import generate_forecast_frame, persist_forecast, get_product_or_sync

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    run_id: str
    product_code: str
    horizon: int
    model_name: str
    forecast_rows: list[dict]
    forecast_meta: dict
    critic: dict
    plan: dict
    explanation: str
    forecast_run_id: int | None
    recommendation_id: int | None
    status: str
    logs: list[dict]
    review_required: bool
    error: str | None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return str(value)[:10] if not isinstance(value, datetime) else value.isoformat()
    if isinstance(value, (pd.Timedelta,)):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    return value


def _append_log(state: AgentState, name: str, status: str, started: datetime, payload_in: dict, payload_out: dict, warnings: list[str], error: str | None = None) -> None:
    logs = list(state.get("logs") or [])
    logs.append(
        {
            "agent_name": name,
            "status": status,
            "started_at": started,
            "completed_at": datetime.utcnow(),
            "input_summary": json.dumps(_json_safe(payload_in))[:4000],
            "output_summary": json.dumps(_json_safe(payload_out))[:4000],
            "warnings": warnings,
            "error_message": error,
        }
    )
    state["logs"] = logs


def forecast_agent(state: AgentState) -> AgentState:
    started = datetime.utcnow()
    product_code = state["product_code"]
    horizon = int(state.get("horizon") or 7)
    model_name = state.get("model_name") or "lightgbm"
    try:
        preds, _hist, meta = generate_forecast_frame(product_code, horizon, model_name)
        rows = _json_safe(preds.to_dict(orient="records"))
        state["forecast_rows"] = rows
        state["forecast_meta"] = _json_safe(meta)
        state["status"] = "forecasted"
        _append_log(
            state,
            "forecast_agent",
            "success",
            started,
            {"product_code": product_code, "horizon": horizon, "model_name": model_name},
            {"n_points": len(rows), "model": meta.get("model_name"), "version": meta.get("model_version")},
            meta.get("warnings") or [],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Forecast agent failed")
        state["error"] = str(exc)
        state["status"] = "failed"
        state["forecast_rows"] = []
        state["forecast_meta"] = {}
        state["review_required"] = True
        _append_log(state, "forecast_agent", "error", started, {"product_code": product_code}, {}, [], str(exc))
    return state


def critic_agent(state: AgentState) -> AgentState:
    started = datetime.utcnow()
    if state.get("status") == "failed":
        state["critic"] = {
            "valid": False,
            "confidence": "review_required",
            "review_required": True,
            "warnings": [state.get("error") or "Forecast failed"],
        }
        _append_log(state, "critic_agent", "skipped", started, {}, state["critic"], state["critic"]["warnings"])
        return state
    meta = state.get("forecast_meta") or {}
    preds = [float(r["predicted_demand"]) for r in state.get("forecast_rows") or []]
    result = critique_forecast(
        predictions=preds,
        history_mean=float(meta.get("history_mean") or 0),
        history_std=float(meta.get("history_std") or 0),
        history_days=int(meta.get("history_days") or 0),
        min_history_days=get_settings().MIN_HISTORY_DAYS,
    )
    critic = {
        "valid": result.valid,
        "confidence": result.confidence,
        "review_required": result.review_required,
        "warnings": result.warnings,
    }
    state["critic"] = critic
    state["review_required"] = result.review_required
    status = "success" if result.valid else "flagged"
    _append_log(state, "critic_agent", status, started, {"n_preds": len(preds)}, critic, result.warnings)
    return state


def route_after_critic(state: AgentState) -> Literal["planner", "explainer"]:
    critic = state.get("critic") or {}
    if critic.get("valid"):
        return "planner"
    return "explainer"


def planner_agent(state: AgentState) -> AgentState:
    started = datetime.utcnow()
    db = SessionLocal()
    try:
        product_code = state["product_code"]
        inv_repo = InventoryRepository(db)
        product = get_product_or_sync(db, product_code)
        inv = inv_repo.get(product_code)
        if inv is None:
            inv_repo.upsert(
                product,
                {
                    "current_stock": 40.0,
                    "on_order_quantity": 0.0,
                    "lead_time_days": 3,
                    "holding_cost_per_unit": 0.15,
                    "ordering_cost": 25.0,
                    "service_level_target": 0.95,
                },
            )
            inv = inv_repo.get(product_code)
        assert inv is not None
        forecasts = [float(r["predicted_demand"]) for r in state.get("forecast_rows") or []]
        meta = state.get("forecast_meta") or {}
        plan = plan_inventory(
            InventoryInputs(
                current_stock=inv.current_stock,
                on_order_quantity=inv.on_order_quantity,
                lead_time_days=inv.lead_time_days,
                holding_cost_per_unit=inv.holding_cost_per_unit,
                ordering_cost=inv.ordering_cost,
                service_level_target=inv.service_level_target,
                daily_forecast=forecasts,
                residual_std=float(meta.get("residual_std") or meta.get("history_std") or 1.0),
            )
        )
        payload = plan_to_dict(plan)
        payload["current_stock"] = inv.current_stock
        payload["on_order_quantity"] = inv.on_order_quantity
        payload["lead_time_days"] = inv.lead_time_days
        state["plan"] = payload
        _append_log(
            state,
            "inventory_planner_agent",
            "success",
            started,
            {"product_code": product_code, "stock": inv.current_stock, "on_order": inv.on_order_quantity},
            {
                "order_qty": payload["recommended_order_quantity"],
                "rop": payload["reorder_point"],
                "safety_stock": payload["safety_stock"],
            },
            [],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Planner failed")
        state["error"] = str(exc)
        state["review_required"] = True
        _append_log(state, "inventory_planner_agent", "error", started, {}, {}, [], str(exc))
    finally:
        db.close()
    return state


def explainer_agent(state: AgentState) -> AgentState:
    started = datetime.utcnow()
    evidence = {
        "product_code": state.get("product_code"),
        "forecast": [r.get("predicted_demand") for r in state.get("forecast_rows") or []],
        "forecast_dates": [r.get("forecast_date") for r in state.get("forecast_rows") or []],
        "model": (state.get("forecast_meta") or {}).get("model_name"),
        "model_version": (state.get("forecast_meta") or {}).get("model_version"),
        "history_mean": (state.get("forecast_meta") or {}).get("history_mean"),
        "history_days": (state.get("forecast_meta") or {}).get("history_days"),
        "recent_demand": (state.get("forecast_meta") or {}).get("recent_demand"),
        "critic": state.get("critic"),
        "plan": state.get("plan"),
        "synthetic": (state.get("forecast_meta") or {}).get("synthetic"),
        "review_required": state.get("review_required"),
    }
    text = explain(_json_safe(evidence))
    state["explanation"] = text
    _append_log(state, "explanation_agent", "success", started, {"evidence_keys": list(evidence)}, {"explanation": text[:500]}, [])
    return state


def persist_results(state: AgentState) -> AgentState:
    started = datetime.utcnow()
    db: Session = SessionLocal()
    try:
        product_code = state["product_code"]
        rows = state.get("forecast_rows") or []
        meta = state.get("forecast_meta") or {"model_name": "unknown", "model_version": "n/a"}
        status = "review_required" if state.get("review_required") else "completed"
        if state.get("status") == "failed":
            status = "failed"
        forecast_run_id = None
        if rows:
            preds = pd.DataFrame(rows)
            run, _product = persist_forecast(db, product_code, preds, meta, status=status)
            forecast_run_id = run.id
            state["forecast_run_id"] = forecast_run_id
            plan = state.get("plan")
            if plan and forecast_run_id:
                rec = InventoryRepository(db).save_recommendation(forecast_run_id, plan)
                state["recommendation_id"] = rec.id
        log_repo = AgentLogRepository(db)
        for log in state.get("logs") or []:
            log_repo.add(
                run_id=state["run_id"],
                agent_name=log["agent_name"],
                status=log["status"],
                started_at=log["started_at"],
                completed_at=log["completed_at"],
                input_summary=log["input_summary"],
                output_summary=log["output_summary"],
                warnings=log.get("warnings") or [],
                error_message=log.get("error_message"),
            )
        log_repo.add(
            run_id=state["run_id"],
            agent_name="persist_results",
            status="success",
            started_at=started,
            completed_at=datetime.utcnow(),
            input_summary=json.dumps({"forecast_points": len(rows)}),
            output_summary=json.dumps({"forecast_run_id": forecast_run_id, "status": status}),
            warnings=[],
        )
        state["status"] = status
    except Exception as exc:  # noqa: BLE001
        logger.exception("Persist failed")
        db.rollback()
        state["error"] = str(exc)
        state["status"] = "failed"
    finally:
        db.close()
    return state


def build_workflow():
    graph = StateGraph(AgentState)
    graph.add_node("forecast_node", forecast_agent)
    graph.add_node("critic_node", critic_agent)
    graph.add_node("planner_node", planner_agent)
    graph.add_node("explainer_node", explainer_agent)
    graph.add_node("persist_node", persist_results)
    graph.add_edge(START, "forecast_node")
    graph.add_edge("forecast_node", "critic_node")
    graph.add_conditional_edges(
        "critic_node",
        route_after_critic,
        {"planner": "planner_node", "explainer": "explainer_node"},
    )
    graph.add_edge("planner_node", "explainer_node")
    graph.add_edge("explainer_node", "persist_node")
    graph.add_edge("persist_node", END)
    return graph.compile()


_workflow = None


def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow


def run_agent_workflow(product_code: str, horizon: int = 7, model_name: str = "lightgbm") -> AgentState:
    workflow = get_workflow()
    initial: AgentState = {
        "run_id": uuid.uuid4().hex,
        "product_code": product_code,
        "horizon": horizon,
        "model_name": model_name,
        "logs": [],
        "review_required": False,
        "status": "started",
    }
    return workflow.invoke(initial)
