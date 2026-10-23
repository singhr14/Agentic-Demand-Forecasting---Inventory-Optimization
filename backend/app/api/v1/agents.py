from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from backend.app.agents.graph import run_agent_workflow
from backend.app.api.deps import db_session
from backend.app.repositories.base import AgentLogRepository
from backend.app.schemas.api import AgentLogOut, AgentRunOut, AgentRunRequest

router = APIRouter(prefix="/agents", tags=["agents"])


def _to_out(state: dict, logs: list | None = None) -> AgentRunOut:
    log_models = logs or []
    return AgentRunOut(
        run_id=state["run_id"],
        status=state.get("status") or "unknown",
        product_code=state["product_code"],
        review_required=bool(state.get("review_required")),
        forecast_run_id=state.get("forecast_run_id"),
        recommendation=state.get("plan"),
        explanation=state.get("explanation") or "",
        confidence=(state.get("critic") or {}).get("confidence") or "unknown",
        warnings=(state.get("critic") or {}).get("warnings") or [],
        logs=log_models,
    )


def _logs_from_db(db: Session, run_id: str) -> list[AgentLogOut]:
    rows = AgentLogRepository(db).list_for_run(run_id)
    return [
        AgentLogOut(
            agent_name=r.agent_name,
            status=r.status,
            input_summary=r.input_summary,
            output_summary=r.output_summary,
            warnings=json.loads(r.warnings_json or "[]"),
            error_message=r.error_message,
            started_at=r.started_at,
            completed_at=r.completed_at,
        )
        for r in rows
    ]


@router.post("/run", response_model=AgentRunOut)
async def run_agents(payload: AgentRunRequest, db: Session = Depends(db_session)) -> AgentRunOut:
    state = await run_in_threadpool(run_agent_workflow, payload.product_code, payload.horizon, payload.model_name)
    logs = _logs_from_db(db, state["run_id"])
    return _to_out(state, logs)


@router.get("/runs/{run_id}", response_model=AgentRunOut)
def get_run(run_id: str, db: Session = Depends(db_session)) -> AgentRunOut:
    logs = _logs_from_db(db, run_id)
    if not logs:
        raise HTTPException(status_code=404, detail="Agent run not found")
    rec = None
    extra = [r for r in logs if r.agent_name == "persist_results"]
    status = extra[-1].status if extra else logs[-1].status
    product_code = "unknown"
    try:
        first_in = json.loads(logs[0].input_summary or "{}")
        product_code = first_in.get("product_code", product_code)
    except json.JSONDecodeError:
        pass
    plan = None
    explanation = ""
    warnings: list[str] = []
    confidence = "unknown"
    forecast_run_id = None
    for log in logs:
        warnings.extend(log.warnings)
        if log.agent_name == "inventory_planner_agent" and log.output_summary:
            try:
                plan = json.loads(log.output_summary)
            except json.JSONDecodeError:
                plan = {"summary": log.output_summary}
        if log.agent_name == "explanation_agent" and log.output_summary:
            try:
                explanation = json.loads(log.output_summary).get("explanation", log.output_summary)
            except json.JSONDecodeError:
                explanation = log.output_summary
        if log.agent_name == "critic_agent" and log.output_summary:
            try:
                critic = json.loads(log.output_summary)
                confidence = critic.get("confidence", confidence)
            except json.JSONDecodeError:
                pass
        if log.agent_name == "persist_results" and log.output_summary:
            try:
                forecast_run_id = json.loads(log.output_summary).get("forecast_run_id")
            except json.JSONDecodeError:
                pass
    return AgentRunOut(
        run_id=run_id,
        status=status,
        product_code=product_code,
        review_required=status in {"review_required", "flagged", "failed"},
        forecast_run_id=forecast_run_id,
        recommendation=plan,
        explanation=explanation,
        confidence=confidence,
        warnings=warnings,
        logs=logs,
    )
