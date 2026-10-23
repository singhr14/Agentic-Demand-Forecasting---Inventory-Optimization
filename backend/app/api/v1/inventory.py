from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from backend.app.api.deps import db_session
from backend.app.ml.inventory import InventoryInputs, plan_inventory, plan_to_dict
from backend.app.repositories.base import InventoryRepository
from backend.app.schemas.api import InventoryStateIn, InventoryStateOut, RecommendationOut, RecommendationRequest
from backend.app.services.forecasting import generate_forecast_frame, get_product_or_sync, persist_forecast

router = APIRouter(prefix="/inventory", tags=["inventory"])


@router.get("/{product_code}", response_model=InventoryStateOut)
def get_inventory(product_code: str, db: Session = Depends(db_session)) -> InventoryStateOut:
    product = get_product_or_sync(db, product_code)
    state = InventoryRepository(db).get(product.product_code)
    if state is None:
        raise HTTPException(status_code=404, detail="Inventory state not found")
    return InventoryStateOut(
        product_code=state.product_code,
        current_stock=state.current_stock,
        on_order_quantity=state.on_order_quantity,
        lead_time_days=state.lead_time_days,
        holding_cost_per_unit=state.holding_cost_per_unit,
        ordering_cost=state.ordering_cost,
        service_level_target=state.service_level_target,
        updated_at=state.updated_at,
    )


@router.put("/{product_code}", response_model=InventoryStateOut)
def update_inventory(product_code: str, payload: InventoryStateIn, db: Session = Depends(db_session)) -> InventoryStateOut:
    try:
        product = get_product_or_sync(db, product_code)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    state = InventoryRepository(db).upsert(product, payload.model_dump())
    return InventoryStateOut(
        product_code=state.product_code,
        current_stock=state.current_stock,
        on_order_quantity=state.on_order_quantity,
        lead_time_days=state.lead_time_days,
        holding_cost_per_unit=state.holding_cost_per_unit,
        ordering_cost=state.ordering_cost,
        service_level_target=state.service_level_target,
        updated_at=state.updated_at,
    )


def _recommend(db: Session, payload: RecommendationRequest) -> RecommendationOut:
    preds, _hist, meta = generate_forecast_frame(payload.product_code, payload.horizon, payload.model_name)
    product = get_product_or_sync(db, payload.product_code)
    repo = InventoryRepository(db)
    state = repo.get(payload.product_code)
    values = {
        "current_stock": payload.current_stock,
        "on_order_quantity": payload.on_order_quantity,
        "lead_time_days": payload.lead_time_days,
        "holding_cost_per_unit": payload.holding_cost_per_unit,
        "ordering_cost": payload.ordering_cost,
        "service_level_target": payload.service_level_target,
    }
    merged = {}
    for key, override in values.items():
        if override is not None:
            merged[key] = override
        elif state is not None:
            merged[key] = getattr(state, key)
        else:
            merged[key] = {
                "current_stock": 40.0,
                "on_order_quantity": 0.0,
                "lead_time_days": 3,
                "holding_cost_per_unit": 0.15,
                "ordering_cost": 25.0,
                "service_level_target": 0.95,
            }[key]
    if state is None or any(v is not None for v in values.values()):
        state = repo.upsert(product, merged)
    plan = plan_inventory(
        InventoryInputs(
            current_stock=merged["current_stock"],
            on_order_quantity=merged["on_order_quantity"],
            lead_time_days=merged["lead_time_days"],
            holding_cost_per_unit=merged["holding_cost_per_unit"],
            ordering_cost=merged["ordering_cost"],
            service_level_target=merged["service_level_target"],
            daily_forecast=preds["predicted_demand"].astype(float).tolist(),
            residual_std=float(meta.get("residual_std") or 1.0),
        )
    )
    run, _ = persist_forecast(db, payload.product_code, preds, meta)
    rec_payload = plan_to_dict(plan)
    repo.save_recommendation(run.id, rec_payload)
    return RecommendationOut(
        product_code=payload.product_code,
        forecast_run_id=run.id,
        expected_demand_lead_time=plan.expected_demand_lead_time,
        safety_stock=plan.safety_stock,
        reorder_point=plan.reorder_point,
        recommended_order_quantity=plan.recommended_order_quantity,
        expected_stockout_risk=plan.expected_stockout_risk,
        estimated_holding_cost=plan.estimated_holding_cost,
        estimated_ordering_cost=plan.estimated_ordering_cost,
        stockout_flag=plan.stockout_flag,
        excess_inventory_flag=plan.excess_inventory_flag,
        assumptions=plan.assumptions,
    )


@router.post("/recommendations", response_model=RecommendationOut)
async def create_recommendation(payload: RecommendationRequest, db: Session = Depends(db_session)) -> RecommendationOut:
    try:
        return await run_in_threadpool(_recommend, db, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown product {exc}") from exc
