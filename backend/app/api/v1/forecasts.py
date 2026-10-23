from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from backend.app.api.deps import db_session
from backend.app.repositories.base import ForecastRepository
from backend.app.schemas.api import ForecastPoint, ForecastRequest, ForecastRunOut
from backend.app.services.forecasting import InsufficientHistoryError, generate_forecast_frame, persist_forecast

router = APIRouter(prefix="/forecasts", tags=["forecasts"])


def _run_out(run, warnings: list[str] | None = None) -> ForecastRunOut:
    results = [
        ForecastPoint(
            forecast_date=r.forecast_date,
            predicted_demand=r.predicted_demand,
            lower_bound=r.lower_bound,
            upper_bound=r.upper_bound,
            horizon_step=i + 1,
        )
        for i, r in enumerate(sorted(run.results, key=lambda x: x.forecast_date))
    ]
    return ForecastRunOut(
        id=run.id,
        product_code=run.product_code,
        model_name=run.model_name,
        model_version=run.model_version,
        forecast_start_date=run.forecast_start_date,
        forecast_horizon=run.forecast_horizon,
        status=run.status,
        created_at=run.created_at,
        results=results,
        warnings=warnings or [],
    )


def _create(db: Session, payload: ForecastRequest) -> ForecastRunOut:
    try:
        preds, _hist, meta = generate_forecast_frame(payload.product_code, payload.horizon, payload.model_name)
        run, _ = persist_forecast(db, payload.product_code, preds, meta)
        return _run_out(run, meta.get("warnings") or [])
    except InsufficientHistoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown product {exc}") from exc


@router.post("", response_model=ForecastRunOut)
async def create_forecast(payload: ForecastRequest, db: Session = Depends(db_session)) -> ForecastRunOut:
    return await run_in_threadpool(_create, db, payload)


@router.get("", response_model=list[ForecastRunOut])
def list_forecasts(db: Session = Depends(db_session)) -> list[ForecastRunOut]:
    runs = ForecastRepository(db).list_runs()
    return [_run_out(run) for run in runs]


@router.get("/{forecast_run_id}", response_model=ForecastRunOut)
def get_forecast(forecast_run_id: int, db: Session = Depends(db_session)) -> ForecastRunOut:
    run = ForecastRepository(db).get(forecast_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Forecast run not found")
    return _run_out(run)
