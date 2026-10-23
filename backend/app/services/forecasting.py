from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.ml.baseline import SeasonalNaiveForecaster
from backend.app.ml.data_store import data_store
from backend.app.ml.lightgbm_model import LightGBMForecaster
from backend.app.models.entities import Product
from backend.app.repositories.base import ForecastRepository, ProductRepository
from backend.app.services.catalog import sync_products

_lgbm: LightGBMForecaster | None = None


class InsufficientHistoryError(ValueError):
    pass


def get_lgbm() -> LightGBMForecaster:
    global _lgbm
    settings = get_settings()
    if _lgbm is None:
        _lgbm = LightGBMForecaster(settings.MODEL_DIR)
        if _lgbm.model_path.exists():
            _lgbm.load()
    return _lgbm


def ensure_models() -> LightGBMForecaster:
    model = get_lgbm()
    if model.model is None and not model.model_path.exists():
        df = data_store.load()
        model.fit(df, horizon=get_settings().FORECAST_HORIZON)
    elif model.model is None:
        model.load()
    return model


def product_history(product_code: str) -> pd.DataFrame:
    hist = data_store.history(product_code)
    if hist.empty:
        raise KeyError(f"Unknown product_code: {product_code}")
    return hist


def generate_forecast_frame(product_code: str, horizon: int, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    settings = get_settings()
    hist = product_history(product_code)
    min_days = settings.MIN_HISTORY_DAYS
    warnings: list[str] = []
    if len(hist) < min_days:
        warnings.append(f"Insufficient history: {len(hist)} days available, recommended {min_days}.")
        if len(hist) < 7:
            raise InsufficientHistoryError(
                f"Product {product_code} has only {len(hist)} days of history; at least 7 are required."
            )
    residual = float(hist["demand"].tail(28).std() or 1.0)
    if model_name == "seasonal_naive":
        preds = SeasonalNaiveForecaster().forecast(hist, horizon, residual_std=residual)
        version = SeasonalNaiveForecaster.version
        name = SeasonalNaiveForecaster.name
    else:
        model = ensure_models()
        preds = model.forecast(hist, horizon)
        version = model.metadata.get("model_version", "1.0.0")
        name = "lightgbm"
        residual = model.residual_std
    meta = {
        "model_name": name,
        "model_version": version,
        "residual_std": residual,
        "history_days": int(len(hist)),
        "history_mean": float(hist["demand"].tail(28).mean()),
        "history_std": residual,
        "last_actual_date": str(pd.to_datetime(hist["date"].max()).date()),
        "recent_demand": hist.tail(14)[["date", "demand", "sell_price", "event_name_1"]].to_dict(orient="records"),
        "warnings": warnings,
        "synthetic": bool(hist["is_synthetic"].max()) if "is_synthetic" in hist.columns else False,
    }
    return preds, hist, meta


def persist_forecast(db: Session, product_code: str, preds: pd.DataFrame, meta: dict, status: str = "completed"):
    repo_p = ProductRepository(db)
    product = repo_p.get_by_code(product_code)
    if product is None:
        sync_products(db)
        product = repo_p.get_by_code(product_code)
    if product is None:
        raise KeyError(product_code)
    rows = []
    for rec in preds.to_dict(orient="records"):
        fd = rec["forecast_date"]
        if not isinstance(fd, date):
            fd = pd.to_datetime(fd).date()
        rows.append(
            {
                "forecast_date": fd,
                "predicted_demand": float(rec["predicted_demand"]),
                "lower_bound": rec.get("lower_bound"),
                "upper_bound": rec.get("upper_bound"),
            }
        )
    start = rows[0]["forecast_date"]
    run = ForecastRepository(db).create_run(
        product=product,
        model_name=meta["model_name"],
        model_version=meta["model_version"],
        forecast_start=start,
        horizon=len(rows),
        status=status,
        rows=rows,
    )
    return run, product


def get_product_or_sync(db: Session, product_code: str) -> Product:
    product = ProductRepository(db).get_by_code(product_code)
    if product is None:
        sync_products(db)
        product = ProductRepository(db).get_by_code(product_code)
    if product is None:
        raise KeyError(product_code)
    return product
