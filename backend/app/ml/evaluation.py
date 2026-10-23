from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.app.ml.baseline import SeasonalNaiveForecaster
from backend.app.ml.features import chronological_splits
from backend.app.ml.inventory import InventoryInputs, plan_inventory
from backend.app.ml.lightgbm_model import LightGBMForecaster
from backend.app.ml.metrics import metrics_from_frame, regression_metrics


@dataclass
class InventorySimConfig:
    lead_time_days: int = 3
    initial_inventory: float = 40.0
    holding_cost_per_unit: float = 0.15
    ordering_cost: float = 25.0
    service_level_target: float = 0.95
    residual_std: float = 3.0


def _forecast_at_cutoff(
    history: pd.DataFrame,
    model_name: str,
    horizon: int,
    lgbm: LightGBMForecaster | None,
) -> pd.DataFrame:
    if model_name == "lightgbm":
        if lgbm is None:
            raise ValueError("LightGBM model required")
        return lgbm.forecast(history, horizon)
    return SeasonalNaiveForecaster().forecast(history, horizon)


def evaluate_point_forecasts(
    df: pd.DataFrame,
    lgbm: LightGBMForecaster,
    horizon: int = 7,
) -> dict:
    _, _, test = chronological_splits(df)
    if test.empty:
        raise ValueError("Test split is empty")
    origin = test["date"].min() - pd.Timedelta(days=1)
    products = df["product_code"].unique()
    records: list[dict] = []
    for product in products:
        hist = df[(df["product_code"] == product) & (df["date"] <= origin)].copy()
        actuals = df[(df["product_code"] == product) & (df["date"] > origin)].head(horizon)
        if hist.empty or actuals.empty:
            continue
        for model_name in ("seasonal_naive", "lightgbm"):
            preds = _forecast_at_cutoff(hist, model_name, min(horizon, len(actuals)), lgbm)
            merged = actuals[["date", "demand"]].copy()
            merged["date"] = pd.to_datetime(merged["date"]).dt.date
            preds = preds.copy()
            preds["forecast_date"] = pd.to_datetime(preds["forecast_date"]).dt.date
            joined = merged.merge(preds, left_on="date", right_on="forecast_date", how="inner")
            for _, row in joined.iterrows():
                records.append(
                    {
                        "product_code": product,
                        "model_name": model_name,
                        "date": str(row["date"]),
                        "actual": float(row["demand"]),
                        "predicted": float(row["predicted_demand"]),
                    }
                )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No overlapping forecast/actual rows for evaluation")
    by_model = {}
    for model_name, grp in frame.groupby("model_name"):
        by_model[model_name] = metrics_from_frame(grp, "actual", "predicted")
    return {
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "models": by_model,
        "n_products": int(frame["product_code"].nunique()),
        "kind": "measured_forecast_accuracy_on_chronological_holdout",
    }


def rolling_backtest(
    df: pd.DataFrame,
    lgbm: LightGBMForecaster,
    horizon: int = 7,
    cutoffs: int = 3,
) -> dict:
    dates = sorted(df["date"].unique())
    if len(dates) < horizon + 40:
        return {"windows": [], "note": "Insufficient history for rolling backtest"}
    ends = dates[-(horizon * cutoffs) :: horizon]
    windows = []
    for origin in ends:
        hist = df[df["date"] <= origin]
        actuals = df[(df["date"] > origin) & (df["date"] <= origin + pd.Timedelta(days=horizon))]
        if actuals.empty:
            continue
        for model_name in ("seasonal_naive", "lightgbm"):
            rows = []
            for product, p_actual in actuals.groupby("product_code"):
                p_hist = hist[hist["product_code"] == product]
                if p_hist.empty:
                    continue
                preds = _forecast_at_cutoff(p_hist, model_name, horizon, lgbm)
                p_actual = p_actual.copy()
                p_actual["d"] = pd.to_datetime(p_actual["date"]).dt.date
                preds["forecast_date"] = pd.to_datetime(preds["forecast_date"]).dt.date
                joined = p_actual.merge(preds, left_on="d", right_on="forecast_date", how="inner")
                rows.extend(zip(joined["demand"].astype(float), joined["predicted_demand"].astype(float)))
            if not rows:
                continue
            y, yhat = zip(*rows)
            metrics = regression_metrics(np.array(y), np.array(yhat))
            windows.append(
                {
                    "origin": str(pd.to_datetime(origin).date()),
                    "model_name": model_name,
                    **metrics,
                }
            )
    return {"windows": windows, "kind": "rolling_chronological_backtest"}


def simulate_inventory(
    actual_demand: np.ndarray,
    forecasts_by_day: np.ndarray | None,
    config: InventorySimConfig,
    policy: str = "proposed",
) -> dict:
    """Daily inventory simulation. Inventory levels are simulated, not taken from M5."""
    on_hand = float(config.initial_inventory)
    pipeline: list[tuple[int, float]] = []
    stockouts = 0
    unmet = 0.0
    served = 0.0
    demand_total = 0.0
    inventory_sum = 0.0
    holding = 0.0
    ordering = 0.0
    orders = 0
    n = len(actual_demand)
    for t in range(n):
        arriving = [qty for day, qty in pipeline if day == t]
        on_hand += sum(arriving)
        pipeline = [(day, qty) for day, qty in pipeline if day != t]
        demand = float(actual_demand[t])
        demand_total += demand
        served_t = min(on_hand, demand)
        lost = demand - served_t
        on_hand -= served_t
        served += served_t
        unmet += lost
        if lost > 0:
            stockouts += 1
        inventory_sum += on_hand
        holding += on_hand * config.holding_cost_per_unit
        on_order = sum(qty for _, qty in pipeline)
        if policy == "proposed":
            start = t
            end = min(n, t + config.lead_time_days)
            if forecasts_by_day is not None and end > start:
                daily = list(forecasts_by_day[start:end])
            else:
                daily = [float(np.mean(actual_demand[max(0, t - 7) : t + 1]))] * config.lead_time_days
            plan = plan_inventory(
                InventoryInputs(
                    current_stock=on_hand,
                    on_order_quantity=on_order,
                    lead_time_days=config.lead_time_days,
                    holding_cost_per_unit=config.holding_cost_per_unit,
                    ordering_cost=config.ordering_cost,
                    service_level_target=config.service_level_target,
                    daily_forecast=daily,
                    residual_std=config.residual_std,
                )
            )
            qty = plan.recommended_order_quantity
        else:
            recent = float(np.mean(actual_demand[max(0, t - 7) : t + 1]))
            target = recent * config.lead_time_days
            qty = max(0.0, np.ceil(target - on_hand - on_order))
        if qty > 0:
            pipeline.append((t + config.lead_time_days, qty))
            ordering += config.ordering_cost
            orders += 1
    return {
        "stockout_rate": round(stockouts / max(n, 1), 4),
        "fill_rate": round(served / max(demand_total, 1e-9), 4),
        "units_lost": round(unmet, 4),
        "average_inventory": round(inventory_sum / max(n, 1), 4),
        "holding_cost": round(holding, 4),
        "ordering_cost": round(ordering, 4),
        "total_cost": round(holding + ordering, 4),
        "orders_placed": orders,
        "policy": policy,
        "kind": "simulated_inventory_outcomes",
        "assumptions": {
            "lead_time_days": config.lead_time_days,
            "initial_inventory": config.initial_inventory,
            "holding_cost_per_unit": config.holding_cost_per_unit,
            "ordering_cost": config.ordering_cost,
        },
    }


def compare_inventory_policies(
    df: pd.DataFrame,
    lgbm: LightGBMForecaster,
    horizon: int = 7,
    config: InventorySimConfig | None = None,
) -> dict:
    config = config or InventorySimConfig(residual_std=lgbm.residual_std)
    _, _, test = chronological_splits(df)
    origin = test["date"].min() - pd.Timedelta(days=1)
    products = list(df["product_code"].unique())[:8]
    proposed_stats = []
    baseline_stats = []
    for product in products:
        hist = df[(df["product_code"] == product) & (df["date"] <= origin)]
        actuals = df[(df["product_code"] == product) & (df["date"] > origin)].head(max(horizon * 2, 14))
        if hist.empty or actuals.empty:
            continue
        preds = lgbm.forecast(hist, len(actuals))
        y = actuals["demand"].to_numpy(dtype=float)
        yhat = preds["predicted_demand"].to_numpy(dtype=float)[: len(y)]
        proposed_stats.append(simulate_inventory(y, yhat, config, "proposed"))
        baseline_stats.append(simulate_inventory(y, None, config, "naive_ma_target"))

    def _avg(items: list[dict], key: str) -> float:
        if not items:
            return 0.0
        return round(float(np.mean([x[key] for x in items])), 4)

    return {
        "kind": "simulated_inventory_outcomes",
        "note": "Inventory levels, lead times, and costs are simulation inputs, not M5 dataset fields.",
        "proposed_policy": {
            "stockout_rate": _avg(proposed_stats, "stockout_rate"),
            "fill_rate": _avg(proposed_stats, "fill_rate"),
            "units_lost": _avg(proposed_stats, "units_lost"),
            "average_inventory": _avg(proposed_stats, "average_inventory"),
            "holding_cost": _avg(proposed_stats, "holding_cost"),
            "ordering_cost": _avg(proposed_stats, "ordering_cost"),
            "total_cost": _avg(proposed_stats, "total_cost"),
        },
        "naive_policy": {
            "stockout_rate": _avg(baseline_stats, "stockout_rate"),
            "fill_rate": _avg(baseline_stats, "fill_rate"),
            "units_lost": _avg(baseline_stats, "units_lost"),
            "average_inventory": _avg(baseline_stats, "average_inventory"),
            "holding_cost": _avg(baseline_stats, "holding_cost"),
            "ordering_cost": _avg(baseline_stats, "ordering_cost"),
            "total_cost": _avg(baseline_stats, "total_cost"),
        },
        "n_products_simulated": len(proposed_stats),
    }
