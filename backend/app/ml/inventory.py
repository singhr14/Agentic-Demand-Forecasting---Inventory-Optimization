from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import norm


Z_BY_SERVICE = {
    0.80: 0.8416,
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.96,
    0.99: 2.3263,
}


@dataclass
class InventoryInputs:
    current_stock: float
    on_order_quantity: float
    lead_time_days: int
    holding_cost_per_unit: float
    ordering_cost: float
    service_level_target: float
    daily_forecast: list[float]
    residual_std: float = 1.0


@dataclass
class InventoryPlan:
    expected_demand_lead_time: float
    safety_stock: float
    reorder_point: float
    recommended_order_quantity: float
    expected_stockout_risk: float
    estimated_holding_cost: float
    estimated_ordering_cost: float
    inventory_position: float
    stockout_flag: bool
    excess_inventory_flag: bool
    assumptions: dict


def z_for_service_level(service_level: float) -> float:
    if service_level in Z_BY_SERVICE:
        return Z_BY_SERVICE[service_level]
    clipped = min(max(service_level, 0.5), 0.999)
    return float(norm.ppf(clipped))


def plan_inventory(inputs: InventoryInputs) -> InventoryPlan:
    """(s, Q) policy using forecasted lead-time demand and z-score safety stock.

    Simplifying assumptions:
    - Demand is treated as i.i.d. around the forecast with residual_std as the daily sigma.
    - Lead time is deterministic.
    - Review is continuous for this recommendation (single snapshot).
    - M5 does not provide real inventory, lead times, or costs; those are simulation inputs.
    """
    horizon = max(1, inputs.lead_time_days)
    forecasts = list(inputs.daily_forecast)
    if not forecasts:
        forecasts = [0.0]
    if len(forecasts) < horizon:
        mean_f = float(np.mean(forecasts))
        forecasts = forecasts + [mean_f] * (horizon - len(forecasts))
    lt_demand = float(sum(forecasts[:horizon]))
    daily_sigma = max(float(inputs.residual_std), 0.0)
    z = z_for_service_level(inputs.service_level_target)
    safety = max(0.0, z * daily_sigma * np.sqrt(horizon))
    reorder_point = lt_demand + safety
    position = float(inputs.current_stock + inputs.on_order_quantity)
    order_qty = max(0.0, np.ceil(reorder_point - position))
    stockout_risk = float(max(0.0, min(1.0, 1.0 - inputs.service_level_target)))
    if position < lt_demand:
        stockout_risk = max(stockout_risk, min(0.99, (lt_demand - position) / max(lt_demand, 1.0)))
    holding = float(max(position + order_qty - lt_demand, 0.0) * inputs.holding_cost_per_unit)
    ordering = float(inputs.ordering_cost if order_qty > 0 else 0.0)
    return InventoryPlan(
        expected_demand_lead_time=round(lt_demand, 4),
        safety_stock=round(float(safety), 4),
        reorder_point=round(float(reorder_point), 4),
        recommended_order_quantity=round(float(order_qty), 4),
        expected_stockout_risk=round(stockout_risk, 4),
        estimated_holding_cost=round(holding, 4),
        estimated_ordering_cost=round(ordering, 4),
        inventory_position=round(position, 4),
        stockout_flag=bool(position < reorder_point),
        excess_inventory_flag=bool(position > reorder_point * 1.5 and order_qty == 0),
        assumptions={
            "policy": "reorder_point_with_safety_stock",
            "lead_time_days": inputs.lead_time_days,
            "service_level_target": inputs.service_level_target,
            "z_score": round(z, 4),
            "daily_demand_sigma": daily_sigma,
            "sigma_source": "model residual std or historical demand std",
            "inventory_and_cost_inputs": "simulation assumptions, not M5 fields",
        },
    )


def plan_to_dict(plan: InventoryPlan) -> dict:
    payload = asdict(plan)
    payload["stockout_flag"] = bool(payload["stockout_flag"])
    payload["excess_inventory_flag"] = bool(payload["excess_inventory_flag"])
    for key in (
        "expected_demand_lead_time",
        "safety_stock",
        "reorder_point",
        "recommended_order_quantity",
        "expected_stockout_risk",
        "estimated_holding_cost",
        "estimated_ordering_cost",
        "inventory_position",
    ):
        payload[key] = float(payload[key])
    return payload
