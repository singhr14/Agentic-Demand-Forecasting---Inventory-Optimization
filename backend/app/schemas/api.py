from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class ProductOut(BaseModel):
    product_code: str
    item_id: str
    category: str
    department: str
    store_code: str
    state_code: str
    is_synthetic: bool


class HistoryPoint(BaseModel):
    date: date
    demand: float
    sell_price: float | None = None
    event_name_1: str | None = None


class ProductHistoryOut(BaseModel):
    product_code: str
    data_label: str
    points: list[HistoryPoint]


class DataValidateOut(BaseModel):
    valid: bool
    synthetic: bool
    message: str
    metadata: dict


class ForecastRequest(BaseModel):
    product_code: str
    horizon: int = Field(default=7, ge=1, le=28)
    model_name: str = Field(default="lightgbm")


class ForecastPoint(BaseModel):
    forecast_date: date
    predicted_demand: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    horizon_step: int


class ForecastRunOut(BaseModel):
    id: int
    product_code: str
    model_name: str
    model_version: str
    forecast_start_date: date
    forecast_horizon: int
    status: str
    created_at: datetime
    results: list[ForecastPoint] = []
    warnings: list[str] = []


class InventoryStateIn(BaseModel):
    current_stock: float = Field(ge=0)
    on_order_quantity: float = Field(ge=0)
    lead_time_days: int = Field(ge=1, le=30)
    holding_cost_per_unit: float = Field(ge=0)
    ordering_cost: float = Field(ge=0)
    service_level_target: float = Field(gt=0.5, lt=0.999)


class InventoryStateOut(InventoryStateIn):
    product_code: str
    updated_at: datetime | None = None


class RecommendationRequest(BaseModel):
    product_code: str
    horizon: int = Field(default=7, ge=1, le=28)
    model_name: str = "lightgbm"
    current_stock: float | None = Field(default=None, ge=0)
    on_order_quantity: float | None = Field(default=None, ge=0)
    lead_time_days: int | None = Field(default=None, ge=1, le=30)
    holding_cost_per_unit: float | None = Field(default=None, ge=0)
    ordering_cost: float | None = Field(default=None, ge=0)
    service_level_target: float | None = Field(default=None, gt=0.5, lt=0.999)


class RecommendationOut(BaseModel):
    product_code: str
    forecast_run_id: int
    expected_demand_lead_time: float
    safety_stock: float
    reorder_point: float
    recommended_order_quantity: float
    expected_stockout_risk: float
    estimated_holding_cost: float
    estimated_ordering_cost: float
    stockout_flag: bool
    excess_inventory_flag: bool
    assumptions: dict


class AgentRunRequest(BaseModel):
    product_code: str
    horizon: int = Field(default=7, ge=1, le=28)
    model_name: str = "lightgbm"


class AgentLogOut(BaseModel):
    agent_name: str
    status: str
    input_summary: str
    output_summary: str
    warnings: list[str] = []
    error_message: str | None = None
    started_at: datetime
    completed_at: datetime | None = None


class AgentRunOut(BaseModel):
    run_id: str
    status: str
    product_code: str
    review_required: bool
    forecast_run_id: int | None = None
    recommendation: dict | None = None
    explanation: str
    confidence: str
    warnings: list[str] = []
    logs: list[AgentLogOut] = []


class EvaluationRequest(BaseModel):
    horizon: int = Field(default=7, ge=1, le=14)


class ErrorOut(BaseModel):
    error: str
    request_id: str
    details: str | None = None
