from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (UniqueConstraint("product_code", name="uq_products_code"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_code: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    item_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    product_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    department: Mapped[str] = mapped_column(String(64), nullable=False)
    store_code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    state_code: Mapped[str] = mapped_column(String(16), nullable=False)
    is_synthetic: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    inventory: Mapped["InventoryState | None"] = relationship(back_populates="product")
    forecast_runs: Mapped[list["ForecastRun"]] = relationship(back_populates="product")


class ForecastRun(Base):
    __tablename__ = "forecast_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True, nullable=False)
    product_code: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    forecast_start_date: Mapped[date] = mapped_column(Date, nullable=False)
    forecast_horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    product: Mapped[Product] = relationship(back_populates="forecast_runs")
    results: Mapped[list["ForecastResult"]] = relationship(back_populates="forecast_run", cascade="all, delete-orphan")
    recommendation: Mapped["InventoryRecommendation | None"] = relationship(back_populates="forecast_run")


class ForecastResult(Base):
    __tablename__ = "forecast_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    forecast_run_id: Mapped[int] = mapped_column(ForeignKey("forecast_runs.id"), index=True, nullable=False)
    forecast_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    predicted_demand: Mapped[float] = mapped_column(Float, nullable=False)
    lower_bound: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper_bound: Mapped[float | None] = mapped_column(Float, nullable=True)

    forecast_run: Mapped[ForecastRun] = relationship(back_populates="results")


class InventoryState(Base):
    __tablename__ = "inventory_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), unique=True, nullable=False)
    product_code: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    current_stock: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    on_order_quantity: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    lead_time_days: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    holding_cost_per_unit: Mapped[float] = mapped_column(Float, nullable=False, default=0.15)
    ordering_cost: Mapped[float] = mapped_column(Float, nullable=False, default=25.0)
    service_level_target: Mapped[float] = mapped_column(Float, nullable=False, default=0.95)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    product: Mapped[Product] = relationship(back_populates="inventory")


class InventoryRecommendation(Base):
    __tablename__ = "inventory_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    forecast_run_id: Mapped[int] = mapped_column(ForeignKey("forecast_runs.id"), index=True, nullable=False)
    recommended_order_quantity: Mapped[float] = mapped_column(Float, nullable=False)
    reorder_point: Mapped[float] = mapped_column(Float, nullable=False)
    safety_stock: Mapped[float] = mapped_column(Float, nullable=False)
    expected_stockout_risk: Mapped[float] = mapped_column(Float, nullable=False)
    estimated_holding_cost: Mapped[float] = mapped_column(Float, nullable=False)
    estimated_ordering_cost: Mapped[float] = mapped_column(Float, nullable=False)
    assumptions_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    forecast_run: Mapped[ForecastRun] = relationship(back_populates="recommendation")


class AgentExecutionLog(Base):
    __tablename__ = "agent_execution_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    agent_name: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    input_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    output_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    warnings_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    test_start_date: Mapped[date] = mapped_column(Date, nullable=False)
    test_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
