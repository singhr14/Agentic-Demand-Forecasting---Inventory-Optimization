from __future__ import annotations

import json
from datetime import date, datetime

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from backend.app.models.entities import (
    AgentExecutionLog,
    EvaluationRun,
    ForecastResult,
    ForecastRun,
    InventoryRecommendation,
    InventoryState,
    Product,
)


class ProductRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert_from_frame_row(self, row: dict) -> Product:
        product = self.db.scalar(select(Product).where(Product.product_code == row["product_code"]))
        if product is None:
            product = Product(
                product_code=row["product_code"],
                item_id=row["item_id"],
                product_name=row.get("item_id"),
                category=row["cat_id"],
                department=row["dept_id"],
                store_code=row["store_id"],
                state_code=row["state_id"],
                is_synthetic=int(row.get("is_synthetic", 0)),
            )
            self.db.add(product)
            self.db.flush()
        return product

    def get_by_code(self, product_code: str) -> Product | None:
        return self.db.scalar(select(Product).where(Product.product_code == product_code))

    def list_all(self) -> list[Product]:
        return list(self.db.scalars(select(Product).order_by(Product.product_code)).all())


class ForecastRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_run(
        self,
        product: Product,
        model_name: str,
        model_version: str,
        forecast_start: date,
        horizon: int,
        status: str,
        rows: list[dict],
    ) -> ForecastRun:
        run = ForecastRun(
            product_id=product.id,
            product_code=product.product_code,
            model_name=model_name,
            model_version=model_version,
            forecast_start_date=forecast_start,
            forecast_horizon=horizon,
            status=status,
        )
        self.db.add(run)
        self.db.flush()
        for row in rows:
            self.db.add(
                ForecastResult(
                    forecast_run_id=run.id,
                    forecast_date=row["forecast_date"],
                    predicted_demand=row["predicted_demand"],
                    lower_bound=row.get("lower_bound"),
                    upper_bound=row.get("upper_bound"),
                )
            )
        self.db.commit()
        self.db.refresh(run)
        return run

    def get(self, run_id: int) -> ForecastRun | None:
        return self.db.get(ForecastRun, run_id)

    def list_runs(self, limit: int = 50) -> list[ForecastRun]:
        stmt = select(ForecastRun).order_by(desc(ForecastRun.created_at)).limit(limit)
        return list(self.db.scalars(stmt).all())


class InventoryRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get(self, product_code: str) -> InventoryState | None:
        return self.db.scalar(select(InventoryState).where(InventoryState.product_code == product_code))

    def upsert(self, product: Product, payload: dict) -> InventoryState:
        state = self.get(product.product_code)
        if state is None:
            state = InventoryState(product_id=product.id, product_code=product.product_code)
            self.db.add(state)
        for key, value in payload.items():
            setattr(state, key, value)
        self.db.commit()
        self.db.refresh(state)
        return state

    def save_recommendation(self, forecast_run_id: int, plan: dict) -> InventoryRecommendation:
        rec = InventoryRecommendation(
            forecast_run_id=forecast_run_id,
            recommended_order_quantity=plan["recommended_order_quantity"],
            reorder_point=plan["reorder_point"],
            safety_stock=plan["safety_stock"],
            expected_stockout_risk=plan["expected_stockout_risk"],
            estimated_holding_cost=plan["estimated_holding_cost"],
            estimated_ordering_cost=plan["estimated_ordering_cost"],
            assumptions_json=json.dumps(plan.get("assumptions", {})),
        )
        self.db.add(rec)
        self.db.commit()
        self.db.refresh(rec)
        return rec

    def list_at_risk(self, product_codes: list[str]) -> list[InventoryState]:
        if not product_codes:
            return []
        stmt = select(InventoryState).where(InventoryState.product_code.in_(product_codes))
        return list(self.db.scalars(stmt).all())


class AgentLogRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add(
        self,
        run_id: str,
        agent_name: str,
        status: str,
        started_at: datetime,
        completed_at: datetime | None,
        input_summary: str,
        output_summary: str,
        warnings: list[str],
        error_message: str | None = None,
    ) -> AgentExecutionLog:
        log = AgentExecutionLog(
            run_id=run_id,
            agent_name=agent_name,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            input_summary=input_summary,
            output_summary=output_summary,
            warnings_json=json.dumps(warnings),
            error_message=error_message,
        )
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        return log

    def list_for_run(self, run_id: str) -> list[AgentExecutionLog]:
        stmt = select(AgentExecutionLog).where(AgentExecutionLog.run_id == run_id).order_by(AgentExecutionLog.started_at)
        return list(self.db.scalars(stmt).all())

    def recent(self, limit: int = 20) -> list[AgentExecutionLog]:
        stmt = select(AgentExecutionLog).order_by(desc(AgentExecutionLog.started_at)).limit(limit)
        return list(self.db.scalars(stmt).all())


class EvaluationRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add(self, model_name: str, start: date, end: date, metrics: dict) -> EvaluationRun:
        run = EvaluationRun(
            model_name=model_name,
            test_start_date=start,
            test_end_date=end,
            metrics_json=json.dumps(metrics),
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def latest(self) -> list[EvaluationRun]:
        stmt = select(EvaluationRun).order_by(desc(EvaluationRun.created_at)).limit(10)
        return list(self.db.scalars(stmt).all())
