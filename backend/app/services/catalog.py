from __future__ import annotations

from sqlalchemy.orm import Session

from backend.app.ml.data_store import data_store
from backend.app.models.entities import InventoryState
from backend.app.repositories.base import ProductRepository


def sync_products(db: Session) -> int:
    products_df = data_store.products()
    repo = ProductRepository(db)
    count = 0
    for row in products_df.to_dict(orient="records"):
        product = repo.upsert_from_frame_row(row)
        existing = db.query(InventoryState).filter(InventoryState.product_id == product.id).one_or_none()
        if existing is None:
            recent = data_store.history(product.product_code).tail(14)
            avg = float(recent["demand"].mean()) if not recent.empty else 10.0
            db.add(
                InventoryState(
                    product_id=product.id,
                    product_code=product.product_code,
                    current_stock=max(avg * 5, 8),
                    on_order_quantity=0,
                    lead_time_days=3,
                    holding_cost_per_unit=0.15,
                    ordering_cost=25.0,
                    service_level_target=0.95,
                )
            )
        count += 1
    db.commit()
    return count
