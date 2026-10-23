from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.api.deps import db_session
from backend.app.ml.data_store import data_store
from backend.app.repositories.base import ProductRepository
from backend.app.schemas.api import HistoryPoint, ProductHistoryOut, ProductOut
from backend.app.services.catalog import sync_products

router = APIRouter(tags=["products"])


@router.get("/products", response_model=list[ProductOut])
def list_products(db: Session = Depends(db_session)) -> list[ProductOut]:
    repo = ProductRepository(db)
    items = repo.list_all()
    if not items:
        sync_products(db)
        items = repo.list_all()
    return [
        ProductOut(
            product_code=p.product_code,
            item_id=p.item_id,
            category=p.category,
            department=p.department,
            store_code=p.store_code,
            state_code=p.state_code,
            is_synthetic=bool(p.is_synthetic),
        )
        for p in items
    ]


@router.get("/products/{product_code}/history", response_model=ProductHistoryOut)
def product_history(product_code: str) -> ProductHistoryOut:
    hist = data_store.history(product_code)
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No history for {product_code}")
    label = "demonstration data" if int(hist["is_synthetic"].max()) else "m5"
    points = []
    for rec in hist.to_dict(orient="records"):
        points.append(
            HistoryPoint(
                date=rec["date"].date() if hasattr(rec["date"], "date") else rec["date"],
                demand=float(rec["demand"]),
                sell_price=float(rec["sell_price"]) if rec.get("sell_price") is not None else None,
                event_name_1=str(rec.get("event_name_1") or "") or None,
            )
        )
    return ProductHistoryOut(product_code=product_code, data_label=label, points=points)
