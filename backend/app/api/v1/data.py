from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.core.config import get_settings
from backend.app.ml.data_ingestion import DataValidationError, ingest, load_m5_tables
from backend.app.ml.data_store import data_store
from backend.app.schemas.api import DataValidateOut

router = APIRouter(prefix="/data", tags=["data"])


@router.post("/validate", response_model=DataValidateOut)
def validate_data(refresh: bool = False) -> DataValidateOut:
    settings = get_settings()
    try:
        if refresh:
            ingest(settings, use_sample=settings.SAMPLE_DATA_MODE)
            data_store.load(force=True)
        source = settings.sample_dir if settings.SAMPLE_DATA_MODE else settings.raw_dir
        load_m5_tables(source)
        meta = data_store.metadata()
        label = meta.get("label", "unknown")
        return DataValidateOut(
            valid=True,
            synthetic=bool(meta.get("synthetic")),
            message=f"Data source is valid ({label}).",
            metadata=meta,
        )
    except DataValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
