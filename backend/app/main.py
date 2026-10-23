from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging
from backend.app.database import init_db
from backend.app.api.v1 import api_router
from backend.app.ml.data_ingestion import ingest
from backend.app.ml.data_store import data_store
from backend.app.services.catalog import sync_products
from backend.app.database import SessionLocal

logger = logging.getLogger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    init_db()
    try:
        ingest(settings, use_sample=settings.SAMPLE_DATA_MODE)
        data_store.load(force=True)
        db = SessionLocal()
        try:
            sync_products(db)
        finally:
            db.close()
        logger.info(
            "Startup complete sample_mode=%s llm_configured=%s",
            settings.SAMPLE_DATA_MODE,
            settings.llm_configured,
        )
        if not settings.llm_configured:
            logger.info("LLM_API_KEY is empty; explanations will use the deterministic fallback via OpenRouter skip.")
    except Exception:
        logger.exception("Startup data initialization failed; API will still boot")
    yield


settings = get_settings()
app = FastAPI(
    title="Agentic Demand Forecasting & Inventory Optimization",
    version="1.0.0",
    description="Forecast demand, critique predictions, and recommend replenishment with a LangGraph multi-agent workflow.",
    lifespan=lifespan,
)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "sample_data_mode": s.SAMPLE_DATA_MODE,
        "llm_provider": s.LLM_PROVIDER,
        "llm_configured": s.llm_configured,
    }


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "request_id": getattr(request.state, "request_id", "-"),
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "request_id": getattr(request.state, "request_id", "-"),
                "details": exc.detail,
            },
        )
    logger.exception("Unhandled error request_id=%s", getattr(request.state, "request_id", "-"))
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "request_id": getattr(request.state, "request_id", "-"),
            "details": "An unexpected error occurred.",
        },
    )
