from fastapi import APIRouter

from backend.app.api.v1 import agents, dashboard, data, evaluations, forecasts, inventory, products

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(products.router)
api_router.include_router(data.router)
api_router.include_router(forecasts.router)
api_router.include_router(inventory.router)
api_router.include_router(agents.router)
api_router.include_router(evaluations.router)
api_router.include_router(dashboard.router)
