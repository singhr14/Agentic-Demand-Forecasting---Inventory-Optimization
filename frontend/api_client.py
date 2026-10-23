from __future__ import annotations

import os
from typing import Any

import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


class APIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class BackendUnavailable(APIError):
    pass


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE_URL, timeout=120.0)


def request(method: str, path: str, **kwargs: Any) -> Any:
    try:
        with _client() as client:
            response = client.request(method, path, **kwargs)
    except httpx.ConnectError as exc:
        raise BackendUnavailable(
            f"Cannot reach the API at {API_BASE_URL}. Start the FastAPI server first."
        ) from exc
    except httpx.HTTPError as exc:
        raise APIError(str(exc)) from exc
    if response.status_code >= 400:
        detail = response.text
        try:
            payload = response.json()
            detail = payload.get("detail") or payload.get("error") or detail
        except Exception:  # noqa: BLE001
            pass
        raise APIError(str(detail), status_code=response.status_code)
    if response.headers.get("content-type", "").startswith("application/json"):
        return response.json()
    return response.text


def health() -> dict:
    return request("GET", "/health")


def products() -> list[dict]:
    return request("GET", "/api/v1/products")


def history(product_code: str) -> dict:
    return request("GET", f"/api/v1/products/{product_code}/history")


def validate_data(refresh: bool = False) -> dict:
    return request("POST", "/api/v1/data/validate", params={"refresh": refresh})


def create_forecast(product_code: str, horizon: int, model_name: str) -> dict:
    return request(
        "POST",
        "/api/v1/forecasts",
        json={"product_code": product_code, "horizon": horizon, "model_name": model_name},
    )


def list_forecasts() -> list[dict]:
    return request("GET", "/api/v1/forecasts")


def get_forecast(run_id: int) -> dict:
    return request("GET", f"/api/v1/forecasts/{run_id}")


def get_inventory(product_code: str) -> dict:
    return request("GET", f"/api/v1/inventory/{product_code}")


def update_inventory(product_code: str, payload: dict) -> dict:
    return request("PUT", f"/api/v1/inventory/{product_code}", json=payload)


def recommend(payload: dict) -> dict:
    return request("POST", "/api/v1/inventory/recommendations", json=payload)


def run_agents(product_code: str, horizon: int, model_name: str) -> dict:
    return request(
        "POST",
        "/api/v1/agents/run",
        json={"product_code": product_code, "horizon": horizon, "model_name": model_name},
    )


def get_agent_run(run_id: str) -> dict:
    return request("GET", f"/api/v1/agents/runs/{run_id}")


def run_evaluation(horizon: int = 7) -> dict:
    return request("POST", "/api/v1/evaluations/run", json={"horizon": horizon})


def latest_evaluation() -> dict:
    return request("GET", "/api/v1/evaluations/latest")
