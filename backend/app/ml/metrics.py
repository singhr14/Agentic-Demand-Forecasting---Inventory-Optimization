from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """Weighted Absolute Percentage Error. Safe when total demand is zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.sum(np.abs(y_true)))
    if denom < eps:
        return 0.0 if float(np.sum(np.abs(y_pred))) < eps else 1.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error (prediction minus actual). Positive means over-forecast."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "wape": round(wape(y_true, y_pred), 4),
        "bias": round(forecast_bias(y_true, y_pred), 4),
        "n_obs": int(len(y_true)),
    }


def metrics_from_frame(frame: pd.DataFrame, actual_col: str = "actual", pred_col: str = "predicted") -> dict[str, float]:
    return regression_metrics(frame[actual_col].to_numpy(), frame[pred_col].to_numpy())
