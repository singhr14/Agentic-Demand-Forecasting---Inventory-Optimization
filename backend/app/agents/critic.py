from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CriticResult:
    valid: bool
    confidence: str
    review_required: bool
    warnings: list[str]


def critique_forecast(
    predictions: list[float],
    history_mean: float,
    history_std: float,
    history_days: int,
    min_history_days: int = 28,
) -> CriticResult:
    warnings: list[str] = []
    valid = True
    if not predictions:
        return CriticResult(False, "review_required", True, ["Forecast is empty."])
    arr = np.asarray(predictions, dtype=float)
    if np.isnan(arr).any():
        valid = False
        warnings.append("Forecast contains missing values.")
    if (arr < 0).any():
        valid = False
        warnings.append("Forecast contains negative demand values.")
    if history_days < min_history_days:
        warnings.append(f"History is short ({history_days} days); estimates may be unstable.")
    mean_pred = float(np.nanmean(arr))
    if history_mean > 0 and mean_pred > 3.0 * history_mean:
        warnings.append(
            f"Mean forecast {mean_pred:.1f} is more than 3x recent mean demand {history_mean:.1f}."
        )
    if history_mean > 1 and mean_pred < 0.1 * history_mean:
        warnings.append(
            f"Mean forecast {mean_pred:.1f} is far below recent mean demand {history_mean:.1f}."
        )
    if history_std > 0 and abs(mean_pred - history_mean) > 4 * history_std:
        warnings.append("Forecast mean is outside a 4-sigma band of recent demand.")
    unusual = any("3x" in w or "far below" in w or "4-sigma" in w for w in warnings)
    if not valid:
        confidence = "review_required"
        review = True
    elif history_days < min_history_days or unusual:
        confidence = "low"
        review = True
    elif warnings:
        confidence = "medium"
        review = False
    else:
        confidence = "high"
        review = False
    return CriticResult(valid=valid, confidence=confidence, review_required=review, warnings=warnings)
