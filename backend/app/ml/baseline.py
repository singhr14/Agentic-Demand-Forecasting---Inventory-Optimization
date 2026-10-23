from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


class SeasonalNaiveForecaster:
    """Seasonal-naive baseline using only observations before the prediction origin."""

    name = "seasonal_naive"
    version = "1.0.0"

    def __init__(self, season: int = 7) -> None:
        self.season = season

    def forecast(
        self,
        history: pd.DataFrame,
        horizon: int,
        residual_std: float | None = None,
    ) -> pd.DataFrame:
        history = history.sort_values("date")
        if history.empty:
            raise ValueError("No history available for baseline forecast")
        origin = pd.to_datetime(history["date"].max())
        values = history["demand"].astype(float).to_numpy()
        season_slice = values[-self.season :] if len(values) >= self.season else values
        if len(season_slice) == 0:
            season_slice = np.array([0.0])
        sigma = residual_std if residual_std is not None else float(np.nanstd(values[-28:]) if len(values) > 2 else 1.0)
        sigma = max(sigma, 0.0)
        rows = []
        for step in range(1, horizon + 1):
            pred = max(0.0, float(season_slice[(step - 1) % len(season_slice)]))
            dt = (origin + timedelta(days=step)).date()
            rows.append(
                {
                    "forecast_date": dt,
                    "predicted_demand": pred,
                    "lower_bound": max(0.0, pred - 1.96 * sigma),
                    "upper_bound": pred + 1.96 * sigma,
                    "model_name": self.name,
                    "model_version": self.version,
                    "horizon_step": step,
                }
            )
        return pd.DataFrame(rows)
