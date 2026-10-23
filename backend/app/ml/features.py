from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 14, 28]
CATEGORICALS = ["item_id", "store_id", "cat_id", "dept_id"]
NUMERIC_BASE = [
    "sell_price",
    "weekday_num",
    "month",
    "is_event",
    "wday",
]


@dataclass
class FeaturePipeline:
    encoders: dict[str, LabelEncoder] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    lag_list: list[int] = field(default_factory=lambda: list(LAGS))
    roll_windows: list[int] = field(default_factory=lambda: list(ROLL_WINDOWS))

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        out["weekday_num"] = out["date"].dt.dayofweek
        out["month"] = out["date"].dt.month
        if "is_event" not in out.columns:
            event = out["event_name_1"] if "event_name_1" in out.columns else ""
            out["is_event"] = (pd.Series(event, index=out.index).astype(str).str.len() > 0).astype(int)
        if "wday" not in out.columns:
            out["wday"] = out["weekday_num"] + 1
        out["sell_price"] = pd.to_numeric(out["sell_price"], errors="coerce").ffill().bfill().fillna(0)
        return out

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lags and rolling stats use only prior observations (shifted demand)."""
        out = df.sort_values(["product_code", "date"]).copy()
        grouped = out.groupby("product_code", observed=False)["demand"]
        for lag in self.lag_list:
            out[f"lag_{lag}"] = grouped.shift(lag)
        out["_demand_shifted"] = grouped.shift(1)
        shifted_group = out.groupby("product_code", observed=False)["_demand_shifted"]
        for window in self.roll_windows:
            min_p = max(3, window // 3)
            out[f"roll_mean_{window}"] = shifted_group.transform(lambda s: s.rolling(window, min_periods=min_p).mean())
            out[f"roll_std_{window}"] = shifted_group.transform(lambda s: s.rolling(window, min_periods=min_p).std())
        return out.drop(columns=["_demand_shifted"])

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        featured = self.add_lag_features(self.add_time_features(df))
        for col in CATEGORICALS:
            encoder = LabelEncoder()
            values = featured[col].astype(str).fillna("UNK")
            encoder.fit(pd.concat([values, pd.Series(["UNK"])], ignore_index=True))
            self.encoders[col] = encoder
            featured[f"{col}_enc"] = encoder.transform(values)
        self.feature_columns = self._columns(featured)
        return featured

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        featured = self.add_lag_features(self.add_time_features(df))
        for col in CATEGORICALS:
            encoder = self.encoders[col]
            values = featured[col].astype(str).fillna("UNK")
            known = set(encoder.classes_)
            values = values.where(values.isin(known), "UNK")
            featured[f"{col}_enc"] = encoder.transform(values)
        return featured

    def _columns(self, df: pd.DataFrame) -> list[str]:
        lag_cols = [f"lag_{lag}" for lag in self.lag_list]
        roll_cols = [f"roll_mean_{w}" for w in self.roll_windows] + [f"roll_std_{w}" for w in self.roll_windows]
        cat_cols = [f"{c}_enc" for c in CATEGORICALS]
        cols = NUMERIC_BASE + lag_cols + roll_cols + cat_cols
        return [c for c in cols if c in df.columns]

    def matrix(self, df: pd.DataFrame, dropna: bool = True) -> tuple[pd.DataFrame, pd.Series | None]:
        featured = df.copy()
        if dropna:
            featured = featured.dropna(subset=self.feature_columns)
        X = featured[self.feature_columns].astype(float).fillna(0)
        y = featured["demand"].astype(float) if "demand" in featured.columns else None
        return X, y


def chronological_splits(
    df: pd.DataFrame,
    valid_days: int = 14,
    test_days: int = 14,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date")
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    valid_start = test_start - pd.Timedelta(days=valid_days)
    train = df[df["date"] < valid_start]
    valid = df[(df["date"] >= valid_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start]
    return train, valid, test
