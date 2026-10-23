from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

SALES_REQUIRED = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_REQUIRED = ["date", "wm_yr_wk", "weekday", "wday", "month", "year", "d"]
PRICES_REQUIRED = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
LONG_REQUIRED = [
    "product_code",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "date",
    "demand",
    "sell_price",
]


class DataValidationError(ValueError):
    pass


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataValidationError(f"{name} is missing required columns: {missing}")


def _require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise DataValidationError(f"Missing file: {path}. {hint}")


def generate_synthetic_m5(
    output_dir: Path,
    n_days: int = 120,
    seed: int = 42,
    start: str = "2015-01-05",
) -> dict:
    """Create a labeled demonstration dataset that mirrors M5 file shapes."""
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    stores = [
        ("CA_1", "CA"),
        ("TX_1", "TX"),
    ]
    catalog = [
        ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES", 8.0, 0.35),
        ("HOBBIES_1_002", "HOBBIES_1", "HOBBIES", 5.5, 0.20),
        ("HOUSEHOLD_1_001", "HOUSEHOLD_1", "HOUSEHOLD", 12.0, 0.45),
        ("HOUSEHOLD_1_002", "HOUSEHOLD_1", "HOUSEHOLD", 7.0, 0.25),
        ("FOODS_1_001", "FOODS_1", "FOODS", 18.0, 0.15),
        ("FOODS_1_002", "FOODS_1", "FOODS", 22.0, 0.10),
        ("FOODS_2_001", "FOODS_2", "FOODS", 14.0, 0.30),
        ("FOODS_3_001", "FOODS_3", "FOODS", 30.0, 0.08),
    ]

    dates = pd.date_range(start, periods=n_days, freq="D")
    calendar_rows = []
    for i, dt in enumerate(dates, start=1):
        event_name = None
        event_type = None
        if dt.month == 2 and dt.day == 14:
            event_name, event_type = "Valentines", "Cultural"
        elif dt.month == 7 and dt.day == 4:
            event_name, event_type = "IndependenceDay", "National"
        elif dt.month == 12 and dt.day == 25:
            event_name, event_type = "Christmas", "National"
        wm_yr_wk = int(dt.strftime("%y%W")) + 11100
        calendar_rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "wm_yr_wk": wm_yr_wk,
                "weekday": dt.strftime("%A"),
                "wday": ((dt.weekday() + 1) % 7) + 1,
                "month": dt.month,
                "year": dt.year,
                "d": f"d_{i}",
                "event_name_1": event_name,
                "event_type_1": event_type,
                "event_name_2": None,
                "event_type_2": None,
                "snap_CA": int(dt.day <= 10),
                "snap_TX": int(5 <= dt.day <= 14),
                "snap_WI": int(dt.day % 7 == 0),
            }
        )
    calendar = pd.DataFrame(calendar_rows)

    sales_rows = []
    price_rows = []
    for item_id, dept_id, cat_id, base, zero_p in catalog:
        for store_id, state_id in stores:
            store_factor = 1.15 if store_id.startswith("CA") else 0.9
            price = {
                "HOBBIES": 8.99,
                "HOUSEHOLD": 6.49,
                "FOODS": 2.79,
            }[cat_id]
            series_id = f"{item_id}_{store_id}_validation"
            row = {
                "id": series_id,
                "item_id": item_id,
                "dept_id": dept_id,
                "cat_id": cat_id,
                "store_id": store_id,
                "state_id": state_id,
            }
            demand = []
            for i, dt in enumerate(dates):
                dow = dt.weekday()
                weekend = 1.25 if dow >= 5 else 1.0
                month_season = 1.15 if dt.month in (11, 12) else (0.9 if dt.month in (1, 2) else 1.0)
                event_boost = 1.4 if calendar.loc[i, "event_name_1"] else 1.0
                noise = rng.normal(0, base * 0.12)
                value = max(0.0, base * store_factor * weekend * month_season * event_boost + noise)
                if rng.random() < zero_p:
                    value = 0.0
                demand.append(int(round(value)))
                row[f"d_{i + 1}"] = demand[-1]
                if i == 0 or calendar.loc[i, "wm_yr_wk"] != calendar.loc[i - 1, "wm_yr_wk"]:
                    weekly_price = round(price * (0.95 if dt.month in (1, 7) else 1.0), 2)
                    price_rows.append(
                        {
                            "store_id": store_id,
                            "item_id": item_id,
                            "wm_yr_wk": int(calendar.loc[i, "wm_yr_wk"]),
                            "sell_price": weekly_price,
                        }
                    )
            sales_rows.append(row)

    sales = pd.DataFrame(sales_rows)
    prices = pd.DataFrame(price_rows).drop_duplicates(["store_id", "item_id", "wm_yr_wk"])

    sales.to_csv(output_dir / "sales_train_validation.csv", index=False)
    calendar.to_csv(output_dir / "calendar.csv", index=False)
    prices.to_csv(output_dir / "sell_prices.csv", index=False)

    metadata = {
        "synthetic": True,
        "label": "demonstration data",
        "n_days": n_days,
        "n_series": len(sales),
        "seed": seed,
        "warning": "This dataset is synthetic demonstration data, not the official M5 competition files.",
    }
    (output_dir / "DATA_SOURCE.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote synthetic M5-shaped files to %s", output_dir)
    return metadata


def load_m5_tables(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales_path = source_dir / "sales_train_validation.csv"
    calendar_path = source_dir / "calendar.csv"
    prices_path = source_dir / "sell_prices.csv"
    _require_file(sales_path, "Place M5 sales_train_validation.csv in this directory, or generate sample data.")
    _require_file(calendar_path, "Place M5 calendar.csv in this directory.")
    _require_file(prices_path, "Place M5 sell_prices.csv in this directory.")

    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(prices_path)
    _require_columns(sales, SALES_REQUIRED, "sales_train_validation.csv")
    _require_columns(calendar, CALENDAR_REQUIRED, "calendar.csv")
    _require_columns(prices, PRICES_REQUIRED, "sell_prices.csv")
    return sales, calendar, prices


def wide_to_long(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    store_filter: str | None = None,
    max_items: int | None = None,
) -> pd.DataFrame:
    if store_filter:
        sales = sales[sales["store_id"] == store_filter].copy()
    if max_items and max_items > 0:
        item_ids = sales["item_id"].drop_duplicates().head(max_items)
        sales = sales[sales["item_id"].isin(item_ids)].copy()

    id_vars = [col for col in SALES_REQUIRED if col in sales.columns]
    value_vars = [col for col in sales.columns if str(col).startswith("d_")]
    if not value_vars:
        raise DataValidationError("sales file has no d_* demand columns")

    long_df = sales.melt(id_vars=id_vars, value_vars=value_vars, var_name="d", value_name="demand")
    calendar = calendar.copy()
    calendar["date"] = pd.to_datetime(calendar["date"])
    long_df = long_df.merge(calendar, on="d", how="left")
    if long_df["date"].isna().any():
        missing = int(long_df["date"].isna().sum())
        raise DataValidationError(f"Date alignment failed for {missing} sales rows. Check calendar d keys.")

    prices = prices.copy()
    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype(int)
    long_df["wm_yr_wk"] = long_df["wm_yr_wk"].astype(int)
    long_df = long_df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long_df["sell_price"] = long_df.groupby(["store_id", "item_id"], observed=False)["sell_price"].ffill()
    long_df["sell_price"] = long_df.groupby(["store_id", "item_id"], observed=False)["sell_price"].bfill()
    long_df["sell_price"] = long_df["sell_price"].fillna(long_df["sell_price"].median())
    long_df["demand"] = pd.to_numeric(long_df["demand"], errors="coerce").fillna(0).clip(lower=0)
    long_df["product_code"] = long_df["item_id"].astype(str) + "_" + long_df["store_id"].astype(str)
    long_df["event_name_1"] = long_df.get("event_name_1", pd.Series(index=long_df.index)).fillna("")
    long_df["event_type_1"] = long_df.get("event_type_1", pd.Series(index=long_df.index)).fillna("")
    long_df["is_event"] = (long_df["event_name_1"].astype(str).str.len() > 0).astype(int)
    long_df["weekday_num"] = long_df["date"].dt.dayofweek
    long_df = long_df.sort_values(["product_code", "date"]).reset_index(drop=True)
    long_df["is_synthetic"] = 0
    return long_df


def label_synthetic(df: pd.DataFrame, synthetic: bool) -> pd.DataFrame:
    out = df.copy()
    out["is_synthetic"] = int(synthetic)
    out["data_label"] = "demonstration data" if synthetic else "m5"
    return out


def save_processed(df: pd.DataFrame, settings: Settings | None = None, synthetic: bool = False) -> Path:
    settings = settings or get_settings()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    path = settings.daily_sales_path
    labeled = label_synthetic(df, synthetic)
    labeled.to_parquet(path, index=False)
    metadata = {
        "synthetic": synthetic,
        "label": "demonstration data" if synthetic else "m5",
        "rows": int(len(labeled)),
        "products": int(labeled["product_code"].nunique()),
        "min_date": str(labeled["date"].min().date()),
        "max_date": str(labeled["date"].max().date()),
        "path": str(path),
    }
    settings.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def ingest(settings: Settings | None = None, use_sample: bool | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    use_sample = settings.SAMPLE_DATA_MODE if use_sample is None else use_sample
    source_dir = settings.sample_dir if use_sample else settings.raw_dir
    if use_sample and not (source_dir / "sales_train_validation.csv").exists():
        generate_synthetic_m5(source_dir)

    sales, calendar, prices = load_m5_tables(source_dir)
    store_filter = settings.M5_STORE_FILTER or None
    max_items = settings.M5_MAX_ITEMS if not use_sample else None
    long_df = wide_to_long(sales, calendar, prices, store_filter=store_filter, max_items=max_items)
    long_df = label_synthetic(long_df, synthetic=use_sample)
    save_processed(long_df, settings, synthetic=use_sample)
    if use_sample:
        long_df.to_parquet(settings.sample_sales_path, index=False)
    return long_df


def load_processed(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path = settings.daily_sales_path
    if not path.exists():
        logger.info("Processed sales missing; ingesting data")
        return ingest(settings)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def filter_history(
    df: pd.DataFrame,
    product_code: str | None = None,
    store_code: str | None = None,
    category: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    out = df
    if product_code:
        out = out[out["product_code"] == product_code]
    if store_code:
        out = out[out["store_id"] == store_code]
    if category:
        out = out[out["cat_id"] == category]
    if start_date:
        out = out[out["date"] >= pd.to_datetime(start_date)]
    if end_date:
        out = out[out["date"] <= pd.to_datetime(end_date)]
    return out.sort_values(["product_code", "date"])
