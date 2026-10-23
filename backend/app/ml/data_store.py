from __future__ import annotations

import threading

import pandas as pd

from backend.app.core.config import get_settings
from backend.app.ml.data_ingestion import filter_history, load_processed


class DataStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._df: pd.DataFrame | None = None

    def load(self, force: bool = False) -> pd.DataFrame:
        with self._lock:
            if self._df is None or force:
                self._df = load_processed(get_settings())
            return self._df

    def products(self) -> pd.DataFrame:
        df = self.load()
        cols = ["product_code", "item_id", "dept_id", "cat_id", "store_id", "state_id", "is_synthetic"]
        return df[cols].drop_duplicates("product_code").sort_values("product_code")

    def history(self, product_code: str) -> pd.DataFrame:
        return filter_history(self.load(), product_code=product_code)

    def metadata(self) -> dict:
        df = self.load()
        return {
            "synthetic": bool(df["is_synthetic"].max()) if "is_synthetic" in df.columns else False,
            "label": "demonstration data" if ("is_synthetic" in df.columns and df["is_synthetic"].max()) else "m5",
            "rows": int(len(df)),
            "products": int(df["product_code"].nunique()),
            "min_date": str(df["date"].min().date()),
            "max_date": str(df["date"].max().date()),
            "stores": sorted(df["store_id"].unique().tolist()),
            "categories": sorted(df["cat_id"].unique().tolist()),
        }


data_store = DataStore()
