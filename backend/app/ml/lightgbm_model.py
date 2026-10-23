from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from backend.app.ml.features import FeaturePipeline, chronological_splits

logger = logging.getLogger(__name__)

MODEL_NAME = "lightgbm"
MODEL_VERSION = "1.0.0"


class LightGBMForecaster:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: lgb.Booster | None = None
        self.pipeline = FeaturePipeline()
        self.residual_std: float = 1.0
        self.metadata: dict = {}

    @property
    def model_path(self) -> Path:
        return self.model_dir / "lightgbm_model.txt"

    @property
    def pipeline_path(self) -> Path:
        return self.model_dir / "feature_pipeline.joblib"

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / "model_metadata.json"

    def fit(self, df: pd.DataFrame, horizon: int = 7) -> dict:
        train, valid, test = chronological_splits(df)
        if train.empty:
            raise ValueError("Training split is empty. Provide a longer history.")

        featured_train = self.pipeline.fit(train)
        X_train, y_train = self.pipeline.matrix(featured_train, dropna=True)
        if valid.empty:
            featured_valid = featured_train.iloc[0:0]
        else:
            featured_valid = self.pipeline.transform(pd.concat([train, valid], ignore_index=True))
            featured_valid = featured_valid[featured_valid["date"] >= valid["date"].min()]
        X_valid, y_valid = self.pipeline.matrix(featured_valid, dropna=True) if not featured_valid.empty else (None, None)

        dataset = lgb.Dataset(X_train, label=y_train)
        valid_sets = [dataset]
        valid_names = ["train"]
        if X_valid is not None and y_valid is not None and len(X_valid) > 0:
            valid_sets.append(lgb.Dataset(X_valid, label=y_valid, reference=dataset))
            valid_names.append("valid")

        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 10,
            "feature_fraction": 0.9,
            "verbosity": -1,
            "seed": 42,
        }
        callbacks = [lgb.log_evaluation(0)]
        if len(valid_sets) > 1:
            callbacks.insert(0, lgb.early_stopping(30, verbose=False))
        self.model = lgb.train(
            params,
            dataset,
            num_boost_round=250,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        train_pred = np.clip(self.model.predict(X_train), 0, None)
        self.residual_std = float(np.std(y_train.to_numpy() - train_pred)) or 1.0
        self.metadata = {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "horizon_default": horizon,
            "n_train": int(len(X_train)),
            "n_valid": int(len(X_valid) if X_valid is not None else 0),
            "n_test_rows": int(len(test)),
            "residual_std": self.residual_std,
            "feature_columns": self.pipeline.feature_columns,
            "train_end": str(train["date"].max().date()) if not train.empty else None,
            "valid_end": str(valid["date"].max().date()) if not valid.empty else None,
            "test_end": str(test["date"].max().date()) if not test.empty else None,
            "limitations": [
                "Uncertainty intervals use residual standard deviation from training errors, not calibrated prediction intervals.",
                "Recursive multi-step forecasts feed predicted values into lag features after the origin, so error compounds with horizon.",
                "Label encoding of product/store IDs cannot generalize to unseen IDs without retraining.",
            ],
        }
        self.save()
        logger.info("Trained LightGBM on %s rows", len(X_train))
        return self.metadata

    def save(self) -> None:
        if self.model is None:
            raise RuntimeError("No model to save")
        self.model.save_model(str(self.model_path))
        joblib.dump(self.pipeline, self.pipeline_path)
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"LightGBM artifact missing at {self.model_path}. Train models first.")
        self.model = lgb.Booster(model_file=str(self.model_path))
        self.pipeline = joblib.load(self.pipeline_path)
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.residual_std = float(self.metadata.get("residual_std", 1.0))

    def forecast(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if self.model is None:
            self.load()
        assert self.model is not None
        hist = history.sort_values("date").copy()
        if hist.empty:
            raise ValueError("No history available for LightGBM forecast")
        origin = pd.to_datetime(hist["date"].max())
        sigma = self.residual_std
        rows = []
        working = hist.copy()
        for step in range(1, horizon + 1):
            next_date = origin + timedelta(days=step)
            template = working.iloc[[-1]].copy()
            template["date"] = next_date
            template["demand"] = np.nan
            template["weekday_num"] = next_date.dayofweek
            template["month"] = next_date.month
            template["wday"] = next_date.dayofweek + 1
            template["is_event"] = 0
            combined = pd.concat([working, template], ignore_index=True)
            featured = self.pipeline.transform(combined)
            latest = featured.iloc[[-1]]
            X, _ = self.pipeline.matrix(latest, dropna=False)
            pred = float(max(0.0, self.model.predict(X)[0]))
            rows.append(
                {
                    "forecast_date": next_date.date(),
                    "predicted_demand": pred,
                    "lower_bound": max(0.0, pred - 1.96 * sigma),
                    "upper_bound": pred + 1.96 * sigma,
                    "model_name": MODEL_NAME,
                    "model_version": MODEL_VERSION,
                    "horizon_step": step,
                }
            )
            template["demand"] = pred
            working = pd.concat([working, template], ignore_index=True)
        return pd.DataFrame(rows)
