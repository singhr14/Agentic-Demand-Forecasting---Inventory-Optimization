from backend.app.core.config import get_settings
from backend.app.ml.data_store import data_store
from backend.app.ml.lightgbm_model import LightGBMForecaster


def main() -> None:
    settings = get_settings()
    df = data_store.load()
    model = LightGBMForecaster(settings.MODEL_DIR)
    meta = model.fit(df, horizon=settings.FORECAST_HORIZON)
    print("Trained LightGBM. Metadata:")
    for key, value in meta.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
