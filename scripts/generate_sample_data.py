from backend.app.core.config import get_settings
from backend.app.ml.data_ingestion import generate_synthetic_m5, ingest


def main() -> None:
    settings = get_settings()
    meta = generate_synthetic_m5(settings.sample_dir)
    ingest(settings, use_sample=True)
    print("Generated labeled demonstration data:")
    print(meta)


if __name__ == "__main__":
    main()
