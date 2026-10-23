from backend.app.database import init_db
from backend.app.ml.data_ingestion import ingest
from backend.app.ml.data_store import data_store
from backend.app.services.catalog import sync_products
from backend.app.database import SessionLocal
from backend.app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    init_db()
    ingest(settings, use_sample=settings.SAMPLE_DATA_MODE)
    data_store.load(force=True)
    db = SessionLocal()
    try:
        n = sync_products(db)
        print(f"Database initialized. Synced {n} products.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
