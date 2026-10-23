from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.core.config import get_settings


class Base(DeclarativeBase):
    pass


def _sqlite_connect_args(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"check_same_thread": False, "timeout": 30}
    return {}


settings = get_settings()
engine = create_engine(
    settings.sqlalchemy_url,
    connect_args=_sqlite_connect_args(settings.DATABASE_URL),
    future=True,
    echo=False,
)


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
    if settings.DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    from backend.app.models import entities  # noqa: F401

    Base.metadata.create_all(bind=engine)
