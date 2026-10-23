from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: str = "sqlite:///./data/processed/forecasting.db"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODEL_DIR: Path = PROJECT_ROOT / "artifacts"
    API_BASE_URL: str = "http://127.0.0.1:8000"

    LLM_PROVIDER: str = "openrouter"
    LLM_MODEL: str = "openai/gpt-4o-mini"
    LLM_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_SITE_URL: str = "http://localhost:8501"
    OPENROUTER_APP_NAME: str = "Agentic Demand Forecasting"

    LOG_LEVEL: str = "INFO"
    FORECAST_HORIZON: int = 7
    SAMPLE_DATA_MODE: bool = True
    M5_STORE_FILTER: str = ""
    M5_MAX_ITEMS: int = 50

    REQUEST_BODY_LIMIT_BYTES: int = 2_000_000
    MIN_HISTORY_DAYS: int = 28

    @field_validator("DATA_DIR", "MODEL_DIR", mode="before")
    @classmethod
    def resolve_paths(cls, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @property
    def raw_dir(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.DATA_DIR / "processed"

    @property
    def sample_dir(self) -> Path:
        return self.DATA_DIR / "sample"

    @property
    def daily_sales_path(self) -> Path:
        return self.processed_dir / "daily_sales.parquet"

    @property
    def sample_sales_path(self) -> Path:
        return self.sample_dir / "daily_sales.parquet"

    @property
    def metadata_path(self) -> Path:
        return self.processed_dir / "dataset_metadata.json"

    @property
    def sqlite_path(self) -> Path | None:
        if self.DATABASE_URL.startswith("sqlite"):
            raw = self.DATABASE_URL.split("///", 1)[-1]
            path = Path(raw)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            return path
        return None

    @property
    def sqlalchemy_url(self) -> str:
        if self.sqlite_path is not None:
            return f"sqlite:///{self.sqlite_path.as_posix()}"
        return self.DATABASE_URL

    @property
    def llm_configured(self) -> bool:
        return bool(self.LLM_API_KEY.strip())


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.sample_dir.mkdir(parents=True, exist_ok=True)
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if settings.sqlite_path:
        settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
