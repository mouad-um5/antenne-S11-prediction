from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    environment: str
    cors_origins: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    enable_docs: bool
    require_verified_model_scope: bool

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Resolve from the backend directory, regardless of the launch directory.
    # Explicit environment variables (including Docker) keep precedence.
    load_dotenv(ENV_FILE, override=False, encoding="utf-8-sig")
    environment = os.getenv("APP_ENV", "development").strip().lower()
    if environment not in {"development", "test", "production"}:
        raise ValueError("APP_ENV doit valoir development, test ou production")

    default_hosts = "localhost,127.0.0.1,testserver"
    return Settings(
        environment=environment,
        cors_origins=_split_csv(
            os.getenv(
                "CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            )
        ),
        allowed_hosts=_split_csv(os.getenv("ALLOWED_HOSTS", default_hosts)),
        enable_docs=_as_bool("ENABLE_API_DOCS", not environment == "production"),
        require_verified_model_scope=_as_bool(
            "REQUIRE_VERIFIED_MODEL_SCOPE",
            environment == "production",
        ),
    )
