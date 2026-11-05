import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_INSECURE_DEFAULT_SECRET = "insecure-dev-secret-change-me"  # noqa: S105


class Settings(BaseSettings):
    """Application settings, loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    app_name: str = "Data Analytics Platform"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production", "test"] = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    docs_enabled: bool = True

    # Security
    secret_key: str = _INSECURE_DEFAULT_SECRET
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    access_token_expire_minutes: int = Field(default=30, gt=0)
    refresh_token_expire_minutes: int = Field(default=60 * 24 * 7, gt=0)
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    db_echo: bool = False
    db_pool_size: int = Field(default=10, gt=0)
    db_max_overflow: int = Field(default=20, ge=0)
    auto_create_tables: bool = False

    # Storage & analytics
    storage_dir: Path = Path("./data/datasets")
    max_upload_size_mb: int = Field(default=100, gt=0)
    max_result_rows: int = Field(default=10_000, gt=0)

    # DuckDB analytics engine. Unset values use DuckDB's defaults
    # (all cores, 80% of RAM, spilling under STORAGE_DIR/.duckdb_tmp).
    duckdb_threads: int | None = Field(default=None, gt=0)
    duckdb_memory_limit: str | None = None
    duckdb_temp_dir: Path | None = None

    # AI analyst (Claude API)
    # When unset, the Anthropic SDK resolves credentials itself
    # (ANTHROPIC_API_KEY env var, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile).
    anthropic_api_key: SecretStr | None = None
    ai_model: str = "claude-opus-5-5"
    ai_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    ai_max_tokens: int = Field(default=16_000, gt=0)
    ai_max_tool_rounds: int = Field(default=4, ge=1, le=10)
    ai_max_tool_rows: int = Field(default=200, gt=0)
    ai_timeout_seconds: float = Field(default=120.0, gt=0)
    ai_strict_tools: bool = True
    ai_fallbacks_enabled: bool = True
    # Answer with a keyword matcher instead of Claude: no credentials and no API costs.
    ai_demo_mode: bool = False

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> Any:
        """Accept either a JSON list or a comma-separated string."""
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("["):
                return json.loads(value)
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @model_validator(mode="after")
    def _check_production_safety(self) -> "Settings":
        if self.environment == "production":
            if self.secret_key == _INSECURE_DEFAULT_SECRET or len(self.secret_key) < 32:
                raise ValueError(
                    "SECRET_KEY must be a random value of at least 32 characters in production"
                )
            if self.debug:
                raise ValueError("DEBUG must be disabled in production")
            if "*" in self.cors_origins:
                raise ValueError("Wildcard CORS origins are not allowed in production")
            if self.ai_demo_mode:
                raise ValueError("AI_DEMO_MODE must be disabled in production")
        return self

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()
