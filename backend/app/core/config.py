"""
Application configuration using Pydantic Settings.
All configuration is driven by environment variables (12-factor app methodology).
"""

import json
from functools import lru_cache
from typing import Any, List, Optional

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, EnvSettingsSource, SettingsConfigDict
from pydantic_settings.sources.providers.dotenv import DotEnvSettingsSource


def _make_comma_list_mixin():
    """Return a mixin class that handles comma-separated list env values."""

    class _Mixin:
        def decode_complex_value(
            self, field_name: str, field_info: Any, value: Any
        ) -> Any:
            try:
                return super().decode_complex_value(field_name, field_info, value)
            except Exception:
                if isinstance(value, str):
                    return [v.strip() for v in value.split(",") if v.strip()]
                return value

    return _Mixin


_Mixin = _make_comma_list_mixin()


class _CommaListEnvSource(_Mixin, EnvSettingsSource):
    """EnvSettingsSource that falls back to comma-split for List fields."""


class _CommaListDotEnvSource(_Mixin, DotEnvSettingsSource):
    """DotEnvSettingsSource that falls back to comma-split for List fields."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "IntelliCover AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"  # development | production

    # ── Security ─────────────────────────────────────────────────────────────
    SECRET_KEY: str = "changeme_super_secret_key_for_development_only"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    ALGORITHM: str = "HS256"

    # ── CORS ─────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

    # ── Database (PostgreSQL) ─────────────────────────────────────────────────
    DATABASE_URL: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/intellicover"
    )
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # ── Redis / Celery ────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ── AWS S3 (File Storage) ─────────────────────────────────────────────────
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "ap-south-1"
    S3_BUCKET_NAME: str = "intellicover-resumes"
    USE_LOCAL_STORAGE: bool = True  # True = save files locally during dev
    LOCAL_UPLOAD_DIR: str = "uploads"

    # ── NLP / AI Models ───────────────────────────────────────────────────────
    SPACY_MODEL: str = Field(...)
    NER_MODEL: str = "dslim/bert-base-NER"
    SENTENCE_TRANSFORMER_MODEL: str = Field(...)
    SUGGESTION_LLM_MODEL: str = Field(...)
    COVER_LETTER_BASE_MODEL: str = Field(...)
    COVER_LETTER_LORA_ADAPTER: str = Field(...)

    # ── Model inference settings ──────────────────────────────────────────────
    LLM_MAX_NEW_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 0.9
    # 0 means "no truncation" (use full JD text in prompt context).
    LLM_JD_CONTEXT_MAX_CHARS: int = 0
    # 0 means "no truncation" (use full parsed resume context in prompts).
    LLM_RESUME_CONTEXT_MAX_CHARS: int = 0
    USE_GPU: bool = False  # set to True in prod with CUDA
    USE_QUANTIZATION: bool = True  # 4-bit quantization via bitsandbytes

    # ── Hugging Face ──────────────────────────────────────────────────────────
    HUGGINGFACE_TOKEN: Optional[str] = None
    HF_HOME: str = "./model_cache"  # where model weights are cached

    # ── Web Scraper ───────────────────────────────────────────────────────────
    SCRAPER_TIMEOUT_SECONDS: int = 30
    SCRAPER_USE_PLAYWRIGHT_FALLBACK: bool = True
    # Preserve as much scraped JD text as possible for downstream LLM prompting.
    # 0 means "no truncation".
    JD_RAW_TEXT_MAX_CHARS: int = 0
    # Preserve as much parsed resume raw text as possible for downstream LLM prompts.
    # 0 means "no truncation".
    RESUME_RAW_TEXT_MAX_CHARS: int = 0

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 20

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # "json" | "pretty"
    LOG_DIR: str = "logs"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        **kwargs,
    ):
        """Replace default sources with comma-list-aware versions."""
        return (
            init_settings,
            _CommaListEnvSource(settings_cls),
            _CommaListDotEnvSource(settings_cls),
        )


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings singleton.
    Using lru_cache ensures the .env file is only read once.
    """
    return Settings()


# Module-level convenience accessor
settings = get_settings()
