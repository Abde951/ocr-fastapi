"""Application configuration."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="OCR FastAPI", validation_alias="OCR_APP_NAME")
    app_version: str = Field(default="0.1.0", validation_alias="OCR_APP_VERSION")
    debug: bool = Field(default=False, validation_alias="OCR_DEBUG")
    google_application_credentials: str | None = Field(
        default=None,
        validation_alias="GOOGLE_APPLICATION_CREDENTIALS",
    )


settings = Settings()
