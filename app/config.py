"""Application configuration."""

from pydantic import BaseModel


class Settings(BaseModel):
    """Static application settings."""

    app_name: str = "OCR FastAPI"
    app_version: str = "0.1.0"
    debug: bool = False


settings = Settings()
