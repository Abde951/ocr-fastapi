"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.config import settings
from app.routes.ocr import router as ocr_router

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    description="A small production-style OCR API built with FastAPI and Google Cloud Vision.",
)

app.include_router(ocr_router)


@app.get("/health", tags=["health"], summary="Health check")
async def health_check() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}
