"""Pydantic schemas for API responses."""

from pydantic import BaseModel, Field


class OCRResponse(BaseModel):
    """OCR extraction response payload."""

    text: str = Field(..., description="Extracted text from the uploaded image.")
    confidence: float | None = Field(
        default=None,
        description="Optional aggregate OCR confidence score.",
    )
