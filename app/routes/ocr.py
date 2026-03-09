"""OCR API routes."""

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas import OCRResponse
from app.services.ocr_service import extract_text_from_upload

router = APIRouter(tags=["ocr"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}


@router.post("/ocr", response_model=OCRResponse, summary="Extract text from an image")
async def run_ocr(file: UploadFile = File(...)) -> OCRResponse:
    """Accept an uploaded image and return extracted OCR text."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Upload a valid image.",
        )

    text = await extract_text_from_upload(file)
    return OCRResponse(text=text, confidence=None)
