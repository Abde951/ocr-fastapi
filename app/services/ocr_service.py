"""OCR service implementation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.config import settings
from app.utils.image_preprocess import preprocess_image


@lru_cache(maxsize=1)
def get_vision_client() -> Any:
    """Create and cache the Google Cloud Vision client."""
    try:
        from google.cloud import vision
        from google.oauth2 import service_account
    except ModuleNotFoundError as exc:
        raise RuntimeError("google-cloud-vision is not installed.") from exc

    credentials_path = settings.google_application_credentials
    if credentials_path:
        credentials = service_account.Credentials.from_service_account_file(
            Path(credentials_path),
        )
        return vision.ImageAnnotatorClient(credentials=credentials)

    return vision.ImageAnnotatorClient()


def _encode_processed_image(image: np.ndarray) -> bytes:
    """Encode the processed image into PNG bytes for Vision API requests."""
    ok, encoded_image = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to prepare the image for OCR.",
        )

    return encoded_image.tobytes()


def build_vision_image(content: bytes) -> Any:
    """Create a Vision API image payload lazily to avoid import-time hard dependency."""
    try:
        from google.cloud import vision
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="google-cloud-vision is not installed.",
        ) from exc

    return vision.Image(content=content)


def _extract_confidence(annotation: Any) -> float | None:
    """Compute an aggregate confidence score from word annotations when available."""
    if annotation is None:
        return None

    confidences: list[float] = []
    for page in annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    if word.confidence is not None:
                        confidences.append(float(word.confidence))

    if not confidences:
        return None

    return round(sum(confidences) / len(confidences), 4)


async def extract_text_from_upload(file: UploadFile) -> tuple[str, float | None]:
    """Read an uploaded image, preprocess it, and run OCR with Google Cloud Vision."""
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to decode the uploaded image.",
        )

    processed_image = preprocess_image(image)
    image_bytes = _encode_processed_image(processed_image)

    try:
        response = get_vision_client().document_text_detection(
            image=build_vision_image(image_bytes),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    # except Exception as exc:
    #     raise HTTPException(
    #         status_code=status.HTTP_502_BAD_GATEWAY,
    #         detail="Google Cloud Vision OCR request failed.",
    #     ) from exc

    if response.error.message:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Google Cloud Vision error: {response.error.message}",
        )

    extracted_text = response.full_text_annotation.text.strip()
    confidence = _extract_confidence(response.full_text_annotation)
    return extracted_text, confidence
