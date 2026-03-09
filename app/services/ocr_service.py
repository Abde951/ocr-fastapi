"""OCR service implementation."""

from __future__ import annotations

from fastapi import HTTPException, UploadFile, status
import cv2
import numpy as np
import pytesseract

from app.utils.image_preprocess import preprocess_image


async def extract_text_from_upload(file: UploadFile) -> str:
    """Read an uploaded image, preprocess it, and run OCR."""
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
    extracted_text = pytesseract.image_to_string(processed_image)
    return extracted_text.strip()
