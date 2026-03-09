"""Tests for OCR endpoints."""

from io import BytesIO

import httpx
from PIL import Image
import pytest

from app.main import app


def create_test_image() -> bytes:
    """Create an in-memory PNG image for upload tests."""
    image = Image.new("RGB", (120, 60), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.anyio
async def test_health_endpoint() -> None:
    """Health endpoint returns expected payload."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_ocr_endpoint_rejects_non_image_upload() -> None:
    """OCR endpoint rejects unsupported file types."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/ocr",
            files={"file": ("sample.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type. Upload a valid image."


@pytest.mark.parametrize("ocr_output", ["Hello world from OCR", ""])
@pytest.mark.anyio
async def test_ocr_endpoint_returns_ocr_response(
    monkeypatch: pytest.MonkeyPatch,
    ocr_output: str,
) -> None:
    """OCR endpoint returns extracted text from the OCR service."""

    def mock_image_to_string(_image: object) -> str:
        return ocr_output

    monkeypatch.setattr("app.services.ocr_service.pytesseract.image_to_string", mock_image_to_string)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/ocr",
            files={"file": ("sample.png", create_test_image(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json() == {"text": ocr_output, "confidence": None}
