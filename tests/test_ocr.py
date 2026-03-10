"""Tests for OCR endpoints."""

from dataclasses import dataclass
from io import BytesIO
from types import SimpleNamespace

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


@dataclass
class FakeWord:
    confidence: float | None


@dataclass
class FakeParagraph:
    words: list[FakeWord]


@dataclass
class FakeBlock:
    paragraphs: list[FakeParagraph]


@dataclass
class FakePage:
    blocks: list[FakeBlock]


@dataclass
class FakeAnnotation:
    text: str
    pages: list[FakePage]


@pytest.mark.parametrize(
    ("ocr_output", "confidence"),
    [("Hello world from OCR", 0.85), ("", None)],
)
@pytest.mark.anyio
async def test_ocr_endpoint_returns_ocr_response(
    monkeypatch: pytest.MonkeyPatch,
    ocr_output: str,
    confidence: float | None,
) -> None:
    """OCR endpoint returns extracted text from the OCR service."""

    words = [] if confidence is None else [FakeWord(confidence=confidence), FakeWord(confidence=confidence)]
    annotation = FakeAnnotation(
        text=ocr_output,
        pages=[FakePage(blocks=[FakeBlock(paragraphs=[FakeParagraph(words=words)])])],
    )

    class FakeVisionClient:
        def document_text_detection(self, *, image: object) -> SimpleNamespace:
            assert image is not None
            return SimpleNamespace(
                full_text_annotation=annotation,
                error=SimpleNamespace(message=""),
            )

    monkeypatch.setattr("app.services.ocr_service.get_vision_client", lambda: FakeVisionClient())
    monkeypatch.setattr("app.services.ocr_service.build_vision_image", lambda content: content)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/ocr",
            files={"file": ("sample.png", create_test_image(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json() == {"text": ocr_output, "confidence": confidence}
