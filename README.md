# OCR FastAPI

A small production-style computer vision API built with FastAPI, OpenCV, and Tesseract OCR. The service accepts an uploaded image, applies light preprocessing, extracts text with `pytesseract`, and returns the result as JSON.

## Architecture Overview

The project is organized around a clean service-oriented structure:

- `app/main.py`: FastAPI application setup and health endpoint
- `app/routes/ocr.py`: HTTP route and request validation
- `app/services/ocr_service.py`: image decoding and OCR execution
- `app/utils/image_preprocess.py`: preprocessing pipeline with OpenCV
- `app/schemas.py`: response models
- `tests/test_ocr.py`: API tests

## OCR Pipeline

1. Receive an uploaded image through `POST /ocr`
2. Convert the upload bytes into an OpenCV image
3. Convert to grayscale
4. Apply Gaussian blur and binary thresholding
5. Extract text with `pytesseract.image_to_string()`
6. Return extracted text as JSON

## Installation

### Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### System dependency

Tesseract OCR must be installed on the host machine and available on `PATH`.

For Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

## Running the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

FastAPI automatically provides Swagger documentation at:

- `/docs`

## Example Request

```bash
curl -X POST "http://localhost:8000/ocr" \
-F "file=@sample.png"
```

## Example Response

```json
{
  "text": "Hello world from OCR",
  "confidence": null
}
```

## Testing

```bash
pytest
```

## Docker

Build and run the container:

```bash
docker build -t ocr-fastapi .
docker run -p 8000:8000 ocr-fastapi
```

## Future Improvements

- Return word-level or document-level confidence scores
- Support multiple OCR languages and configurable Tesseract settings
- Add structured logging and request tracing
- Improve preprocessing for noisy or skewed images
- Add asynchronous background processing for larger workloads
