# OCR FastAPI

A small production-style computer vision API built with FastAPI, OpenCV, and Google Cloud Vision API. The service accepts an uploaded image, applies light preprocessing, sends it to Vision for OCR, and returns the result as JSON.

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
5. Send the processed image to `document_text_detection()`
6. Return extracted text and an aggregate confidence score as JSON

## Installation

### Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Google Cloud Vision setup

1. Create or select a Google Cloud project.
2. Enable the Vision API.
3. Create a service account with Vision access.
4. Download the service account JSON key.
5. Point `GOOGLE_APPLICATION_CREDENTIALS` at that file.

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
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
  "confidence": 0.9821
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
docker run -p 8000:8000 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcp-vision.json \
  -v /absolute/path/to/service-account.json:/run/secrets/gcp-vision.json:ro \
  ocr-fastapi
```

## Future Improvements

- Support multiple OCR feature modes such as dense document parsing
- Add request-level caching or rate limiting around Vision API usage
- Add structured logging and request tracing
- Improve preprocessing for noisy or skewed images
- Add asynchronous background processing for larger workloads
