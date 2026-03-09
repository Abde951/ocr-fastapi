"""Image preprocessing helpers for OCR."""

import cv2
import numpy as np


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Prepare an image for OCR using grayscale, blur, and thresholding."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, thresholded = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return thresholded
