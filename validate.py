# validate.py
# Validates uploaded images before sending to the ML model.
# Two checks:
#   1. Blur detection — rejects blurry or out-of-focus images
#   2. Confidence gate — rejects predictions that are too uncertain

import logging

# cv2 is wrapped in try/except because it links against NumPy C extensions.
# If the installed NumPy version is incompatible (e.g. NumPy 2.x with cv2 built
# against NumPy 1.x), the import raises an ImportError/AttributeError at runtime.
# Wrapping here means the app keeps running even if cv2 fails — the blur check
# is simply skipped with a warning.
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False
    logging.warning(
        "[validate] cv2 (opencv-python) could not be imported. "
        "Blur check will be skipped. Ensure numpy==1.24.3 and opencv-python==4.8.1.78."
    )


def check_blur(image_path: str) -> tuple[bool, str]:
    """
    Checks if an image is too blurry for reliable disease detection.

    Uses the Laplacian variance method:
    - Converts image to grayscale
    - Applies Laplacian operator to measure edge sharpness
    - A low variance means the image is blurry (few sharp edges)

    Threshold: 100 — anything below is considered too blurry.

    If cv2 is unavailable, the check is skipped and the image passes.

    Returns:
        (True, "") if image is sharp enough (or cv2 unavailable)
        (False, error_message) if image is too blurry
    """
    # Skip blur check gracefully if cv2 failed to import
    if not CV2_AVAILABLE:
        logging.warning("[validate] Skipping blur check — cv2 not available.")
        return True, ""

    # Load image in grayscale mode for edge detection
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return False, "Could not read the image file. Please upload a valid image."

    # Compute Laplacian and measure its variance — low variance = blurry
    laplacian_variance = cv2.Laplacian(image, cv2.CV_64F).var()

    if laplacian_variance < 100:
        return False, "Image is unclear — please capture a closer leaf photo."

    return True, ""


def check_confidence(confidence: float) -> tuple[bool, str]:
    """
    Checks if the model's confidence is high enough to trust the prediction.

    If the model is less than 60% confident, the image may be:
    - A non-crop image
    - A very unusual disease presentation
    - A poor quality photo

    Args:
        confidence: prediction confidence as a percentage (0–100)

    Returns:
        (True, "") if confidence is acceptable
        (False, error_message) if confidence is too low
    """
    if confidence < 60.0:
        return False, "Confidence too low — please re-upload a clearer image."

    return True, ""
