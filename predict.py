# predict.py
# Handles loading the EfficientNet-B3 model and running inference on a crop leaf image.
# The model was trained on the PlantVillage dataset covering Tomato, Apple, and Grape.
#
# NOTE on model loading strategy:
# The model was trained in Google Colab with TF 2.16+ (Keras 3), but this app runs
# TF 2.15 (Keras 2). Rather than patching the Keras 3 config JSON (which has many
# serialisation differences), we rebuild the exact same architecture in code and call
# load_weights() to populate it. This completely bypasses config deserialisation.

import json
import os
import numpy as np
from PIL import Image

# TensorFlow import — wrapped in try/except so the app starts cleanly even if
# TensorFlow is not installed. If TF is unavailable, inference is disabled and
# every call to predict_disease() returns a structured error dict instead of crashing.
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    tf = None
    TF_AVAILABLE = False
    print("[predict] TensorFlow not available. ML inference will be disabled.")

# Model and class label file paths (relative to project root)
MODEL_PATH = "model/agrivision_efficientnetb3_final.h5"
CLASS_INDICES_PATH = "model/class_indices.json"

# Module-level variables — set once at startup, reused on every request
_model = None
_class_names = None  # Dict: {integer_index: "ClassName___label"}


def load_model():
    """
    Builds the EfficientNetB3 architecture and loads trained weights from disk.
    Called once at application startup by app.py.

    Strategy: rebuild the exact architecture from code (matching the Colab training
    notebook) then call load_weights(). This avoids all Keras 3 vs Keras 2 config
    deserialisation issues that occur with tf.keras.models.load_model().

    Returns True if successful, False otherwise.
    """
    global _model, _class_names

    if not TF_AVAILABLE:
        print("[predict] TensorFlow not installed. Cannot load model.")
        return False

    if not os.path.exists(MODEL_PATH):
        print(f"[predict] Model file not found at {MODEL_PATH}. "
              "Inference will be unavailable until model is added.")
        return False

    if not os.path.exists(CLASS_INDICES_PATH):
        print(f"[predict] Class indices file not found at {CLASS_INDICES_PATH}.")
        return False

    # --- Load class indices first (needed to know num_classes for the output layer) ---
    try:
        with open(CLASS_INDICES_PATH, "r") as f:
            class_indices = json.load(f)
        # Invert: {"Apple___Apple_scab": 0, ...} → {0: "Apple___Apple_scab", ...}
        _class_names = {v: k for k, v in class_indices.items()}
        num_classes = len(_class_names)
        print(f"[predict] Loaded {num_classes} class labels.")
    except Exception as e:
        print(f"[predict] Failed to load class indices: {e}")
        _class_names = None
        return False

    # --- Rebuild the exact architecture used in the Colab training notebook ---
    print("[predict] Building EfficientNetB3 architecture...")
    try:
        # EfficientNetB3 base — weights=None because we load trained weights below
        base = tf.keras.applications.EfficientNetB3(
            weights=None,
            include_top=False,
            input_shape=(300, 300, 3)
        )

        # Custom classification head (mirrors the Colab training code exactly)
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        _model = tf.keras.Model(inputs=base.input, outputs=out)

    except Exception as e:
        print(f"[predict] Failed to build model architecture: {e}")
        _model = None
        _class_names = None
        return False

    # --- Load the trained weights into the rebuilt architecture ---
    # load_weights() reads only the weight tensors from the H5 file, completely
    # skipping the Keras 3 model_config JSON that causes deserialisation errors.
    # by_name=True matches weights to layers by name — robust to layer counter
    # differences between the Colab session and this inference session.
    print("[predict] Loading trained weights from H5 file...")
    try:
        # by_name=False loads weights positionally (layer order), not by layer name.
        # The Colab training session numbered layers as dense_2/dense_3/batch_normalization_1
        # but our rebuilt model names them dense/dense_1/batch_normalization.
        # by_name=True would silently skip those mismatched layers leaving random weights.
        # by_name=False matches by order — safe because architecture is identical.
        _model.load_weights(MODEL_PATH, by_name=False)
        print("[predict] Model loaded successfully.")
        return True
    except Exception as e:
        print(f"[predict] Failed to load weights: {e}")
        _model = None
        _class_names = None
        return False


def is_model_loaded() -> bool:
    """Returns True if the model is loaded and ready for inference."""
    return _model is not None and _class_names is not None


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Loads and preprocesses a crop leaf image for EfficientNet-B3 inference.

    Pipeline (order matters):
    1. PIL open + convert to RGB — handles PNG, JPEG, RGBA, grayscale
    2. Resize to (300, 300) — EfficientNet-B3 input resolution
    3. Convert to float32 numpy array — shape (300, 300, 3)
    4. Add batch dimension — shape becomes (1, 300, 300, 3)
    5. Apply EfficientNet preprocess_input — scales pixel values to [-1, 1]

    Returns:
        numpy array of shape (1, 300, 300, 3) ready for model.predict()
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((300, 300))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)          # → (1, 300, 300, 3)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array


def predict_with_tta(model, image_path: str, n: int = 5) -> np.ndarray:
    """
    Test-Time Augmentation (TTA): runs inference on n augmented versions of
    the image and returns the averaged softmax probabilities.

    PlantVillage training images are clean lab shots on white backgrounds.
    Real-world field images have soil, shadows, multiple leaves, and varying
    lighting. TTA improves robustness on these without retraining by averaging
    predictions over small realistic variations of the same image.

    Augmentations applied (always includes the original):
        1. Original (no change)
        2. Horizontal flip
        3. Vertical flip
        4. Small rotation (+10°)
        5. Brightness boost (+20 pixel values, clipped at 255)

    Args:
        model: loaded Keras model
        image_path: path to the source image
        n: number of augmented versions to use (default 5, max 5)

    Returns:
        1-D numpy array of averaged class probabilities (length = num_classes)
    """
    img = Image.open(image_path).convert("RGB")
    img_300 = img.resize((300, 300))

    augments = [
        np.array(img_300, dtype=np.float32),                                   # 1: original
        np.array(img_300.transpose(Image.FLIP_LEFT_RIGHT), dtype=np.float32),  # 2: h-flip
        np.array(img_300.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.float32),  # 3: v-flip
        np.array(img_300.rotate(10), dtype=np.float32),                        # 4: rotate +10°
        np.clip(np.array(img_300, dtype=np.float32) + 20.0, 0, 255),           # 5: brighter
    ]

    all_preds = []
    for arr in augments[:n]:
        batch = np.expand_dims(arr, axis=0)
        batch = tf.keras.applications.efficientnet.preprocess_input(batch)
        pred = model.predict(batch, verbose=0)
        all_preds.append(pred[0])

    return np.mean(all_preds, axis=0)


# Maps crop_type dropdown value → expected class_name prefix
_CROP_PREFIXES = {
    "Apple":  "Apple___",
    "Grape":  "Grape___",
    "Tomato": "Tomato___",
}


def predict_disease(image_path: str, crop_type: str = None) -> dict:
    """
    Runs EfficientNet-B3 inference on a leaf image and returns a structured result.

    If crop_type is provided, validates that the top prediction belongs to that
    crop. Returns a clear error dict if the class prefix does not match — this
    prevents a "Tomato" user from getting an Apple disease result.

    Imported by app.py as `run_prediction` to avoid a naming collision with the
    FastAPI route handler that is also called predict_disease in that file.

    Returns:
        {
            "success": True | False,
            "disease": str  — readable name e.g. "Tomato Early Blight",
            "confidence": float — percentage e.g. 87.34,
            "class_name": str  — raw key e.g. "Tomato___Early_blight",
            "warning": str | None,
            "error": str | None
        }
    """
    if not is_model_loaded():
        return {
            "success": False,
            "disease": None,
            "confidence": 0.0,
            "class_name": None,
            "warning": None,
            "error": "Model not loaded yet. Please add model/best_model.h5"
        }

    try:
        # TTA: average predictions over 5 augmented versions for better
        # real-world accuracy on images that differ from clean lab training data
        avg_preds = predict_with_tta(_model, image_path)

        predicted_index = int(np.argmax(avg_preds))
        confidence = float(avg_preds[predicted_index]) * 100.0

        class_name = _class_names.get(predicted_index, "Unknown")

        # Crop type mismatch: reject if the predicted class belongs to a different crop
        if crop_type and crop_type in _CROP_PREFIXES:
            expected_prefix = _CROP_PREFIXES[crop_type]
            if not class_name.startswith(expected_prefix):
                return {
                    "success": False,
                    "disease": None,
                    "confidence": round(confidence, 2),
                    "class_name": class_name,
                    "warning": None,
                    "error": (
                        "Detected disease does not match selected crop type. "
                        "Please verify your crop selection or try a clearer image."
                    )
                }

        readable_disease = (
            class_name.replace("___", " ").replace("_", " ")
            if class_name != "Unknown"
            else "Unknown"
        )

        # Moderate confidence warning: 60–80% may indicate a real-world image
        # that looks different from the clean PlantVillage training set
        warning = None
        if 60.0 <= confidence < 80.0:
            warning = (
                "Moderate confidence — result may be less reliable on real-world images. "
                "For best results use a clear close-up leaf photo on a plain background."
            )

        return {
            "success": True,
            "disease": readable_disease,
            "confidence": round(confidence, 2),
            "class_name": class_name,
            "warning": warning,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "disease": None,
            "confidence": 0.0,
            "class_name": None,
            "warning": None,
            "error": f"Inference error: {str(e)}"
        }
