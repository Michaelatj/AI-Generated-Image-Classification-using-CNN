"""Inference helpers for the selected Custom CNN v3 deployment model."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_DIR = ARTIFACTS_DIR / "configs"

# Deployment and preprocessing values come from the v3 deployment configs.
with (CONFIG_DIR / "deployment_config_v3.json").open(encoding="utf-8") as file:
    DEPLOYMENT_CONFIG = json.load(file)
with (CONFIG_DIR / "preprocessing_config.json").open(encoding="utf-8") as file:
    PREPROCESSING_CONFIG = json.load(file)
with (CONFIG_DIR / "label_mapping.json").open(encoding="utf-8") as file:
    LABEL_MAPPING_CONFIG = json.load(file)

MODEL_NAME = DEPLOYMENT_CONFIG["default_model"]
MODEL_CONFIG = DEPLOYMENT_CONFIG["models"][MODEL_NAME]
IMAGE_SIZE = int(DEPLOYMENT_CONFIG["preprocessing"]["image_size"])
CENTER_SQUARE_CROP = bool(DEPLOYMENT_CONFIG["preprocessing"]["center_square_crop"])
COLOR_MODE = PREPROCESSING_CONFIG["shared_image_input"]["color_mode"]
RESIZE_INTERPOLATION = Image.Resampling[
    PREPROCESSING_CONFIG["shared_image_input"]["resize_interpolation"].upper()
]
CLASS_NAMES = {
    int(label): name for label, name in LABEL_MAPPING_CONFIG["Label_A"].items()
}
AI_THRESHOLD = float(MODEL_CONFIG["threshold"])
MAX_IMAGE_PIXELS = 40_000_000

if MODEL_NAME != "Custom CNN":
    raise ValueError(f"Unexpected deployment model: {MODEL_NAME}")
if CLASS_NAMES != {0: "Real", 1: "AI-Generated"}:
    raise ValueError("The deployment label mapping does not match the binary model.")

MODEL_PATH = ARTIFACTS_DIR / MODEL_CONFIG["file"]
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Deployment model was not found: {MODEL_PATH}")

# Custom CNN v3 has no custom layers; its Rescaling(1/255) layer is embedded.
MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)


@dataclass(frozen=True)
class Prediction:
    """A binary prediction and the sigmoid probability produced by the model."""

    label: str
    confidence: float
    probability_ai_generated: float


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prepare one image exactly as the v3 evaluation pipeline did.

    Converts to RGB, center-crops to a square, and resizes with Pillow bilinear
    interpolation to 224x224. Returns raw uint8 RGB pixels; the model performs
    its own 1/255 rescaling. No training augmentation is applied.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input harus berupa file gambar yang valid.")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Ukuran gambar tidak valid.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            "Ukuran gambar terlalu besar. Gunakan gambar di bawah 40 megapiksel."
        )
    if COLOR_MODE != "RGB":
        raise ValueError(f"Unsupported configured color mode: {COLOR_MODE}")

    prepared = image.convert("RGB")
    if CENTER_SQUARE_CROP:
        width, height = prepared.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        prepared = prepared.crop((left, top, left + side, top + side))

    prepared = prepared.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        resample=RESIZE_INTERPOLATION,
    )
    return np.asarray(prepared, dtype=np.uint8)


def predict_image(image: Image.Image | None) -> Prediction:
    """Predict Real or AI-generated for one image using the loaded model."""
    if image is None:
        raise ValueError("Silakan unggah gambar terlebih dahulu.")

    image_array = preprocess_image(image)
    batch = np.expand_dims(image_array, axis=0)
    raw_output = np.asarray(MODEL.predict(batch, verbose=0))
    if raw_output.shape != (1, 1):
        raise RuntimeError(
            f"Expected one sigmoid output, received shape {raw_output.shape}."
        )

    probability_ai = float(raw_output[0, 0])
    if not math.isfinite(probability_ai) or not 0.0 <= probability_ai <= 1.0:
        raise RuntimeError("The model returned an invalid probability.")

    if probability_ai >= AI_THRESHOLD:
        return Prediction(
            label=CLASS_NAMES[1],
            confidence=probability_ai,
            probability_ai_generated=probability_ai,
        )
    return Prediction(
        label=CLASS_NAMES[0],
        confidence=1.0 - probability_ai,
        probability_ai_generated=probability_ai,
    )


def predict_uploaded_file(image_path: str | None) -> Prediction:
    """Open and predict a Gradio-uploaded image path with friendly input checks."""
    if not image_path:
        raise ValueError("Silakan unggah gambar terlebih dahulu.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(image_path) as image:
                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise ValueError(
                        "Ukuran gambar terlalu besar. Gunakan gambar di bawah 40 megapiksel."
                    )
                image.load()
                return predict_image(image)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError(
            "File tidak dapat dibaca sebagai gambar. Coba PNG, JPG, atau WEBP."
        ) from error
    except Image.DecompressionBombWarning as error:
        raise ValueError(
            "Ukuran gambar terlalu besar untuk diproses dengan aman."
        ) from error
