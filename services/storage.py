import os
import uuid
from typing import Optional

import cv2
import numpy as np


# =========================================================
# Storage config
# =========================================================

BASE_STORAGE_DIR = os.getenv("STORAGE_DIR", "/app/data")
IMAGES_DIR = os.path.join(BASE_STORAGE_DIR, "images")

os.makedirs(IMAGES_DIR, exist_ok=True)


# =========================================================
# Public API (IMPORT EDİLENLER)
# =========================================================

def save_image(
    image: np.ndarray,
    *,
    prefix: str = "img",
    ext: str = ".jpg"
) -> str:
    """
    Saves an image to disk and returns the absolute file path.

    Used by:
      app.py
      pipeline.py
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("save_image expects a numpy ndarray")

    filename = f"{prefix}_{uuid.uuid4().hex}{ext}"
    path = os.path.join(IMAGES_DIR, filename)

    success = cv2.imwrite(path, image)
    if not success:
        raise IOError(f"Failed to write image to {path}")

    return path


def load_image(path: str) -> np.ndarray:
    """
    Loads an image from disk (BGR).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to read image: {path}")

    return img


# =========================================================
# Optional helpers (future-proof)
# =========================================================

def delete_image(path: str) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def ensure_storage_ready() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)
