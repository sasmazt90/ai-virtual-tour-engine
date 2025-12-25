import cv2
import numpy as np
from typing import List

from services.stitching import stitch_images
from services.seam import make_seamless_horizontal


def detect_black_areas(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Returns a mask where black/unfilled areas exist.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray < threshold
    return mask.astype(np.uint8) * 255


def inpaint_black_areas(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Classical inpainting (AI hook later).
    """
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def build_panorama_pipeline(images: List[np.ndarray]) -> np.ndarray:
    """
    Full deterministic panorama pipeline.
    """
    panorama = stitch_images(images)

    mask = detect_black_areas(panorama)
    if mask.sum() > 0:
        panorama = inpaint_black_areas(panorama, mask)

    panorama = make_seamless_horizontal(panorama)

    return panorama
