import cv2
import numpy as np


def make_seamless_horizontal(image: np.ndarray, blend_width: int = 80) -> np.ndarray:
    """
    Makes panorama horizontally seamless (left-right wrap).
    Required for 360° viewing.
    """
    h, w, c = image.shape

    left = image[:, :blend_width].astype(np.float32)
    right = image[:, w - blend_width:].astype(np.float32)

    alpha = np.linspace(0, 1, blend_width).reshape(1, -1, 1)

    blended = left * (1 - alpha) + right * alpha
    blended = blended.astype(np.uint8)

    result = image.copy()
    result[:, :blend_width] = blended
    result[:, w - blend_width:] = blended

    return result
