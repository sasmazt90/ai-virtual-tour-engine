import cv2
import numpy as np


def find_hotspot_position(source_img_path: str) -> tuple[int, int]:
    """
    Returns (x_percent, y_percent)
    Heuristic:
    - Find strongest vertical edge cluster (door / opening)
    - Fallback to center-right
    """

    img = cv2.imread(source_img_path)
    if img is None:
        return 60, 55

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 80, 160)
    h, w = edges.shape

    vertical_sum = edges.sum(axis=0)
    max_x = int(np.argmax(vertical_sum))

    x_percent = int((max_x / w) * 100)
    y_percent = 55

    # Clamp to sane bounds
    x_percent = max(15, min(85, x_percent))

    return x_percent, y_percent
