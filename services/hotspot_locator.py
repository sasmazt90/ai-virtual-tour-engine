import cv2
import numpy as np
from typing import Tuple


def detect_transition_point(image_path: str) -> Tuple[int, int]:
    """
    Heuristic door / opening detector.
    Returns (x_percent, y_percent)
    """

    img = cv2.imread(image_path)
    if img is None:
        return 50, 55

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 80, 160)

    # vertical structures → doors / openings
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25))
    verticals = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)

    contours, _ = cv2.findContours(verticals, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch

        # door-like ratio
        if ch > cw * 1.5 and area > best_area:
            best = (x, y, cw, ch)
            best_area = area

    if best is None:
        return 50, 55

    x, y, cw, ch = best

    cx = x + cw // 2
    cy = y + ch // 2

    return int((cx / w) * 100), int((cy / h) * 100)
