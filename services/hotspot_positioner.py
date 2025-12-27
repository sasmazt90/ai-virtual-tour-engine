import cv2
import numpy as np


def estimate_hotspot_position(src_img_path: str, tgt_img_path: str):
    """
    Returns (x, y) in percentage based on feature flow direction
    """

    img1 = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(tgt_img_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return 50, 55

    orb = cv2.ORB_create(800)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)

    if d1 is None or d2 is None:
        return 50, 55

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)

    if len(matches) < 10:
        return 50, 55

    deltas = []
    for m in matches:
        p1 = k1[m.queryIdx].pt
        p2 = k2[m.trainIdx].pt
        deltas.append(p2[0] - p1[0])

    avg_dx = np.mean(deltas)

    if avg_dx > 20:
        return 70, 55  # right
    elif avg_dx < -20:
        return 30, 55  # left
    else:
        return 50, 60  # forward-ish
