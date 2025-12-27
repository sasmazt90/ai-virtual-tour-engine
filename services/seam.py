from _future_ import annotations
import cv2
import numpy as np

def synthetic_panorama_from_single(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mirrored = cv2.flip(img, 1)
    pano = np.concatenate([img, mirrored], axis=1)
    pano = cv2.GaussianBlur(pano, (0, 0), sigmaX=1.0)
    return pano

def make_seamless_horizontal(pano: np.ndarray, blend_width: int = 100) -> np.ndarray:
    h, w = pano.shape[:2]
    bw = min(blend_width, w // 4)
    left = pano[:, :bw]
    right = pano[:, -bw:]

    alpha = np.linspace(0, 1, bw).reshape(1, bw, 1)
    blended = left * (1 - alpha) + right * alpha

    pano[:, :bw] = blended
    pano[:, -bw:] = blended[:, ::-1]
    return pano
