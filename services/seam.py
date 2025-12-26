# ai-virtual-tour-engine/services/seam.py
from __future__ import annotations
import numpy as np
import cv2

def make_seamless_horizontal(img: np.ndarray, blend_width: int = 80) -> np.ndarray:
    """
    Çok basit yatay dikiş yumuşatma:
    Sol ve sağ kenarı blendleyip "wrap" hissi verir.
    """
    if img is None or img.size == 0:
        raise ValueError("Invalid image")

    h, w = img.shape[:2]
    bw = max(10, min(blend_width, w // 4))

    left = img[:, :bw].astype(np.float32)
    right = img[:, w - bw:].astype(np.float32)

    # linear alpha blend
    alpha = np.linspace(0.0, 1.0, bw, dtype=np.float32)[None, :, None]
    blended = (left * (1.0 - alpha) + right * alpha).astype(np.uint8)

    out = img.copy()
    out[:, :bw] = blended
    out[:, w - bw:] = blended
    return out


def synthetic_panorama_from_single(img: np.ndarray, target_width_factor: float = 2.0) -> np.ndarray:
    """
    Tek foto için "AI yoksa" kullanılacak deterministik panorama:
    - Görseli yatayda genişletir (tile + blur edge)
    - Seamless blend uygular
    """
    if img is None or img.size == 0:
        raise ValueError("Invalid image")

    h, w = img.shape[:2]
    target_w = int(max(w * 1.6, w * target_width_factor))

    # tile: [img | flipped img | img ...] then crop
    flipped = cv2.flip(img, 1)
    tiled = np.concatenate([img, flipped, img], axis=1)
    start = max(0, (tiled.shape[1] - target_w) // 2)
    pano = tiled[:, start:start + target_w].copy()

    # light blur to reduce obvious repetition
    pano = cv2.GaussianBlur(pano, (0, 0), sigmaX=0.6)
    pano = make_seamless_horizontal(pano, blend_width=min(120, target_w // 6))
    return pano
