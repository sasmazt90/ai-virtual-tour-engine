# ai-virtual-tour-engine/services/ai_panorama.py
from __future__ import annotations
from typing import List
import cv2
import numpy as np
import os

from services.seam import synthetic_panorama_from_single, make_seamless_horizontal

def _read(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def generate_ai_panorama(image_paths: List[str]) -> np.ndarray:
    """
    BUGÜN: gerçek AI model entegrasyonu yoksa bile 1-3 foto için panorama üret.
    - 1 foto: synthetic panorama
    - 2-3 foto: side-by-side + seam smooth (basit ama kırılmaz)
    """
    paths = [p for p in image_paths if p and os.path.exists(p)]
    if len(paths) == 0:
        raise ValueError("No images for AI panorama")

    if len(paths) == 1:
        img = _read(paths[0])
        return synthetic_panorama_from_single(img)

    imgs = [_read(p) for p in paths]

    # normalize heights
    h = min(i.shape[0] for i in imgs)
    resized = []
    for i in imgs:
        scale = h / i.shape[0]
        resized.append(cv2.resize(i, (int(i.shape[1] * scale), h), interpolation=cv2.INTER_AREA))

    pano = np.concatenate(resized, axis=1)

    # seam soften a bit
    pano = cv2.GaussianBlur(pano, (0, 0), sigmaX=0.7)
    pano = make_seamless_horizontal(pano, blend_width=min(100, pano.shape[1] // 8))
    return pano
