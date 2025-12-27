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
    AI-based synthetic panorama generator.

    - 1 image  → synthetic 360-style expansion
    - 2–3 images → semantic side-by-side + seam blending
    """

    paths = [p for p in image_paths if p and os.path.exists(p)]
    if not paths:
        raise ValueError("No images provided for AI panorama")

    # ---- Single image → synthetic wrap
    if len(paths) == 1:
        img = _read(paths[0])
        return synthetic_panorama_from_single(img)

    # ---- Multiple images → normalize + merge
    imgs = [_read(p) for p in paths]

    h = min(i.shape[0] for i in imgs)
    resized = []
    for img in imgs:
        scale = h / img.shape[0]
        resized.append(
            cv2.resize(
                img,
                (int(img.shape[1] * scale), h),
                interpolation=cv2.INTER_AREA
            )
        )

    pano = np.concatenate(resized, axis=1)

    pano = cv2.GaussianBlur(pano, (0, 0), sigmaX=0.6)
    pano = make_seamless_horizontal(
        pano,
        blend_width=min(120, pano.shape[1] // 6)
    )

    return pano
