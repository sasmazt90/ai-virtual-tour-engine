import os
from typing import List, Tuple, Optional

import cv2
import numpy as np


# =========================
# CONFIG
# =========================

# 360 seam fix için kenarlardan kaç px'lik bölgeyi blend edeceğiz
DEFAULT_SEAM_BLEND_PX = 160

# Siyah alan tespiti için eşik
BLACK_THRESH = 12

# Minimum kabul edilen pano boyutu
MIN_W = 900
MIN_H = 400


# =========================
# BASIC HELPERS
# =========================

def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Image is None")
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def _read_images(paths: List[str]) -> List[np.ndarray]:
    imgs = []
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            raise ValueError(f"Could not read image: {p}")
        imgs.append(_ensure_rgb(im))
    return imgs

def _is_probably_pano(img: np.ndarray) -> bool:
    """
    Equirectangular panoramalar genelde 2:1 oranına yakın olur.
    Bu sadece heuristik (tek görsel pano verilirse stitch yapmadan geçebiliriz).
    """
    h, w = img.shape[:2]
    if h == 0:
        return False
    ratio = w / float(h)
    return 1.75 <= ratio <= 2.35 and w >= MIN_W and h >= MIN_H

def _black_mask(img: np.ndarray) -> np.ndarray:
    """
    Siyah/boş bölgeleri maskelemek için.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > BLACK_THRESH).astype(np.uint8) * 255
    return mask

def _crop_to_content(img: np.ndarray) -> np.ndarray:
    """
    Siyah border'ları kırp.
    """
    mask = _black_mask(img)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)
    # Güvenlik: çok agresif kırpma olmasın
    if w < 200 or h < 200:
        return img
    return img[y:y+h, x:x+w]

def _linear_blend(a: np.ndarray, b: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    a ve b aynı boyutta olmalı.
    axis=1 → yatay blend (sol-sağ)
    """
    if a.shape != b.shape:
        raise ValueError("Blend inputs must have the same shape")

    h, w = a.shape[:2]
    if axis == 1:
        ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
        ramp = np.repeat(ramp, h, axis=0)
    else:
        ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        ramp = np.repeat(ramp, w, axis=1)

    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    out = (1.0 - ramp) * a_f + ramp * b_f
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================
# 360 SEAM FIX
# =========================

def make_360_seamless(pano: np.ndarray, blend_px: int = DEFAULT_SEAM_BLEND_PX) -> np.ndarray:
    """
    360 görüntü için kritik: sol ve sağ ucu görsel olarak yaklaştır.
    Bunu kenarlardaki dar bir bandı karşılıklı blend ederek yapıyoruz.
    Bu "matematiksel olarak aynı piksel" garanti etmez ama pratikte
    viewer'da seam'i dramatik şekilde azaltır.

    Not:
    - Pano zaten iyi stitch edilmiş olmalı.
    - blend_px çok yüksek olursa detaylar "yumuşar".
    """
    pano = _ensure_rgb(pano)
    h, w = pano.shape[:2]

    if w < 2 * blend_px + 10:
        # Çok küçük pano, blend yapma
        return pano

    left_band = pano[:, :blend_px].copy()
    right_band = pano[:, w - blend_px:].copy()

    # Sağ bandı sol ile, sol bandı sağ ile blend ediyoruz:
    blended_left = _linear_blend(left_band, right_band, axis=1)
    blended_right = _linear_blend(right_band, left_band, axis=1)

    out = pano.copy()
    out[:, :blend_px] = blended_left
    out[:, w - blend_px:] = blended_right
    return out


# =========================
# STITCHING
# =========================

def stitch_to_panorama(image_paths: List[str]) -> np.ndarray:
    """
    Verilen image list'ini panorama haline getirir.
    - 1 görsel geldiyse ve pano gibi görünüyorsa direkt döner.
    - Stitch başarısız olursa hata fırlatır.
    """
    if not image_paths:
        raise ValueError("No images provided to stitch")

    imgs = _read_images(image_paths)

    if len(imgs) == 1:
        # Tek görsel geldiyse: pano ise pano kabul et, değilse yine de döndür (engine karar versin)
        return imgs[0]

    # OpenCV Stitcher (panorama)
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    except Exception:
        stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

    status, pano = stitcher.stitch(imgs)

    if status != cv2.Stitcher_OK or pano is None:
        raise RuntimeError(f"Stitch failed with status={status}")

    pano = _ensure_rgb(pano)
    if pano.shape[1] < MIN_W or pano.shape[0] < MIN_H:
        raise RuntimeError("Panorama output is too small — likely bad match set")

    return pano


def build_panorama(
    image_paths: List[str],
    out_path: str,
    apply_crop: bool = True,
    apply_360_seam_fix: bool = True,
    seam_blend_px: int = DEFAULT_SEAM_BLEND_PX
) -> str:
    """
    Full pipeline:
    1) stitch
    2) crop black borders
    3) optional: make 360 seamless
    4) save output

    Returns: out_path
    """
    pano = stitch_to_panorama(image_paths)

    # Eğer tek görsel pano değilse kırpma yine de işe yarayabilir
    if apply_crop:
        pano = _crop_to_content(pano)

    # 360 için: sol/sağ seam blending
    if apply_360_seam_fix:
        pano = make_360_seamless(pano, blend_px=seam_blend_px)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ok = cv2.imwrite(out_path, pano)
    if not ok:
        raise RuntimeError(f"Failed to write panorama to: {out_path}")

    return out_path


# =========================
# OPTIONAL: QUALITY CHECK
# =========================

def estimate_seam_difference(pano: np.ndarray, sample_px: int = 12) -> float:
    """
    Pano'nun sol ve sağ uçları ne kadar farklı? (0 iyi, büyük kötü)
    Bu bir debug metriği, viewer seam riskini ölçmek için.
    """
    pano = _ensure_rgb(pano)
    h, w = pano.shape[:2]
    sample_px = max(2, min(sample_px, w // 10))

    left = pano[:, :sample_px].astype(np.float32)
    right = pano[:, w - sample_px:].astype(np.float32)

    # Mean absolute error
    return float(np.mean(np.abs(left - right)))
