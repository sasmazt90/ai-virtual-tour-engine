from __future__ import annotations

import os
import io
import base64
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import cv2
from PIL import Image

import requests


# =========================================================
# Config
# =========================================================

@dataclass
class PanoramaConfig:
    # OpenCV stitcher mode
    stitcher_mode: int = cv2.Stitcher_PANORAMA

    # Resize for stitching speed/robustness (max side)
    stitch_max_side: int = 1400

    # After stitch, final output max width (keeps detail but avoids huge files)
    output_max_width: int = 4096

    # Mask threshold: what counts as "black hole" area to inpaint
    black_thresh: int = 10

    # Expand mask a bit to avoid seams at mask edges
    mask_dilate_px: int = 12

    # If too much of the pano is black, stitching probably failed
    max_black_ratio: float = 0.45

    # 360 seam blending band (pixels)
    seam_blend_band: int = 96

    # Inpaint with OpenAI?
    use_ai_inpaint: bool = True

    # Inpaint prompt: keep room consistent, no new objects
    inpaint_prompt: str = (
        "Fill missing areas realistically using the same room context, lighting, materials, and perspective. "
        "Do NOT add new objects or furniture. Keep architecture consistent. Photorealistic."
    )


# =========================================================
# Public API
# =========================================================

def build_panorama_360(
    images_bgr: List[np.ndarray],
    *,
    config: Optional[PanoramaConfig] = None
) -> np.ndarray:
    """
    Main function:
      - stitch panorama
      - detect black gaps
      - AI inpaint gaps
      - make 360 wrap seam smoother (left-right continuity)
    Returns:
      final panorama BGR uint8
    """
    if config is None:
        config = PanoramaConfig()

    if len(images_bgr) < 2:
        raise ValueError("Need at least 2 images for panorama stitching")

    pano = stitch_panorama(images_bgr, config=config)
    pano = _cap_width(pano, config.output_max_width)

    # Mask black holes
    mask = compute_black_gap_mask(pano, thresh=config.black_thresh, dilate_px=config.mask_dilate_px)

    black_ratio = float(np.mean(mask > 0))
    if black_ratio > config.max_black_ratio:
        # too much missing => stitching unusable
        raise RuntimeError(f"Panorama has too many missing pixels (black_ratio={black_ratio:.2f}).")

    # AI inpaint missing zones if enabled & there is missing area
    if config.use_ai_inpaint and black_ratio > 0.002:
        pano = ai_inpaint_panorama(pano, mask, prompt=config.inpaint_prompt)

    # 360 wrap seam blending
    pano = wrap_seam_blend(pano, band=config.seam_blend_band)

    return pano


def stitch_panorama(images_bgr: List[np.ndarray], *, config: PanoramaConfig) -> np.ndarray:
    """
    OpenCV stitcher.
    """
    imgs = [_prep_for_stitch(img, max_side=config.stitch_max_side) for img in images_bgr]

    stitcher = cv2.Stitcher_create(config.stitcher_mode)
    status, pano = stitcher.stitch(imgs)

    if status != cv2.Stitcher_OK or pano is None:
        raise RuntimeError(f"OpenCV stitching failed (status={status}).")

    pano = _ensure_bgr_uint8(pano)
    return pano


def compute_black_gap_mask(pano_bgr: np.ndarray, *, thresh: int, dilate_px: int) -> np.ndarray:
    """
    Returns a single-channel uint8 mask where 255 indicates areas to inpaint.
    """
    pano = _ensure_bgr_uint8(pano_bgr)
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)

    # black/near-black
    mask = (gray <= thresh).astype(np.uint8) * 255

    # also include transparent-like / undefined borders by catching very low saturation
    hsv = cv2.cvtColor(pano, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask2 = ((val <= thresh + 10) & (sat <= 20)).astype(np.uint8) * 255

    mask = cv2.bitwise_or(mask, mask2)

    # Clean + dilate
    mask = _morph_close(mask, k=7)
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def wrap_seam_blend(pano_bgr: np.ndarray, *, band: int = 96) -> np.ndarray:
    """
    Makes left and right edges blend smoothly to support 360 loop.
    Strategy:
      - take a band from left and right
      - cross-fade them
      - apply result back to both sides
    """
    pano = _ensure_bgr_uint8(pano_bgr)
    h, w = pano.shape[:2]
    band = int(max(8, min(band, w // 6)))

    left = pano[:, :band].copy()
    right = pano[:, w - band:].copy()

    # Create alpha ramp
    alpha = np.linspace(0.0, 1.0, band, dtype=np.float32).reshape(1, band, 1)
    alpha = np.repeat(alpha, h, axis=0)

    # Blend: left edge should look like right edge and vice versa
    blended_left = (right.astype(np.float32) * (1.0 - alpha) + left.astype(np.float32) * alpha).astype(np.uint8)
    blended_right = (right.astype(np.float32) * alpha + left.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

    pano[:, :band] = blended_left
    pano[:, w - band:] = blended_right
    return pano


# =========================================================
# OpenAI Inpaint
# =========================================================

def ai_inpaint_panorama(pano_bgr: np.ndarray, mask_u8: np.ndarray, *, prompt: str) -> np.ndarray:
    """
    Uses OpenAI Images Edit (inpaint) style call.
    Needs:
      OPENAI_API_KEY
    Optional:
      OPENAI_IMAGE_MODEL (default set below)
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        # If key missing, fall back to classical inpaint (not as good)
        return classical_inpaint(pano_bgr, mask_u8)

    model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1").strip()

    # Convert to PNG bytes for image + mask
    img_png = _bgr_to_png_bytes(pano_bgr)
    mask_png = _mask_to_png_bytes(mask_u8)

    # OpenAI Images Edit endpoint (multipart)
    # Note: exact endpoint can vary; this one matches commonly used OpenAI Images API style.
    url = "https://api.openai.com/v1/images/edits"
    headers = {"Authorization": f"Bearer {api_key}"}

    files = {
        "image": ("image.png", img_png, "image/png"),
        "mask": ("mask.png", mask_png, "image/png"),
    }
    data = {
        "model": model,
        "prompt": prompt,
        # Keep same aspect ratio; request 1 result
        "n": "1",
        # Let backend decide size; pano is wide; many APIs accept "size" limited
        # We omit size to allow server inference if supported.
    }

    resp = requests.post(url, headers=headers, files=files, data=data, timeout=180)
    if resp.status_code >= 300:
        # fall back if API fails
        return classical_inpaint(pano_bgr, mask_u8)

    js = resp.json()
    # Expected: data[0].b64_json
    b64 = js["data"][0].get("b64_json")
    if not b64:
        # some variants return url
        img_url = js["data"][0].get("url")
        if img_url:
            out = requests.get(img_url, timeout=120).content
            return _png_bytes_to_bgr(out)
        return classical_inpaint(pano_bgr, mask_u8)

    out_bytes = base64.b64decode(b64)
    return _png_bytes_to_bgr(out_bytes)


def classical_inpaint(pano_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """
    Fallback: OpenCV inpaint. Lower quality, but prevents broken output.
    """
    pano = _ensure_bgr_uint8(pano_bgr)
    mask = (mask_u8 > 0).astype(np.uint8) * 255
    # TELEA tends to look better than NS for edges
    out = cv2.inpaint(pano, mask, 3, cv2.INPAINT_TELEA)
    return out


# =========================================================
# Helpers
# =========================================================

def _prep_for_stitch(img_bgr: np.ndarray, *, max_side: int) -> np.ndarray:
    img = _ensure_bgr_uint8(img_bgr)
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _cap_width(img_bgr: np.ndarray, max_w: int) -> np.ndarray:
    img = _ensure_bgr_uint8(img_bgr)
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    new_h = int(h * scale)
    return cv2.resize(img, (max_w, new_h), interpolation=cv2.INTER_AREA)


def _morph_close(mask: np.ndarray, *, k: int = 7) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def _bgr_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    img = cv2.cvtColor(_ensure_bgr_uint8(img_bgr), cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _mask_to_png_bytes(mask_u8: np.ndarray) -> bytes:
    m = (mask_u8 > 0).astype(np.uint8) * 255
    pil = Image.fromarray(m, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _png_bytes_to_bgr(png_bytes: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.array(pil)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return _ensure_bgr_uint8(bgr)


def _ensure_bgr_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("image is None")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img
