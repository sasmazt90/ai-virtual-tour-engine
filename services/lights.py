from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import cv2
import numpy as np


# =========================
# DATA MODELS
# =========================

@dataclass
class LightAnalysis:
    """
    input görüntüde ışık varsayımları
    """
    window_mask: np.ndarray           # 0/255
    lamp_mask: np.ndarray             # 0/255
    window_dir: Tuple[float, float]   # (dx, dy) normalized
    lamp_center: Tuple[int, int]      # (x,y)
    lamp_strength: float              # 0..1
    ambient_strength: float           # 0..1


# =========================
# BASIC UTILS
# =========================

def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Image is None")
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0

def _to_uint8(img_f: np.ndarray) -> np.ndarray:
    return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)

def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _gamma(img_f: np.ndarray, g: float) -> np.ndarray:
    # g < 1 brightens, g > 1 darkens
    eps = 1e-6
    return np.clip((img_f + eps) ** g, 0.0, 1.0)

def _apply_color_temp(img_f: np.ndarray, warm: float) -> np.ndarray:
    """
    warm: -1..+1
    +1 => warmer (more red/yellow), -1 => cooler (more blue)
    """
    warm = float(max(-1.0, min(1.0, warm)))
    b, g, r = cv2.split(img_f)
    # subtle curve
    r = np.clip(r * (1.0 + 0.18 * warm), 0.0, 1.0)
    b = np.clip(b * (1.0 - 0.18 * warm), 0.0, 1.0)
    g = np.clip(g * (1.0 + 0.06 * warm), 0.0, 1.0)
    return cv2.merge([b, g, r])

def _soft_mask(mask_u8: np.ndarray, ksize: int = 31) -> np.ndarray:
    """
    0/255 mask -> 0..1 soft
    """
    m = (mask_u8.astype(np.float32) / 255.0)
    k = max(3, ksize | 1)
    m = cv2.GaussianBlur(m, (k, k), 0)
    return np.clip(m, 0.0, 1.0)

def _largest_component(mask_u8: np.ndarray) -> np.ndarray:
    """
    mask içindeki en büyük connected component'i al
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask_u8

    # 0 label background
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    out = np.zeros_like(mask_u8)
    out[labels == idx] = 255
    return out

def _center_of_mass(mask_u8: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return (0, 0)
    return (int(np.mean(xs)), int(np.mean(ys)))

def _normalize_vec(dx: float, dy: float) -> Tuple[float, float]:
    n = (dx*dx + dy*dy) ** 0.5
    if n < 1e-6:
        return (1.0, 0.0)
    return (dx / n, dy / n)


# =========================
# LIGHT DETECTION (HEURISTIC)
# =========================

def analyze_lights(img_bgr: np.ndarray) -> LightAnalysis:
    """
    Heuristik:
    - Window: geniş ve parlak alan (genelde soğuk/neutral)
    - Lamp: daha lokal, daha sıcak (yüksek V + yüksek S, kırmızı/yellow ağırlığı)
    """
    img_bgr = _ensure_bgr(img_bgr)
    h, w = img_bgr.shape[:2]

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # 1) Window mask: parlak + düşük/orta saturation (gün ışığı genelde daha az doygun)
    # V yüksek, S orta-alt
    window_raw = np.zeros((h, w), dtype=np.uint8)
    window_raw[(V > 210) & (S < 130)] = 255
    window_raw = cv2.morphologyEx(window_raw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    window_raw = cv2.dilate(window_raw, np.ones((9, 9), np.uint8), iterations=1)
    window_mask = _largest_component(window_raw)

    # 2) Lamp mask: sıcak tonlar + parlak (H: ~10-40 arası sarı/orange), S yüksek, V orta-yüksek
    lamp_raw = np.zeros((h, w), dtype=np.uint8)
    lamp_raw[((H >= 6) & (H <= 35) & (S > 110) & (V > 160))] = 255
    lamp_raw = cv2.morphologyEx(lamp_raw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    lamp_raw = cv2.dilate(lamp_raw, np.ones((7, 7), np.uint8), iterations=1)
    lamp_mask = _largest_component(lamp_raw)

    # Eğer lambayı bulamadıysa: fallback = en parlak lokal küçük bölgeyi bulmaya çalış
    if np.count_nonzero(lamp_mask) < 300:
        # V kanalında top %0.5 pikselleri al, küçük component seç
        v_thresh = np.percentile(V, 99.5)
        cand = (V >= v_thresh).astype(np.uint8) * 255
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # küçükse doğrudan al, değilse en büyük component
        lamp_mask = _largest_component(cand)

    # Window direction: window center -> image center vektörü (ışık içeriden merkeze düşer gibi)
    cx, cy = (w // 2, h // 2)
    wx, wy = _center_of_mass(window_mask) if np.count_nonzero(window_mask) > 300 else (w // 4, h // 3)
    # ışık yönü: pencereden merkeze doğru
    dx, dy = (cx - wx, cy - wy)
    window_dir = _normalize_vec(float(dx), float(dy))

    # Lamp center
    lx, ly = _center_of_mass(lamp_mask) if np.count_nonzero(lamp_mask) > 200 else (int(w * 0.75), int(h * 0.6))

    # Lambanın “mevcut” olma olasılığı ve ortam ışığı seviyesini tahmin et
    # - lamp_strength: lamp mask içinde V ortalaması / 255
    # - ambient_strength: tüm görüntü V ortalaması / 255
    lamp_strength = 0.0
    if np.count_nonzero(lamp_mask) > 0:
        lamp_strength = float(np.mean(V[lamp_mask > 0]) / 255.0)
    ambient_strength = float(np.mean(V) / 255.0)

    # normalize
    lamp_strength = _clamp01((lamp_strength - 0.4) / 0.6)  # 0..1
    ambient_strength = _clamp01((ambient_strength - 0.2) / 0.7)

    return LightAnalysis(
        window_mask=window_mask,
        lamp_mask=lamp_mask,
        window_dir=window_dir,
        lamp_center=(lx, ly),
        lamp_strength=lamp_strength,
        ambient_strength=ambient_strength,
    )


# =========================
# LIGHT SIMULATION
# =========================

def _make_gradient_light(h: int, w: int, direction: Tuple[float, float], strength: float) -> np.ndarray:
    """
    direction: ışık vektörü (dx, dy) normalized
    strength: 0..1
    Output: 0..1 light map
    """
    dx, dy = direction
    # koordinat grid
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # normalize to [-1,1]
    x = (xx / (w - 1)) * 2 - 1
    y = (yy / (h - 1)) * 2 - 1

    # directional ramp: projeksiyon
    proj = x * dx + y * dy
    # proj [-2,2] -> [0,1]
    ramp = (proj + 1.0) / 2.0
    ramp = np.clip(ramp, 0.0, 1.0)

    # daha yumuşak düşüş
    ramp = ramp ** 1.6
    return np.clip(ramp * strength, 0.0, 1.0)

def _make_radial_light(h: int, w: int, center: Tuple[int, int], strength: float, radius: Optional[float] = None) -> np.ndarray:
    cx, cy = center
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    if radius is None:
        radius = 0.55 * float(min(w, h))

    # dist=0 => 1, dist=radius => 0
    m = 1.0 - (dist / max(1.0, radius))
    m = np.clip(m, 0.0, 1.0)
    m = m ** 2.2
    return np.clip(m * strength, 0.0, 1.0)

def _apply_light_map(img_f: np.ndarray, light_map: np.ndarray, warm: float = 0.0) -> np.ndarray:
    """
    light_map: 0..1
    warm: -1..1
    """
    # exposure boost
    boost = 1.0 + 0.55 * light_map[..., None]
    out = np.clip(img_f * boost, 0.0, 1.0)
    if abs(warm) > 1e-3:
        out = _apply_color_temp(out, warm=warm)
    return out


# =========================
# VARIATION GENERATOR
# =========================

def generate_variations(img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Returns:
      dict key -> image_bgr
      keys:
        morning_lamp_on
        morning_lamp_off
        night_lamp_on
        night_lamp_off
    """
    img_bgr = _ensure_bgr(img_bgr)
    h, w = img_bgr.shape[:2]
    img_f = _to_float(img_bgr)

    analysis = analyze_lights(img_bgr)

    # soft masks for blending local effects
    window_soft = _soft_mask(analysis.window_mask, ksize=41)
    lamp_soft = _soft_mask(analysis.lamp_mask, ksize=41)

    # Light maps
    # Morning: dış ışık güçlü, genel daha aydın + hafif soğuk/neutral
    morning_dir = analysis.window_dir
    morning_ambient = _make_gradient_light(h, w, morning_dir, strength=0.85)
    # pencere bölgesine biraz ekstra "fill"
    morning_window_boost = np.clip(window_soft * 0.55, 0.0, 1.0)
    morning_map = np.clip(morning_ambient + morning_window_boost, 0.0, 1.0)

    # Night: dış ışık çok düşük, genel karanlık + mavi ton
    night_ambient = _make_gradient_light(h, w, morning_dir, strength=0.25) * 0.25
    night_map = np.clip(night_ambient, 0.0, 1.0)

    # Lamp: lokal sıcak ışık
    lamp_map = _make_radial_light(h, w, analysis.lamp_center, strength=0.95, radius=0.55 * min(w, h))
    # lambanın gerçekten olduğu yere göre azalt
    lamp_map = np.clip(lamp_map * (0.35 + 0.65 * max(analysis.lamp_strength, 0.35)), 0.0, 1.0)
    # mask ile çarp (lamba yoksa tüm sahneyi yakmasın)
    lamp_map = np.clip(lamp_map * (0.35 + 0.65 * lamp_soft), 0.0, 1.0)

    # --- MORNING + LAMP OFF ---
    out_m_off = img_f.copy()
    # morning exposure + hafif cool
    out_m_off = _apply_light_map(out_m_off, morning_map, warm=-0.08)
    # kontrast biraz artıralım (gündüz daha net)
    out_m_off = cv2.convertScaleAbs(_to_uint8(out_m_off), alpha=1.06, beta=0)
    out_m_off = _to_float(out_m_off)
    out_m_off = np.clip(out_m_off, 0.0, 1.0)
    img_morning_lamp_off = _to_uint8(out_m_off)

    # --- MORNING + LAMP ON ---
    out_m_on = img_f.copy()
    out_m_on = _apply_light_map(out_m_on, morning_map, warm=-0.05)
    # lamba sıcak fill
    out_m_on = _apply_light_map(out_m_on, lamp_map * 0.45, warm=0.20)
    out_m_on = cv2.convertScaleAbs(_to_uint8(out_m_on), alpha=1.05, beta=0)
    out_m_on = _to_float(out_m_on)
    img_morning_lamp_on = _to_uint8(out_m_on)

    # --- NIGHT + LAMP OFF ---
    out_n_off = img_f.copy()
    # önce karart + soğuklaştır
    # gamma > 1 => darken
    out_n_off = _gamma(out_n_off, 1.55)
    # gece ambience çok az
    out_n_off = _apply_light_map(out_n_off, night_map * 0.15, warm=-0.22)
    # global saturation biraz düşsün
    hsv_n = cv2.cvtColor(_to_uint8(out_n_off), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_n[..., 1] *= 0.82
    hsv_n[..., 2] *= 0.88
    hsv_n = np.clip(hsv_n, 0, 255).astype(np.uint8)
    img_night_lamp_off = cv2.cvtColor(hsv_n, cv2.COLOR_HSV2BGR)

    # --- NIGHT + LAMP ON ---
    out_n_on = img_f.copy()
    out_n_on = _gamma(out_n_on, 1.45)
    # gece ambience + cool
    out_n_on = _apply_light_map(out_n_on, night_map * 0.18, warm=-0.18)
    # lamba ana ışık: daha kuvvetli ve sıcak
    out_n_on = _apply_light_map(out_n_on, lamp_map * 0.85, warm=0.35)

    # Lambanın açık olduğu versiyonda, lamba çevresindeki kontrast yumuşasın
    # (ışık bloom hissi)
    bloom = cv2.GaussianBlur(_to_uint8(out_n_on), (0, 0), sigmaX=9, sigmaY=9)
    out_n_on_u8 = _to_uint8(out_n_on)
    out_n_on_u8 = cv2.addWeighted(out_n_on_u8, 0.82, bloom, 0.18, 0)

    img_night_lamp_on = out_n_on_u8

    return {
        "morning_lamp_on": img_morning_lamp_on,
        "morning_lamp_off": img_morning_lamp_off,
        "night_lamp_on": img_night_lamp_on,
        "night_lamp_off": img_night_lamp_off,
    }
