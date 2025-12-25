from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

from services.lights import generate_variations
from services.storage import save_image_bytes


# =========================
# TEMPLATE MODELS
# =========================

@dataclass
class TemplateVariant:
    key: str                 # "morning_lamp_on" etc.
    label: str               # Human readable
    timeOfDay: str           # "morning" | "night"
    lampState: str           # "on" | "off"
    image_url: str           # public URL
    width: int
    height: int


@dataclass
class TemplateOutput:
    template_id: str
    room_id: str
    input_image_url: Optional[str]
    variants: List[TemplateVariant]


# =========================
# INTERNAL HELPERS
# =========================

def _encode_png(img_bgr: np.ndarray) -> bytes:
    if img_bgr is None:
        raise ValueError("img_bgr is None")
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return buf.tobytes()

def _variant_meta(key: str) -> Tuple[str, str, str]:
    """
    returns (label, timeOfDay, lampState)
    """
    mapping = {
        "morning_lamp_on":  ("Morning / Lamp On",  "morning", "on"),
        "morning_lamp_off": ("Morning / Lamp Off", "morning", "off"),
        "night_lamp_on":    ("Night / Lamp On",    "night",   "on"),
        "night_lamp_off":   ("Night / Lamp Off",   "night",   "off"),
    }
    if key not in mapping:
        # generic fallback
        label = key.replace("_", " ").title()
        timeOfDay = "unknown"
        lampState = "unknown"
        return (label, timeOfDay, lampState)
    return mapping[key]

def _build_storage_path(room_id: str, template_id: str, key: str) -> str:
    # render / bucket yapısında düzenli tut
    # örn: rooms/<room_id>/templates/<template_id>/<key>.png
    return f"rooms/{room_id}/templates/{template_id}/{key}.png"


# =========================
# PUBLIC API
# =========================

def build_light_template_from_image(
    img_bgr: np.ndarray,
    *,
    room_id: Optional[str] = None,
    input_image_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tek bir oda görselinden 4 varyasyon üretir ve storage'a kaydeder.
    Dönen yapı frontend / Anything tarafından doğrudan kullanılabilir.

    - Ekstra obje eklemez/silmez: sadece ışık simülasyonu.
    """
    if img_bgr is None:
        raise ValueError("img_bgr is None")

    if room_id is None or not str(room_id).strip():
        room_id = str(uuid.uuid4())

    template_id = str(uuid.uuid4())

    variations = generate_variations(img_bgr)  # dict[str, np.ndarray]

    variants: List[TemplateVariant] = []

    for key, out_bgr in variations.items():
        h, w = out_bgr.shape[:2]
        label, timeOfDay, lampState = _variant_meta(key)

        png_bytes = _encode_png(out_bgr)
        storage_path = _build_storage_path(room_id, template_id, key)

        # public URL dönecek
        url = save_image_bytes(
            png_bytes,
            path=storage_path,
            content_type="image/png",
        )

        variants.append(
            TemplateVariant(
                key=key,
                label=label,
                timeOfDay=timeOfDay,
                lampState=lampState,
                image_url=url,
                width=w,
                height=h,
            )
        )

    output = TemplateOutput(
        template_id=template_id,
        room_id=room_id,
        input_image_url=input_image_url,
        variants=variants,
    )

    # frontend için json
    return {
        "template_id": output.template_id,
        "room_id": output.room_id,
        "input_image_url": output.input_image_url,
        "variants": [asdict(v) for v in output.variants],
    }
