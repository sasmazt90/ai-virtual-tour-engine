# ai-virtual-tour-engine/services/pipeline.py
from __future__ import annotations

from typing import Dict, Any, List
import cv2

from services.stitching import stitch_images
from services.ai_fill import ai_fill_panorama
from services.storage import persist_image_bytes


def _encode_jpg(img) -> bytes:
    ok, buf = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    )
    if not ok:
        raise RuntimeError("Failed to encode jpg")
    return buf.tobytes()


def build_panorama_pipeline(rooms: Dict[str, Any]) -> Dict[str, Any]:
    """
    OPTION B – REAL PANORAMA ONLY (NO FAKE AI PANORAMA)

    Rules:
    - <4 images  → HARD FAIL
    - >=4 images → OpenCV panorama ONLY
    - After OpenCV success → ai_fill_panorama is ALWAYS applied
    - If OpenCV fails → ERROR (never AI collage)
    """

    results: List[Dict[str, Any]] = []

    for room in rooms.get("rooms", []):
        room_id = int(room.get("room_id", 0))
        images = room.get("images", []) or []
        count = len(images)

        # -------------------------
        # No images
        # -------------------------
        if count == 0:
            results.append({
                "room_id": room_id,
                "status": "error",
                "error": "No images in room cluster",
                "image_count": 0,
                "mode": "none"
            })
            continue

        # -------------------------
        # <4 images → HARD FAIL
        # -------------------------
        if count < 4:
            results.append({
                "room_id": room_id,
                "status": "error",
                "error": "Not enough images for panorama (minimum 4 required)",
                "hint": (
                    "Take photos from ONE fixed spot, rotating in place. "
                    "6–12 images recommended for clean 360 panorama."
                ),
                "image_count": count,
                "mode": "insufficient_images"
            })
            continue

        # -------------------------
        # OpenCV panorama (ONLY PATH)
        # -------------------------
        try:
            pano = stitch_images(images)

            # AI fill is allowed ONLY on REAL panoramas
            pano = ai_fill_panorama(pano)

            url = persist_image_bytes(
                _encode_jpg(pano),
                prefix="pano"
            )

            results.append({
                "room_id": room_id,
                "status": "ok",
                "panorama_url": url,
                "image_count": count,
                "mode": "opencv"
            })

        except Exception as e:
            results.append({
                "room_id": room_id,
                "status": "error",
                "error": f"OpenCV panorama failed: {str(e)}",
                "hint": (
                    "Photos likely include parallax (camera moved). "
                    "For true 360 panorama, stand still and rotate camera."
                ),
                "image_count": count,
                "mode": "opencv_failed"
            })

    return {
        "room_count": len(results),
        "panorama_count": len(
            [r for r in results if r.get("status") == "ok"]
        ),
        "panoramas": results
    }
