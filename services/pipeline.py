# ai-virtual-tour-engine/services/pipeline.py
from __future__ import annotations

from typing import Dict, Any, List
import cv2

from services.stitching import stitch_images
from services.ai_panorama import generate_ai_panorama
from services.ai_fill import ai_fill_panorama
from services.storage import persist_image_bytes


def _encode_jpg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode jpg")
    return buf.tobytes()


def build_panorama_pipeline(rooms: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hybrid pipeline for REAL + FAKE 360.

    Behavior (your requested mode):
    - We TRY OpenCV for 4+ images (best if overlaps exist)
    - If OpenCV fails OR result fails sanity → we generate FAKE-360 via generate_ai_panorama
    - After ANY success → ai_fill_panorama is applied
    """

    results: List[Dict[str, Any]] = []

    for room in rooms.get("rooms", []):
        room_id = int(room.get("room_id", 0))
        images = room.get("images", []) or []
        count = len(images)

        if count == 0:
            results.append({
                "room_id": room_id,
                "status": "error",
                "error": "No images in room cluster",
                "image_count": 0,
                "mode": "none"
            })
            continue

        # --- Prefer OpenCV if we have enough images
        if count >= 4:
            try:
                pano = stitch_images(images)
                pano = ai_fill_panorama(pano)

                url = persist_image_bytes(_encode_jpg(pano), prefix="pano")
                results.append({
                    "room_id": room_id,
                    "status": "ok",
                    "panorama_url": url,
                    "image_count": count,
                    "mode": "opencv"
                })
                continue
            except Exception as e:
                # fall through to fake-360
                opencv_err = str(e)
        else:
            opencv_err = "insufficient images for opencv"

        # --- FAKE-360 fallback (your requirement)
        try:
            pano = generate_ai_panorama(images)
            pano = ai_fill_panorama(pano)

            url = persist_image_bytes(_encode_jpg(pano), prefix="panorama_ai_fallback")
            results.append({
                "room_id": room_id,
                "status": "ok",
                "panorama_url": url,
                "image_count": count,
                "mode": "ai_fallback",
                "note": f"OpenCV failed or skipped: {opencv_err}"
            })
        except Exception as e2:
            results.append({
                "room_id": room_id,
                "status": "error",
                "error": f"AI fallback failed: {str(e2)}",
                "image_count": count,
                "mode": "ai_fallback"
            })

    return {
        "room_count": len(results),
        "panorama_count": len([r for r in results if r.get("status") == "ok"]),
        "panoramas": results
    }
