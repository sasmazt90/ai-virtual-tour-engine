# ai-virtual-tour-engine/services/pipeline.py
from __future__ import annotations

from typing import Dict, Any, List
import cv2

from services.stitching import stitch_images
from services.ai_panorama import generate_ai_panorama
from services.storage import persist_image_bytes

def _encode_jpg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode jpg")
    return buf.tobytes()


def build_panorama_pipeline(rooms: Dict[str, Any]) -> Dict[str, Any]:
    """
    rooms format:
    {
      "rooms": [
        {"room_id": 0, "images": [".../a.jpg", ".../b.jpg"]},
        ...
      ]
    }

    Return:
    {
      "room_count": N,
      "panorama_count": OK_COUNT,
      "panoramas": [
        { "room_id": 0, "status": "ok|error", "panorama_url": "/static/..jpg", ... }
      ]
    }
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

        # 1–3: AI (synthetic) route
        if count < 4:
            try:
                pano_img = generate_ai_panorama(images)
                url = persist_image_bytes(_encode_jpg(pano_img), prefix="panorama_ai")
                results.append({
                    "room_id": room_id,
                    "status": "ok",
                    "panorama_url": url,
                    "image_count": count,
                    "mode": "ai"
                })
            except Exception as e:
                results.append({
                    "room_id": room_id,
                    "status": "error",
                    "error": f"AI panorama failed: {str(e)}",
                    "image_count": count,
                    "mode": "ai"
                })
            continue

        # 4+: OpenCV then AI fallback
        try:
            pano_img = stitch_images(images)
            url = persist_image_bytes(_encode_jpg(pano_img), prefix="pano")
            results.append({
                "room_id": room_id,
                "status": "ok",
                "panorama_url": url,
                "image_count": count,
                "mode": "opencv"
            })
        except Exception:
            try:
                pano_img = generate_ai_panorama(images[:3])  # stable fallback
                url = persist_image_bytes(_encode_jpg(pano_img), prefix="panorama_ai_fallback")
                results.append({
                    "room_id": room_id,
                    "status": "ok",
                    "panorama_url": url,
                    "image_count": count,
                    "mode": "ai_fallback"
                })
            except Exception as e2:
                results.append({
                    "room_id": room_id,
                    "status": "error",
                    "error": f"OpenCV failed; AI fallback failed: {str(e2)}",
                    "image_count": count,
                    "mode": "ai_fallback"
                })

    return {
        "room_count": len(results),
        "panorama_count": len([r for r in results if r.get("status") == "ok"]),
        "panoramas": results
    }
