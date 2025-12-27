from _future_ import annotations
from typing import Dict, Any, List
import cv2

from services.ai_panorama import generate_ai_panorama
from services.ai_fill import ai_fill_panorama
from services.storage import persist_image_bytes

def _encode_jpg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode jpg")
    return buf.tobytes()

def build_panorama_pipeline(rooms: Dict[str, Any]) -> Dict[str, Any]:
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

        try:
            pano = generate_ai_panorama(images)
            pano = ai_fill_panorama(pano)

            url = persist_image_bytes(
                _encode_jpg(pano),
                prefix="panorama_ai"
            )
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
                "error": str(e),
                "image_count": count,
                "mode": "ai"
            })

    return {
        "room_count": len(results),
        "panorama_count": len([r for r in results if r.get("status") == "ok"]),
        "panoramas": results
    }
