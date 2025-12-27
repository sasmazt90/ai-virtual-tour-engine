from __future__ import annotations

from typing import Dict, Any, List
import cv2

from services.stitching import stitch_images
from services.ai_panorama import generate_ai_panorama
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
    AI-FIRST panorama pipeline.

    Rules:
    - <4 images: HARD FAIL
    - >=4 images:
        1. AI semantic reconstruction (primary)
        2. OpenCV stitching (supporting)
        3. AI fill ALWAYS applied
    """

    results: List[Dict[str, Any]] = []

    for room in rooms.get("rooms", []):
        room_id = int(room.get("room_id", 0))
        images = room.get("images", []) or []
        count = len(images)

        if count < 4:
            results.append({
                "room_id": room_id,
                "status": "error",
                "error": "Not enough images (minimum 4 required)",
                "image_count": count,
                "mode": "insufficient_images"
            })
            continue

        try:
            # 1️⃣ AI-first panorama (semantic / layout aware)
            pano = generate_ai_panorama(images)

            # 2️⃣ Always fill missing / weak areas
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
                "mode": "ai_primary"
            })

        except Exception:
            # 3️⃣ OpenCV fallback (LAST RESORT)
            try:
                pano = stitch_images(images)
                pano = ai_fill_panorama(pano)

                url = persist_image_bytes(
                    _encode_jpg(pano),
                    prefix="panorama_opencv_fallback"
                )

                results.append({
                    "room_id": room_id,
                    "status": "ok",
                    "panorama_url": url,
                    "image_count": count,
                    "mode": "opencv_fallback"
                })

            except Exception as e2:
                results.append({
                    "room_id": room_id,
                    "status": "error",
                    "error": f"AI and OpenCV failed: {str(e2)}",
                    "image_count": count,
                    "mode": "failed"
                })

    return {
        "room_count": len(results),
        "panorama_count": len([r for r in results if r["status"] == "ok"]),
        "panoramas": results
    }
