# ai-virtual-tour-engine/services/pipeline.py
from __future__ import annotations

from typing import Dict, Any, List
import cv2

from services.stitching import stitch_images
from services.ai_panorama import generate_ai_panorama
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
    Strict panorama pipeline.

    Rules:
    - <4 images: HARD FAIL (no AI fake panoramas)
    - 4–5 images: OpenCV only, no AI fallback
    - >=6 images: OpenCV primary, AI fallback only if OpenCV crashes
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
                "image_count": count,
                "mode": "insufficient_images"
            })
            continue

        # -------------------------
        # 4–5 images → OpenCV ONLY
        # -------------------------
        if 4 <= count <= 5:
            try:
                pano_img = stitch_images(images)
                url = persist_image_bytes(
                    _encode_jpg(pano_img),
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
                    "error": f"OpenCV panorama failed (no AI fallback allowed): {str(e)}",
                    "image_count": count,
                    "mode": "opencv"
                })
            continue

        # -------------------------
        # >=6 images → OpenCV + AI fallback (last resort)
        # -------------------------
        try:
            pano_img = stitch_images(images)
            url = persist_image_bytes(
                _encode_jpg(pano_img),
                prefix="pano"
            )
            results.append({
                "room_id": room_id,
                "status": "ok",
                "panorama_url": url,
                "image_count": count,
                "mode": "opencv"
            })
        except Exception:
            # AI fallback is allowed ONLY here
            try:
                pano_img = generate_ai_panorama(images[:4])
                url = persist_image_bytes(
                    _encode_jpg(pano_img),
                    prefix="panorama_ai_fallback"
                )
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
