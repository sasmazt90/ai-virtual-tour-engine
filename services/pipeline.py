from services.stitching import stitch_images
from services.ai_panorama import generate_ai_panorama


def build_panorama_pipeline(rooms: dict) -> dict:
    """
    rooms format:
    {
      "rooms": [
        {"room_id": 0, "images": [".../a.jpg", ".../b.jpg"]},
        ...
      ]
    }
    """
    results = []

    for room in rooms.get("rooms", []):
        room_id = room.get("room_id", 0)
        images = room.get("images", [])
        count = len(images)

        # ✅ Hard guarantee: hiçbir oda 500'e sebep olmasın
        if count == 0:
            results.append({
                "room_id": room_id,
                "status": "error",
                "error": "No images in room cluster",
                "image_count": 0,
                "mode": "none"
            })
            continue

        # ✅ 1–3 foto: OpenCV'ye hiç sokma → direkt AI panorama
        if count < 4:
            try:
                pano = generate_ai_panorama(images)
                results.append({
                    "room_id": room_id,
                    "status": "ok",
                    "panorama": pano,
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

        # ✅ 4+ foto: önce OpenCV dene, hata olursa AI fallback
        try:
            pano = stitch_images(images)
            results.append({
                "room_id": room_id,
                "status": "ok",
                "panorama": pano,
                "image_count": count,
                "mode": "opencv"
            })
        except Exception:
            try:
                pano = generate_ai_panorama(images)
                results.append({
                    "room_id": room_id,
                    "status": "ok",
                    "panorama": pano,
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
        "panorama_count": len(results),
        "panoramas": results
    }
