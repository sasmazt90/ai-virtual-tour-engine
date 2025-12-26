# services/pipeline.py

from typing import Dict, List
from services.stitching import stitch_images

def build_panorama_pipeline(rooms: Dict) -> List[Dict]:
    """
    Takes grouped rooms and builds panoramas.
    Rooms with < 2 images are skipped safely.
    """

    panoramas = []

    for room in rooms.get("rooms", []):
        room_id = room.get("room_id")
        images = room.get("images", [])

        # 🚫 En kritik fix: 1 fotoğraflı odaları atla
        if len(images) < 2:
            panoramas.append({
                "room_id": room_id,
                "status": "skipped",
                "reason": "not enough images",
                "image_count": len(images)
            })
            continue

        try:
            panorama_path = stitch_images(images)

            panoramas.append({
                "room_id": room_id,
                "status": "ok",
                "panorama": panorama_path,
                "image_count": len(images)
            })

        except Exception as e:
            panoramas.append({
                "room_id": room_id,
                "status": "error",
                "error": str(e),
                "image_count": len(images)
            })

    return panoramas
