from services.stitching import stitch_images
from services.ai_panorama import generate_ai_panorama

def build_panorama_pipeline(rooms: dict):
    results = []

    for room in rooms["rooms"]:
        room_id = room["room_id"]
        images = room["images"]
        count = len(images)

        # 🔴 AI MODE (1–3 foto)
        if count < 4:
            panorama_path = generate_ai_panorama(images)
            results.append({
                "room_id": room_id,
                "status": "ok",
                "panorama": panorama_path,
                "image_count": count,
                "mode": "ai"
            })
            continue

        # 🟢 OPENCV MODE (4+ foto)
        try:
            panorama_path = stitch_images(images)
            results.append({
                "room_id": room_id,
                "status": "ok",
                "panorama": panorama_path,
                "image_count": count,
                "mode": "opencv"
            })
        except Exception as e:
            # 🔁 FALLBACK AI
            panorama_path = generate_ai_panorama(images)
            results.append({
                "room_id": room_id,
                "status": "ok",
                "panorama": panorama_path,
                "image_count": count,
                "mode": "ai_fallback"
            })

    return {
        "room_count": len(results),
        "panorama_count": len(results),
        "panoramas": results
    }
