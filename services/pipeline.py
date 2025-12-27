import os
import uuid
from typing import List

from services.stitching import stitch_images
from services.ai_panorama import generate_ai_panorama

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_panorama_pipeline(images: List[str]) -> dict:
    """
    TEK PANORAMA
    - ODA YOK
    - GROUP YOK
    """

    if not images or len(images) < 2:
        raise ValueError("At least 2 images are required")

    output_name = f"panorama_{uuid.uuid4()}.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    # 1️⃣ OpenCV stitch (>=4 image)
    if len(images) >= 4:
        try:
            stitched = stitch_images(images, output_path)
            if stitched and os.path.exists(output_path):
                return {
                    "method": "opencv_stitcher",
                    "panorama": f"/static/{output_name}"
                }
        except Exception:
            pass  # AI fallback

    # 2️⃣ AI panorama (1 parametre!)
    ai_path = generate_ai_panorama(images)

    if not ai_path or not os.path.exists(ai_path):
        raise RuntimeError("AI panorama generation failed")

    return {
        "method": "ai_panorama",
        "panorama": f"/static/{os.path.basename(ai_path)}"
    }
