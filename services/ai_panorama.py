import uuid
import shutil
import os
from typing import List


def generate_ai_panorama(images: List[str]) -> str:
    """
    AI panorama fallback.

    GUARANTEES:
    - images >= 1
    - Always returns a valid image path
    - Never raises due to image count
    """

    if not images:
        raise ValueError("generate_ai_panorama called with empty image list")

    output_path = f"/tmp/panorama_ai_{uuid.uuid4().hex}.jpg"

    first_image = images[0]

    if not os.path.exists(first_image):
        raise FileNotFoundError(f"Input image not found: {first_image}")

    # For now: SAFE fallback
    # 1 image → clone
    # N images → still clone first (no crash)
    shutil.copy(first_image, output_path)

    return output_path
