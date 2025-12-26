# services/stitching.py

import cv2
import os
import uuid
from typing import List

def stitch_images(image_paths: List[str]) -> str:
    """
    Takes a list of image file paths and returns a stitched panorama path.
    """

    if len(image_paths) < 2:
        raise ValueError("Need at least 2 images to stitch")

    images = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        images.append(img)

    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

    status, pano = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"OpenCV stitching failed with status {status}")

    # 📦 Output path
    out_path = f"/tmp/panorama_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(out_path, pano)

    return out_path
