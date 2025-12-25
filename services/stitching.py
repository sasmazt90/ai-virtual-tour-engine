import cv2
import numpy as np
from typing import List

def stitch_images_to_panorama(image_paths: List[str], out_path: str) -> None:
    imgs = []
    for p in image_paths:
        im = cv2.imread(p)
        if im is None:
            continue
        imgs.append(im)

    if len(imgs) < 2:
        # if only one image, just copy it (still works for viewer)
        if len(imgs) == 1:
            cv2.imwrite(out_path, imgs[0])
            return
        raise RuntimeError("No valid images provided.")

    # OpenCV Stitcher
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    except Exception:
        stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

    status, pano = stitcher.stitch(imgs)
    if status != cv2.Stitcher_OK or pano is None:
        raise RuntimeError(f"Stitch failed with status={status}. Try different overlap/order.")

    cv2.imwrite(out_path, pano)
