import cv2
import uuid


def stitch_images(image_paths: list[str]) -> str:
    """
    Returns a panorama image file path (/tmp/...jpg)

    IMPORTANT:
    - Requires >= 2 images (guarded)
    - Loads cv2 images (numpy arrays) before passing to stitcher
    """
    if len(image_paths) < 2:
        raise ValueError("Not enough images for OpenCV stitching")

    images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            raise ValueError(f"Failed to load image: {p}")
        images.append(img)

    stitcher = cv2.Stitcher_create()
    status, pano = stitcher.stitch(images)

    if status != cv2.Stitcher_OK or pano is None:
        raise RuntimeError(f"OpenCV stitching failed with status {status}")

    output = f"/tmp/pano_{uuid.uuid4().hex}.jpg"
    ok = cv2.imwrite(output, pano)
    if not ok:
        raise RuntimeError("Failed to write panorama output image")

    return output
