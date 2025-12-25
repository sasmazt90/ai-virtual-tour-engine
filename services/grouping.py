from typing import List, Dict
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os


def _image_feature(path: str) -> np.ndarray:
    """
    Lightweight feature:
    - resized grayscale histogram
    - image aspect ratio
    RAM-safe, fast, Render Free compatible
    """
    img = cv2.imread(path)
    if img is None:
        return np.zeros(65, dtype=np.float32)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    h, w = img.shape[:2]
    aspect = np.array([w / max(h, 1)], dtype=np.float32)

    return np.concatenate([hist, aspect], axis=0)


def group_images_into_rooms(image_paths: List[str]) -> Dict:
    """
    Groups images into rooms using visual similarity.

    Returns:
    {
      "rooms": [
        { "room_id": 0, "images": [...] },
        ...
      ]
    }
    """

    if not image_paths:
        return {"rooms": []}

    features = []
    valid_paths = []

    for p in image_paths:
        if os.path.exists(p):
            features.append(_image_feature(p))
            valid_paths.append(p)

    if not features:
        return {"rooms": []}

    X = np.stack(features, axis=0)

    clustering = DBSCAN(
        eps=0.35,
        min_samples=1,
        metric="euclidean"
    ).fit(X)

    labels = clustering.labels_

    rooms: Dict[int, List[str]] = {}
    for path, lab in zip(valid_paths, labels):
        rooms.setdefault(int(lab), []).append(path)

    return {
        "rooms": [
            {"room_id": rid, "images": imgs}
            for rid, imgs in rooms.items()
        ]
    }
