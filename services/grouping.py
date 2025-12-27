from typing import List, Dict
import cv2
import numpy as np
import os


def _load_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def _hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    hist_a = cv2.calcHist([a], [0], None, [64], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [64], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)


def group_images_by_room(
    image_paths: List[str],
    threshold: float = 0.6
) -> List[Dict]:
    """
    Groups visually similar images (likely same room).

    Returns:
    [
      {
        "scene_id": "scene_1",
        "images": [path1, path2, ...]
      },
      ...
    ]
    """

    paths = [p for p in image_paths if os.path.exists(p)]
    if not paths:
        return []

    images = [(p, _load_gray(p)) for p in paths]
    used = set()
    groups = []
    scene_counter = 1

    for i, (path_i, img_i) in enumerate(images):
        if path_i in used:
            continue

        group = [path_i]
        used.add(path_i)

        for j, (path_j, img_j) in enumerate(images):
            if path_j in used:
                continue

            sim = _hist_similarity(img_i, img_j)
            if sim >= threshold:
                group.append(path_j)
                used.add(path_j)

        groups.append({
            "scene_id": f"scene_{scene_counter}",
            "images": group
        })
        scene_counter += 1

    return groups
