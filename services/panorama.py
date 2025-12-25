import cv2
import numpy as np
from typing import List
from services.pipeline import build_panorama_pipeline


def load_images_from_paths(paths: List[str]) -> List[np.ndarray]:
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise ValueError(f"Image could not be loaded: {p}")
        images.append(img)
    return images


def generate_360_panorama(image_paths: List[str]) -> np.ndarray:
    """
    Main public function.
    """
    images = load_images_from_paths(image_paths)
    panorama = build_panorama_pipeline(images)
    return panorama
