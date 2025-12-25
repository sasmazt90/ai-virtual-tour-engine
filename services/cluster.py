import os
import uuid
import numpy as np
from typing import List, Dict
from pathlib import Path

import cv2
from sklearn.cluster import DBSCAN

# =========================
# CONFIG
# =========================

IMAGE_SIZE = (224, 224)

# DBSCAN ayarları:
# eps → ne kadar benzerse aynı oda sayılacak
# min_samples → minimum kaç fotoğraf bir oda sayılır
DBSCAN_EPS = 0.35
DBSCAN_MIN_SAMPLES = 2

# =========================
# FEATURE EXTRACTION
# =========================

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image could not be loaded: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    return img

def extract_color_histogram(image: np.ndarray) -> np.ndarray:
    """
    Lightweight but effective descriptor:
    - Works without GPU
    - Captures room color/light distribution
    """
    hist = cv2.calcHist(
        [image],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_features(image_path: str) -> np.ndarray:
    img = load_image(image_path)
    return extract_color_histogram(img)

# =========================
# CLUSTERING
# =========================

def cluster_images(images: List[Dict]) -> Dict[str, List[str]]:
    """
    Groups images belonging to the same room.

    Input:
    [
        {
            "image_id": "...",
            "path": "/data/uploads/xxx.jpg"
        }
    ]

    Output:
    {
        "room_uuid_1": [image_id, image_id, ...],
        "room_uuid_2": [...]
    }
    """

    if len(images) == 0:
        return {}

    features = []
    valid_images = []

    for img in images:
        try:
            feat = extract_features(img["path"])
            features.append(feat)
            valid_images.append(img)
        except Exception as e:
            print(f"[cluster] Skipped image: {img['path']} → {e}")

    if len(features) < DBSCAN_MIN_SAMPLES:
        # Hepsi tek oda sayılır
        room_id = str(uuid.uuid4())
        return {
            room_id: [img["image_id"] for img in valid_images]
        }

    X = np.array(features)

    clustering = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="euclidean"
    ).fit(X)

    labels = clustering.labels_

    rooms: Dict[str, List[str]] = {}

    for label, img in zip(labels, valid_images):
        if label == -1:
            # Gürültü → tek başına oda
            room_id = str(uuid.uuid4())
        else:
            room_id = f"room_{label}"

        rooms.setdefault(room_id, []).append(img["image_id"])

    # Label’ları gerçek UUID’ye çevir
    final_rooms: Dict[str, List[str]] = {}

    for _, image_ids in rooms.items():
        if len(image_ids) == 0:
            continue
        room_uuid = str(uuid.uuid4())
        final_rooms[room_uuid] = image_ids

    return final_rooms
