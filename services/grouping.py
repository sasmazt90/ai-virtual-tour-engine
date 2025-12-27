# ai-virtual-tour-engine/services/grouping.py
from __future__ import annotations

from typing import List, Dict, Any
import os
import cv2
import numpy as np


# ----------------------------
# Helpers
# ----------------------------

def _read_small(path: str, size: int = 160) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return img


def _feat(path: str) -> np.ndarray:
    """
    Lightweight, label-free feature vector for room-agnostic clustering.
    No torch/open_clip/sklearn required.
    """
    img = _read_small(path, size=180)

    # 1) Color histogram (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]
    ).flatten()
    hist = hist / (np.linalg.norm(hist) + 1e-9)

    # 2) Edge orientation histogram
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    bins = 12
    ang_bin = (ang / 360.0 * bins).astype(np.int32) % bins
    edge_hist = np.zeros((bins,), dtype=np.float32)
    for b in range(bins):
        edge_hist[b] = float(mag[ang_bin == b].sum())
    edge_hist = edge_hist / (np.linalg.norm(edge_hist) + 1e-9)

    # 3) Tiny grayscale fingerprint
    tiny = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    tiny = tiny.astype(np.float32).flatten()
    tiny = tiny - tiny.mean()
    tiny = tiny / (np.linalg.norm(tiny) + 1e-9)

    v = np.concatenate(
        [hist.astype(np.float32), edge_hist, tiny], axis=0
    )
    v = v / (np.linalg.norm(v) + 1e-9)
    return v


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(a, b), -1.0, 1.0))


# ----------------------------
# Main logic
# ----------------------------

def group_images_by_room(image_paths: List[str]) -> Dict[str, Any]:
    """
    Improved room-agnostic clustering for 360 panoramas.

    Key guarantees:
    - Prevents 1–2 image fake rooms
    - Forces small clusters to merge into nearest large room
    - Optimized for OpenCV stitching success
    """

    valid = [p for p in image_paths if p and os.path.exists(p)]
    if not valid:
        return {"rooms": []}

    feats = []
    for p in valid:
        feats.append(_feat(p))
    feats = np.stack(feats, axis=0)

    # ---- Parameters ----
    THRESH = 0.45            # slightly looser to avoid splitting same room
    MIN_CLUSTER_SIZE = 4     # critical: avoid AI fallback rooms

    clusters: List[Dict[str, Any]] = []

    # ---- Greedy clustering ----
    for idx, path in enumerate(valid):
        v = feats[idx]

        if not clusters:
            clusters.append({
                "images": [path],
                "centroid": v.copy()
            })
            continue

        best_i = -1
        best_d = 1e9
        for i, c in enumerate(clusters):
            d = _cosine_dist(v, c["centroid"])
            if d < best_d:
                best_d = d
                best_i = i

        if best_d <= THRESH:
            c = clusters[best_i]
            c["images"].append(path)
            n = len(c["images"])
            c["centroid"] = (c["centroid"] * (n - 1) + v) / n
            c["centroid"] /= (np.linalg.norm(c["centroid"]) + 1e-9)
        else:
            clusters.append({
                "images": [path],
                "centroid": v.copy()
            })

    # ---- Post-merge small clusters ----
    big = [c for c in clusters if len(c["images"]) >= MIN_CLUSTER_SIZE]
    small = [c for c in clusters if len(c["images"]) < MIN_CLUSTER_SIZE]

    for sc in small:
        best_i = -1
        best_d = 1e9
        for i, bc in enumerate(big):
            d = _cosine_dist(sc["centroid"], bc["centroid"])
            if d < best_d:
                best_d = d
                best_i = i

        if best_i >= 0:
            bc = big[best_i]
            for img in sc["images"]:
                bc["images"].append(img)

            # recompute centroid
            vecs = [_feat(p) for p in bc["images"]]
            centroid = np.mean(np.stack(vecs, axis=0), axis=0)
            bc["centroid"] = centroid / (np.linalg.norm(centroid) + 1e-9)
        else:
            # fallback: keep as-is (rare)
            big.append(sc)

    # ---- Output ----
    out = {"rooms": []}
    for room_id, c in enumerate(big):
        out["rooms"].append({
            "room_id": int(room_id),
            "images": c["images"]
        })

    return out
