import uuid
from typing import List, Dict, Tuple

import cv2


def _load_gray(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _similarity(a: str, b: str) -> int:
    """
    ORB feature match count (higher = more overlap).
    Returns integer match count.
    """
    img1 = _load_gray(a)
    img2 = _load_gray(b)

    if img1 is None or img2 is None:
        return 0

    orb = cv2.ORB_create(800)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)

    if d1 is None or d2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)

    return int(len(matches))


def build_hotspots(
    images: List[str],
    threshold: int = 40,
    top_k: int = 5,
    default_x: int = 50,
    default_y: int = 55,
) -> Dict[str, list]:
    """
    Builds a simple link graph between visually-overlapping images in the given list.

    Returns:
    {
      image_path: [
        { "id": "...", "x": 50, "y": 55, "target_image": "..." },
        ...
      ]
    }

    Notes:
    - Positioning is intentionally stable (default_x/default_y).
      Viewer is ready; later you can replace positioning with a locator if you want.
    - No room assumptions, no panorama.
    """

    imgs = [p for p in images if isinstance(p, str) and p.strip()]
    if len(imgs) < 2:
        return {p: [] for p in imgs}

    # score edges
    edges: List[Tuple[str, str, int]] = []
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            score = _similarity(imgs[i], imgs[j])
            if score >= threshold:
                edges.append((imgs[i], imgs[j], score))

    # adjacency with scores
    adj: Dict[str, List[Tuple[str, int]]] = {p: [] for p in imgs}
    for a, b, s in edges:
        adj[a].append((b, s))
        adj[b].append((a, s))

    # build hotspots per src (top_k strongest)
    out: Dict[str, list] = {}
    for src, targets in adj.items():
        targets_sorted = sorted(targets, key=lambda t: t[1], reverse=True)[:top_k]
        spots = []
        for t_path, _score in targets_sorted:
            spots.append({
                "id": str(uuid.uuid4()),
                "x": int(default_x),
                "y": int(default_y),
                "target_image": t_path
            })
        out[src] = spots

    return out
