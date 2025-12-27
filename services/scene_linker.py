import cv2
from typing import List, Dict
import uuid


# -------------------------------------------------
# IMAGE HELPERS
# -------------------------------------------------

def _load_gray(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _similarity(a: str, b: str) -> float:
    img1 = _load_gray(a)
    img2 = _load_gray(b)

    if img1 is None or img2 is None:
        return 0.0

    orb = cv2.ORB_create(800)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)

    if d1 is None or d2 is None:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)

    return float(len(matches))


# -------------------------------------------------
# HOTSPOT BUILDER
# -------------------------------------------------

def build_hotspots(images: List[str]) -> Dict[str, list]:
    """
    Builds a simple graph based on visual similarity.

    Returns:
    {
      image_path: [
        {
          "id": "...",
          "x": 50,
          "y": 55,
          "target_image": "..."
        }
      ]
    }
    """

    # initialize adjacency list
    graph: Dict[str, List[str]] = {img: [] for img in images}

    # pairwise similarity
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            score = _similarity(images[i], images[j])
            if score > 40:  # stable empirical threshold
                graph[images[i]].append(images[j])
                graph[images[j]].append(images[i])

    # build hotspot objects
    hotspots: Dict[str, list] = {}

    for src, targets in graph.items():
        spots = []
        for t in targets:
            spots.append({
                "id": str(uuid.uuid4()),
                "x": 50,   # placeholder – will be replaced by vision-based positioning
                "y": 55,
                "target_image": t
            })
        hotspots[src] = spots

    return hotspots
