from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

import torch
import open_clip


# =========================================================
# Config
# =========================================================

@dataclass
class ClusterConfig:
    """
    Controls grouping behavior.
    """
    # If similarity between images is below this, they shouldn't be in the same room group
    similarity_threshold: float = 0.82

    # Minimum cluster size; small clusters can be merged to nearest cluster
    min_cluster_size: int = 2

    # Hard cap on clusters to avoid over-fragmentation
    max_clusters: int = 6

    # Model selection
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"

    # Speed: resize max side for embedding
    embed_max_side: int = 512

    # If True, apply lightweight normalization to reduce exposure differences
    use_color_normalization: bool = True


# =========================================================
# Public API
# =========================================================

def group_images_into_rooms(
    images_bgr: List[np.ndarray],
    *,
    config: Optional[ClusterConfig] = None
) -> List[List[int]]:
    """
    Groups images into room clusters.

    Input:
      images_bgr: list of images (BGR)

    Output:
      clusters as list of lists of indices, e.g. [[0,3,4],[1,2]]
    """
    if config is None:
        config = ClusterConfig()

    n = len(images_bgr)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    # For very small sets, just return one group
    if n <= 3:
        return [list(range(n))]

    embeddings = embed_images_openclip(images_bgr, config=config)  # (n, d)
    sim = cosine_similarity(embeddings)  # (n, n)

    # Determine number of clusters automatically
    k = _estimate_cluster_count(sim, config=config)
    labels = _cluster_with_agglomerative(embeddings, k=k)

    clusters = _labels_to_clusters(labels)

    # Post-process: merge tiny clusters
    clusters = _merge_small_clusters(clusters, sim, min_size=config.min_cluster_size)

    # If still too many clusters, merge closest
    clusters = _cap_clusters(clusters, sim, max_clusters=config.max_clusters)

    # Sort clusters by size desc, then by smallest index asc for determinism
    clusters = sorted(
        [sorted(c) for c in clusters],
        key=lambda c: (-len(c), c[0] if c else 10**9),
    )
    return clusters


# =========================================================
# Embedding with OpenCLIP
# =========================================================

def embed_images_openclip(
    images_bgr: List[np.ndarray],
    *,
    config: ClusterConfig
) -> np.ndarray:
    """
    Returns L2-normalized OpenCLIP image embeddings as numpy float32 (n, d).
    """
    device = "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.clip_model,
        pretrained=config.clip_pretrained,
        device=device
    )
    model.eval()

    # OpenCLIP preprocess expects PIL; we'll feed torch tensors via manual pipeline for speed.
    # We'll mimic common CLIP preprocessing (resize, center crop, normalize).
    # Using open_clip's preprocess is PIL-based; for server we keep it fully OpenCV + torch.
    # This is good enough for clustering.

    imgs = []
    for img in images_bgr:
        x = _prep_image_for_clip(img, max_side=config.embed_max_side, color_norm=config.use_color_normalization)
        imgs.append(x)

    batch = torch.stack(imgs, dim=0)  # (n, 3, H, W)

    with torch.no_grad():
        feats = model.encode_image(batch)  # (n, d)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    return feats.cpu().numpy().astype(np.float32)


def _prep_image_for_clip(img_bgr: np.ndarray, *, max_side: int, color_norm: bool) -> torch.Tensor:
    """
    OpenCV BGR -> torch float tensor normalized similar to CLIP.

    - resize so max side == max_side
    - center crop square
    - normalize with CLIP mean/std
    """
    img = _ensure_bgr_uint8(img_bgr)

    if color_norm:
        img = _simple_color_normalize(img)

    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # center crop to square
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    img = img[y0:y0 + side, x0:x0 + side]

    # resize to CLIP expected 224
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # to float [0,1]
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3,224,224)

    # CLIP normalization
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    x = (x - mean) / std

    return x


def _simple_color_normalize(img_bgr: np.ndarray) -> np.ndarray:
    """
    Light normalization to reduce exposure/gamma differences across shots.
    Does NOT change geometry; only helps embeddings.
    """
    # Convert to LAB and equalize L channel slightly
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab2 = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out


# =========================================================
# Clustering logic
# =========================================================

def _estimate_cluster_count(sim: np.ndarray, *, config: ClusterConfig) -> int:
    """
    Heuristic:
      - build adjacency based on threshold
      - estimate connected components count as baseline
      - clamp between 1..max_clusters and <= n
    """
    n = sim.shape[0]
    thr = config.similarity_threshold

    # adjacency matrix (exclude self)
    adj = (sim >= thr).astype(np.uint8)
    np.fill_diagonal(adj, 0)

    # count connected components via DFS
    visited = [False] * n
    comps = 0

    for i in range(n):
        if visited[i]:
            continue
        comps += 1
        stack = [i]
        visited[i] = True
        while stack:
            u = stack.pop()
            nbrs = np.where(adj[u] == 1)[0].tolist()
            for v in nbrs:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

    # If everything connected => 1
    k = comps
    k = max(1, min(k, config.max_clusters, n))
    return k


def _cluster_with_agglomerative(embeddings: np.ndarray, *, k: int) -> np.ndarray:
    """
    Agglomerative clustering on cosine distance.
    """
    n = embeddings.shape[0]
    if k <= 1 or n <= 2:
        return np.zeros((n,), dtype=np.int32)

    # AgglomerativeClustering with cosine distance:
    # sklearn uses affinity/metric changes; for compatibility use metric='cosine' if available.
    try:
        clusterer = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="average",
        )
    except TypeError:
        # older sklearn fallback
        clusterer = AgglomerativeClustering(
            n_clusters=k,
            affinity="cosine",
            linkage="average",
        )

    labels = clusterer.fit_predict(embeddings)
    return labels.astype(np.int32)


def _labels_to_clusters(labels: np.ndarray) -> List[List[int]]:
    clusters: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels.tolist()):
        clusters.setdefault(lab, []).append(idx)
    return list(clusters.values())


def _merge_small_clusters(
    clusters: List[List[int]],
    sim: np.ndarray,
    *,
    min_size: int
) -> List[List[int]]:
    """
    Merge clusters smaller than min_size into the most similar other cluster.
    """
    if min_size <= 1:
        return clusters

    clusters = [c[:] for c in clusters]
    changed = True

    while changed:
        changed = False
        clusters = [c for c in clusters if c]  # remove empty

        # find a small cluster
        small_idx = None
        for i, c in enumerate(clusters):
            if len(c) < min_size and len(clusters) > 1:
                small_idx = i
                break

        if small_idx is None:
            break

        small = clusters[small_idx]
        # choose best target cluster by average similarity
        best_j = None
        best_score = -1.0
        for j, c in enumerate(clusters):
            if j == small_idx:
                continue
            score = _avg_similarity_between_sets(sim, small, c)
            if score > best_score:
                best_score = score
                best_j = j

        if best_j is None:
            break

        # merge
        clusters[best_j].extend(small)
        clusters[small_idx] = []
        changed = True

    return [c for c in clusters if c]


def _cap_clusters(
    clusters: List[List[int]],
    sim: np.ndarray,
    *,
    max_clusters: int
) -> List[List[int]]:
    """
    If clusters exceed max, repeatedly merge the two closest clusters (highest similarity).
    """
    clusters = [c[:] for c in clusters]
    while len(clusters) > max_clusters:
        best_pair = None
        best_score = -1.0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                score = _avg_similarity_between_sets(sim, clusters[i], clusters[j])
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        clusters[i].extend(clusters[j])
        clusters.pop(j)

    return clusters


def _avg_similarity_between_sets(sim: np.ndarray, a: List[int], b: List[int]) -> float:
    """
    Mean cosine similarity between all pairs (i in a, j in b).
    """
    if not a or not b:
        return -1.0
    sub = sim[np.ix_(a, b)]
    return float(np.mean(sub))


# =========================================================
# Utils
# =========================================================

def _ensure_bgr_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("image is None")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img
