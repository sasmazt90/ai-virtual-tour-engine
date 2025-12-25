from typing import List, Dict
import numpy as np
from sklearn.cluster import DBSCAN

# Lazy-load heavy model only when needed
_model = None

def _load_clip():
    global _model
    if _model is not None:
        return _model

    import torch
    import open_clip

    device = "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    model.to(device)

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    _model = (model, preprocess, device)
    return _model

def _embed_image(path: str) -> np.ndarray:
    from PIL import Image
    import torch

    model, preprocess, device = _load_clip()
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]

def group_images_by_room(image_paths: List[str]) -> Dict:
    """
    Returns clusters like:
    { "rooms": [ { "room_id": 0, "images": [...] }, ... ] }
    """
    if len(image_paths) == 0:
        return {"rooms": []}

    embs = []
    for p in image_paths:
        embs.append(_embed_image(p))
    X = np.stack(embs, axis=0)

    # DBSCAN with cosine distance
    # eps smaller => stricter grouping
    clustering = DBSCAN(eps=0.25, min_samples=1, metric="cosine").fit(X)
    labels = clustering.labels_.tolist()

    rooms = {}
    for p, lab in zip(image_paths, labels):
        rooms.setdefault(lab, []).append(p)

    out = {"rooms": []}
    for lab, imgs in rooms.items():
        out["rooms"].append({
            "room_id": int(lab),
            "images": imgs
        })
    return out
