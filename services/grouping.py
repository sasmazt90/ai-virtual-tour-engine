from typing import List, Dict
import numpy as np
from sklearn.cluster import DBSCAN

# Lazy-loaded global model
_model = None


def _load_clip():
    global _model
    if _model is not None:
        return _model

    import torch
    import open_clip

    device = "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    model.to(device)

    _model = (model, preprocess, device)
    return _model


def _embed_image(path: str) -> np.ndarray:
    from PIL import Image
    import torch

    model, preprocess, device = _load_clip()

    img = Image.open(path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy()[0]


def group_images_by_room(image_paths: List[str]) -> Dict:
    """
    Groups images into rooms using CLIP embeddings + DBSCAN clustering.
    """
    if not image_paths:
        return {"rooms": []}

    embeddings = [_embed_image(p) for p in image_paths]
    X = np.stack(embeddings, axis=0)

    clustering = DBSCAN(
        eps=0.25,
        min_samples=1,
        metric="cosine"
    ).fit(X)

    rooms = {}
    for path, label in zip(image_paths, clustering.labels_):
        rooms.setdefault(int(label), []).append(path)

    return {
        "rooms": [
            {"room_id": room_id, "images": imgs}
            for room_id, imgs in rooms.items()
        ]
    }
