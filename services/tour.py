from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

from services.storage import Storage
from services.cluster import group_images_into_rooms, ClusterConfig
from services.panorama import build_panorama_360, PanoramaConfig


# =========================================================
# Config
# =========================================================

@dataclass
class TourConfig:
    # clustering
    cluster: ClusterConfig = ClusterConfig()

    # panorama stitching + AI fill + 360 seam blend
    pano: PanoramaConfig = PanoramaConfig()

    # max panoramas to build per request (safety)
    max_rooms: int = 12

    # When a room has too few photos, we can still generate a "pseudo pano"
    # by making a 2x1 canvas and placing the image duplicated; then AI fills gaps.
    allow_single_image_room: bool = True


# =========================================================
# Core service
# =========================================================

class TourService:
    """
    Orchestrates:
      - read uploaded images from storage
      - cluster them into rooms
      - build panorama for each room
      - store panoramas + manifest
    """

    def __init__(self, storage: Storage, config: Optional[TourConfig] = None):
        self.storage = storage
        self.config = config or TourConfig()

    def create_tour(
        self,
        image_keys: List[str],
        *,
        tour_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Input:
          image_keys: storage keys of uploaded images (jpg/png/webp)
        Output:
          manifest dict (also saved as JSON in storage)
        """
        if not image_keys:
            raise ValueError("image_keys is empty")

        tour_id = tour_id or f"tour_{uuid.uuid4().hex}"
        meta = meta or {}

        # 1) load images
        images_bgr, loaded_keys = self._load_images(image_keys)

        # 2) cluster images into rooms
        groups = group_images_into_rooms(images_bgr, keys=loaded_keys, config=self.config.cluster)

        # cap rooms
        if len(groups) > self.config.max_rooms:
            groups = groups[: self.config.max_rooms]

        rooms_manifest: List[Dict[str, Any]] = []

        # 3) build panorama per room
        for room_idx, group in enumerate(groups):
            group_keys = group["keys"]
            group_indices = group["indices"]
            room_label = group.get("label") or f"Room {room_idx + 1}"

            imgs = [images_bgr[i] for i in group_indices]

            pano_bgr = self._build_room_panorama(imgs)

            # store panorama
            pano_png = _bgr_to_png_bytes(pano_bgr)
            pano_key = f"{tour_id}/panos/room_{room_idx+1:02d}.png"
            self.storage.put_bytes(pano_key, pano_png, content_type="image/png")

            rooms_manifest.append(
                {
                    "id": f"room_{room_idx+1:02d}",
                    "name": room_label,
                    "sourceImageKeys": group_keys,
                    "panoramaKey": pano_key,
                    "panoramaUrl": self.storage.public_url(pano_key),
                    # viewer defaults
                    "initialYaw": 0.0,
                    "initialPitch": 0.0,
                }
            )

        # 4) build & store manifest
        manifest = {
            "tourId": tour_id,
            "createdAt": int(time.time()),
            "meta": meta,
            "rooms": rooms_manifest,
            # This is where you can later add navigation graph/hotspots.
            # For now, viewer can present a room list and switch scenes.
            "navigation": {
                "type": "list",
                "order": [r["id"] for r in rooms_manifest],
            },
        }

        manifest_key = f"{tour_id}/tour_manifest.json"
        self.storage.put_bytes(
            manifest_key,
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            content_type="application/json",
        )

        manifest["manifestKey"] = manifest_key
        manifest["manifestUrl"] = self.storage.public_url(manifest_key)

        return manifest

    # -----------------------------------------------------

    def _load_images(self, image_keys: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        images: List[np.ndarray] = []
        keys_ok: List[str] = []

        for k in image_keys:
            try:
                b = self.storage.get_bytes(k)
                img = _decode_image_to_bgr(b)
                if img is None:
                    continue
                images.append(img)
                keys_ok.append(k)
            except Exception:
                # skip broken files
                continue

        if len(images) == 0:
            raise RuntimeError("No valid images could be loaded from storage.")
        return images, keys_ok

    def _build_room_panorama(self, imgs_bgr: List[np.ndarray]) -> np.ndarray:
        # If only 1 image in a room: create a pseudo panoramic canvas and AI-fill the rest
        if len(imgs_bgr) == 1 and self.config.allow_single_image_room:
            pseudo = _make_pseudo_panorama(imgs_bgr[0])
            # We still run build_panorama_360 to do AI fill + seam blend
            return build_panorama_360([pseudo, pseudo], config=self.config.pano)

        # Normal case: stitch 2+ images
        return build_panorama_360(imgs_bgr, config=self.config.pano)


# =========================================================
# Helpers
# =========================================================

def _decode_image_to_bgr(img_bytes: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def _bgr_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return buf.tobytes()


def _make_pseudo_panorama(img_bgr: np.ndarray) -> np.ndarray:
    """
    Creates a wide canvas (2:1-ish) from a single normal photo.
    We keep the original photo centered and leave black sides for AI fill.
    This helps produce a 360-ish scene even when user uploaded only one image.
    """
    h, w = img_bgr.shape[:2]
    target_w = int(max(w * 2.0, w + 800))
    target_h = h

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # resize image to fit height
    scale = target_h / float(h)
    new_w = int(w * scale)
    resized = cv2.resize(img_bgr, (new_w, target_h), interpolation=cv2.INTER_AREA)

    x0 = (target_w - new_w) // 2
    canvas[:, x0 : x0 + new_w] = resized
    return canvas
