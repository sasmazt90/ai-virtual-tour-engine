import os
import uuid
import shutil
from pathlib import Path
from typing import List, Dict
from fastapi import UploadFile

# =========================
# BASE DIRECTORIES
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
ROOMS_DIR = DATA_DIR / "rooms"

# =========================
# DIRECTORY SETUP
# =========================

def ensure_dirs():
    """
    Ensure all required directories exist.
    Called once on app startup.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ROOMS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# IMAGE STORAGE
# =========================

async def save_uploads(files: List[UploadFile]) -> Dict:
    """
    Saves uploaded images to disk with unique IDs.
    Returns metadata for clustering step.
    """
    ensure_dirs()

    saved_images = []

    for file in files:
        if not file.content_type.startswith("image/"):
            continue

        image_id = str(uuid.uuid4())
        ext = Path(file.filename).suffix.lower() or ".jpg"
        filename = f"{image_id}{ext}"
        target_path = UPLOADS_DIR / filename

        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_images.append({
            "image_id": image_id,
            "filename": filename,
            "path": str(target_path),
        })

    return {
        "count": len(saved_images),
        "images": saved_images
    }

# =========================
# ROOM STORAGE
# =========================

def create_room(room_id: str) -> Path:
    """
    Creates directory structure for a room.
    """
    room_dir = ROOMS_DIR / room_id
    images_dir = room_dir / "images"
    outputs_dir = room_dir / "outputs"

    images_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    return room_dir

def move_images_to_room(room_id: str, image_ids: List[str]) -> List[str]:
    """
    Moves clustered images into a room directory.
    """
    room_dir = create_room(room_id)
    images_dir = room_dir / "images"

    moved = []

    for image_id in image_ids:
        candidates = list(UPLOADS_DIR.glob(f"{image_id}.*"))
        if not candidates:
            continue

        src = candidates[0]
        dst = images_dir / src.name
        shutil.move(str(src), str(dst))
        moved.append(str(dst))

    return moved

# =========================
# ROOM METADATA
# =========================

def room_meta_path(room_id: str) -> Path:
    return ROOMS_DIR / room_id / "meta.json"

def save_room_meta(room_id: str, data: Dict):
    import json
    meta_file = room_meta_path(room_id)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_room_meta(room_id: str) -> Dict:
    import json
    meta_file = room_meta_path(room_id)
    if not meta_file.exists():
        return {}
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# ARTIFACT ACCESS
# =========================

def get_room_artifacts(room_id: str) -> Dict:
    """
    Returns URLs and paths for viewer.
    """
    room_dir = ROOMS_DIR / room_id
    outputs_dir = room_dir / "outputs"

    pano = outputs_dir / "panorama.jpg"

    return {
        "room_id": room_id,
        "pano_path": str(pano) if pano.exists() else None,
        "pano_url": f"/rooms/{room_id}/outputs/panorama.jpg" if pano.exists() else None
    }
