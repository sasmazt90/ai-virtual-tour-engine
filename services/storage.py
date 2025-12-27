from __future__ import annotations
import os
import uuid

BASE_DIR = os.path.dirname(os.path.dirname(_file_))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

def persist_image_bytes(data: bytes, prefix: str = "img") -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = f"{prefix}_{uuid.uuid4().hex}.jpg"
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "wb") as f:
        f.write(data)
    return f"/static/{name}"
