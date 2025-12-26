# ai-virtual-tour-engine/services/storage.py
from __future__ import annotations

from fastapi import UploadFile
import os
import uuid
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


async def save_upload_file(upload: UploadFile, dst_path: str) -> None:
    """
    UploadFile -> disk
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "wb") as out:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def persist_image_bytes(data: bytes, ext: str = ".jpg", prefix: str = "panorama") -> str:
    """
    Save bytes into outputs and return public URL path (/static/...)
    """
    ensure_output_dir()
    ext = ext if ext.startswith(".") else f".{ext}"
    name = f"{prefix}_{uuid.uuid4().hex}{ext}"
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "wb") as f:
        f.write(data)
    return f"/static/{name}"


def persist_image_path(src_path: str, ext: Optional[str] = None, prefix: str = "panorama") -> str:
    """
    Copy an existing image file into outputs and return public URL path.
    """
    ensure_output_dir()
    if ext is None:
        _, e = os.path.splitext(src_path)
        ext = e or ".jpg"
    ext = ext if ext.startswith(".") else f".{ext}"
    name = f"{prefix}_{uuid.uuid4().hex}{ext}"
    dst = os.path.join(OUTPUT_DIR, name)

    with open(src_path, "rb") as r, open(dst, "wb") as w:
        w.write(r.read())

    return f"/static/{name}"
