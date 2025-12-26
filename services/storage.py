import os
import shutil
import uuid
from fastapi import UploadFile
from fastapi.responses import FileResponse

DATA_DIR = "/data"
os.makedirs(DATA_DIR, exist_ok=True)


async def save_image(upload_file: UploadFile, dst_path: str):
    with open(dst_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


def persist_panorama(tmp_path: str) -> str:
    """
    /tmp/...jpg  →  /data/pano_<uuid>.jpg
    """
    ext = os.path.splitext(tmp_path)[1]
    filename = f"pano_{uuid.uuid4().hex}{ext}"
    final_path = os.path.join(DATA_DIR, filename)
    shutil.copy(tmp_path, final_path)
    return filename


def serve_file(filename: str):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    return FileResponse(path, media_type="image/jpeg")
