from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import tempfile
import os

from services.grouping import group_images_by_room
from services.pipeline import build_panorama_pipeline
from services.storage import save_image

app = FastAPI(
    title="AI Virtual Tour Engine",
    version="1.0.0"
)

MAX_FILES = 50


@app.get("/")
def health():
    return {"status": "ok"}


@app.post(
    "/panorama",
    summary="Create panorama(s) from uploaded room images",
    description="Upload between 1 and 50 images. Images will be grouped into rooms and processed into panoramas."
)
async def create_panorama(
    files: List[UploadFile] = File(...)
):
    # 🔒 Swagger bazen boş / string item ekleyebiliyor
    valid_files = [
        f for f in files
        if f is not None and hasattr(f, "filename") and f.filename
    ]

    if not valid_files:
        raise HTTPException(
            status_code=400,
            detail="No valid image files uploaded"
        )

    if len(valid_files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_FILES} images allowed"
        )

    # 📁 Temp çalışma alanı
    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []

        for file in valid_files:
            file_path = os.path.join(tmpdir, file.filename)
            await save_image(file, file_path)
            image_paths.append(file_path)

        # 🧠 Görselleri odaya göre grupla
        rooms = group_images_by_room(image_paths)

        # 🧩 Panorama üretim pipeline'ı
        panoramas = build_panorama_pipeline(rooms)

    return {
        "room_groups": rooms,
        "panoramas": panoramas
    }
