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


@app.post("/panorama")
async def create_panorama(
    files: List[UploadFile] = File(
        ...,
        description="Upload between 1 and 50 images"
    )
):
    # 🧹 Swagger'ın eklediği string/boş item'ları temizle
    files = [
        f for f in files
        if hasattr(f, "filename") and f.filename
    ]

    if not files:
        raise HTTPException(status_code=400, detail="No valid files uploaded")

    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_FILES} images allowed"
        )

    # 📁 Temp klasör
    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []

        for file in files:
            path = os.path.join(tmpdir, file.filename)
            await save_image(file, path)
            image_paths.append(path)

        # 🧠 Oda gruplama
        rooms = group_images_by_room(image_paths)

        # 🧩 Panorama pipeline
        panoramas = build_panorama_pipeline(rooms)

    return {
        "rooms": rooms,
        "panoramas": panoramas
    }
