from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import tempfile
import os
import asyncio

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
    # 🧹 Swagger / multipart bazen boş item ekleyebiliyor → temizle
    files = [f for f in files if getattr(f, "filename", None)]

    if not files:
        raise HTTPException(status_code=400, detail="No valid files uploaded")

    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} images allowed")

    # 📁 Temp klasör
    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []

        for file in files:
            # filename None/"" olamaz çünkü üstte filtreledik
            safe_name = os.path.basename(file.filename)
            path = os.path.join(tmpdir, safe_name)

            await save_image(file, path)
            image_paths.append(path)

        # 🧠 Oda gruplama (CPU / model)
        rooms = group_images_by_room(image_paths)

        # 🧩 Pipeline (CPU-heavy) → event loop bloklamasın
        loop = asyncio.get_running_loop()
        panoramas = await loop.run_in_executor(None, build_panorama_pipeline, rooms)

    return panoramas
