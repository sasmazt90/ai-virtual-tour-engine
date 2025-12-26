from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import tempfile
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from services.grouping import group_images_by_room
from services.pipeline import build_panorama_pipeline
from services.storage import save_image

app = FastAPI(
    title="AI Virtual Tour Engine",
    version="1.0.0"
)

MAX_FILES = 50

# 🔧 CPU-bound işler için thread pool
executor = ThreadPoolExecutor(max_workers=1)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/panorama")
async def create_panorama(
    files: List[UploadFile] = File(...)
):
    # 🧹 Swagger / multipart bug fix
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

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []

        # 📥 Dosyaları kaydet
        for file in files:
            path = os.path.join(tmpdir, file.filename)
            await save_image(file, path)
            image_paths.append(path)

        # 🧠 Gruplama (hafif işlem)
        rooms = group_images_by_room(image_paths)

        if not rooms:
            raise HTTPException(
                status_code=400,
                detail="No rooms detected from images"
            )

        # ⚠️ AĞIR İŞLEM → THREADPOOL
        loop = asyncio.get_event_loop()
        panoramas = await loop.run_in_executor(
            executor,
            build_panorama_pipeline,
            rooms
        )

    return {
        "room_count": len(rooms),
        "panorama_count": len(panoramas),
        "panoramas": panoramas
    }
