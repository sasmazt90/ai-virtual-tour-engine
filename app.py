# ai-virtual-tour-engine/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from typing import List
import tempfile
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from services.grouping import group_images_by_room
from services.pipeline import build_panorama_pipeline
from services.storage import save_upload_file, ensure_output_dir

app = FastAPI(title="AI Virtual Tour Engine", version="1.0.0")

MAX_FILES = 50

# Thread pool: OpenCV + image IO CPU heavy
_executor = ThreadPoolExecutor(max_workers=1)

# Static output (public URLs)
OUTPUT_DIR = ensure_output_dir()
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/panorama")
async def create_panorama(
    files: List[UploadFile] = File(..., description="Upload between 1 and 50 images"),
):
    # Swagger bazen boş/garip entry ekliyor -> temizle
    files = [f for f in files if getattr(f, "filename", None)]

    if not files:
        raise HTTPException(status_code=400, detail="No valid files uploaded")

    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} images allowed")

    # Temp input directory
    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths: List[str] = []

        for f in files:
            dst = os.path.join(tmpdir, f.filename)
            await save_upload_file(f, dst)
            image_paths.append(dst)

        rooms = group_images_by_room(image_paths)

        # Pipeline blocking -> thread
        loop = asyncio.get_running_loop()
        panoramas = await loop.run_in_executor(_executor, build_panorama_pipeline, rooms)

    return panoramas
