from _future_ import annotations
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from typing import List
import os
import uuid

from services.pipeline import build_panorama_pipeline
from services.semantic_scene import build_rooms_from_semantics

app = FastAPI()

BASE_DIR = os.path.dirname(_file_)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

@app.post("/panorama")
async def create_panorama(files: List[UploadFile] = File(...)):
    img_paths = []
    batch_id = uuid.uuid4().hex

    for f in files:
        path = os.path.join(OUTPUT_DIR, f"{batch_id}_{f.filename}")
        with open(path, "wb") as out:
            out.write(await f.read())
        img_paths.append(path)

    rooms = build_rooms_from_semantics(img_paths)
    result = build_panorama_pipeline(rooms)
    return result
