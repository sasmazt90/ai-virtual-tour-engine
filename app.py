import os
import uuid
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from services.scene_linker import build_hotspots

app = FastAPI(title="AI Virtual Tour Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    saved = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        name = f"{uuid.uuid4()}{ext}"
        path = os.path.join(UPLOAD_DIR, name)

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        saved.append(path)

    return {"uploaded": len(saved), "files": saved}


@app.post("/build-scenes")
def build_scenes(payload: dict = Body(...)):
    files = payload.get("files", [])
    if not files:
        return {"scenes": []}

    scenes = []
    CHUNK_SIZE = 7

    for i in range(0, len(files), CHUNK_SIZE):
        chunk = files[i:i + CHUNK_SIZE]
        if not chunk:
            continue

        main = chunk[0]
        hotspot_map = build_hotspots(chunk)

        scenes.append({
            "id": f"scene_{len(scenes) + 1}",
            "image": main,
            "hotspots": hotspot_map.get(main, [])
        })

    return {"scenes": scenes}


@app.get("/")
def root():
    return {"status": "ok"}
