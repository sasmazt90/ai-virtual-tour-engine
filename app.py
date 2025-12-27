import os
import uuid
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from services.scene_linker import build_hotspots


# -------------------------------------------------
# APP SETUP
# -------------------------------------------------

app = FastAPI(title="AI Virtual Tour Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# PATHS
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -------------------------------------------------
# UPLOAD ENDPOINT
# -------------------------------------------------

@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    saved_files: List[str] = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        filename = f"{uuid.uuid4()}{ext}"
        dest = os.path.join(UPLOAD_DIR, filename)

        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Absolute path is fine on Render where app runs in /app
        saved_files.append(dest)

    return {
        "uploaded": len(saved_files),
        "files": saved_files
    }

# -------------------------------------------------
# BUILD SCENES (NO PANORAMA / NO ROOM ASSUMPTIONS)
# -------------------------------------------------

@app.post("/build-scenes")
def build_scenes(payload: dict = Body(...)):
    """
    payload:
    {
      "files": ["/app/uploads/xxx.jpg", ...]
    }

    output:
    {
      "scenes": [
        {
          "id": "scene_1",
          "image": "...",
          "hotspots": [
            {"id": "...", "x": 50, "y": 55, "target_image": "..."},
            ...
          ]
        },
        ...
      ]
    }
    """

    files = payload.get("files", [])
    if not files:
        return {"scenes": []}

    scenes = []
    SCENE_SIZE = 7  # stable chunking; no room assumptions

    for i in range(0, len(files), SCENE_SIZE):
        chunk = files[i:i + SCENE_SIZE]
        if not chunk:
            continue

        scene_id = f"scene_{len(scenes) + 1}"
        main_image = chunk[0]

        # Visual-overlap-based links within the chunk
        hotspot_map = build_hotspots(chunk)
        hotspots = hotspot_map.get(main_image, [])

        scenes.append({
            "id": scene_id,
            "image": main_image,
            "hotspots": hotspots
        })

    return {"scenes": scenes}

# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "AI Virtual Tour Engine running"
    }
