import os
import uuid
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from services.grouping import group_images_by_room
from services.scenes import build_scenes_from_groups

# -------------------------------------------------------------------
# APP SETUP
# -------------------------------------------------------------------

app = FastAPI(title="AI Virtual Tour Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -------------------------------------------------------------------
# UPLOAD ENDPOINT
# -------------------------------------------------------------------

@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    saved_files = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        filename = f"{uuid.uuid4()}{ext}"
        dest = os.path.join(UPLOAD_DIR, filename)

        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_files.append(dest)

    return {
        "uploaded": len(saved_files),
        "files": saved_files
    }

# -------------------------------------------------------------------
# BUILD SCENES (PHOTO → GROUP → SCENE)
# -------------------------------------------------------------------

@app.post("/build-scenes")
def build_scenes(payload: dict = Body(...)):
    """
    payload:
    {
      "files": ["/app/uploads/xxx.jpg", ...]
    }
    """

    files = payload.get("files", [])
    if not files:
        return {"scenes": []}

    groups = group_images_by_room(files)
    scenes = build_scenes_from_groups(groups)

    return scenes

# -------------------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "AI Virtual Tour Engine running"
    }
