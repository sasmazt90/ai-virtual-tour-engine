import os
import uuid
import shutil
import json
from typing import List, Dict, Any

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
TOURS_DIR = os.path.join(STATIC_DIR, "tours")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TOURS_DIR, exist_ok=True)

# Serve static and uploads publicly
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def _is_url_path(p: str) -> bool:
    return isinstance(p, str) and (p.startswith("/uploads/") or p.startswith("/static/"))

def _to_abs_upload_path(p: str) -> str:
    """
    Accepts:
      - "/uploads/<file>"  (preferred)
      - "/app/uploads/<file>" (legacy)
      - "<absolute path>" (already)
    Returns absolute FS path.
    """
    if not p:
        return p

    # Preferred web path
    if p.startswith("/uploads/"):
        fname = p.split("/uploads/", 1)[1]
        return os.path.join(UPLOAD_DIR, fname)

    # Legacy Render path
    if "/uploads/" in p:
        fname = p.split("/uploads/", 1)[1]
        return os.path.join(UPLOAD_DIR, fname)

    # Absolute
    return p

def _to_web_upload_path(abs_path: str) -> str:
    """
    Convert absolute FS path into web path under /uploads/.
    """
    if not abs_path:
        return abs_path

    if abs_path.startswith(UPLOAD_DIR + os.sep):
        fname = abs_path[len(UPLOAD_DIR) + 1 :]
        return f"/uploads/{fname}"

    # If it's already a web path, keep it
    if abs_path.startswith("/uploads/"):
        return abs_path

    # Fallback: try to extract last filename
    return f"/uploads/{os.path.basename(abs_path)}"

def _save_tour_json(tour_id: str, data: Dict[str, Any]) -> str:
    """
    Writes JSON to /static/tours/<tour_id>.json and returns web URL.
    """
    out_path = os.path.join(TOURS_DIR, f"{tour_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return f"/static/tours/{tour_id}.json"

# -------------------------------------------------
# UPLOAD ENDPOINT
# -------------------------------------------------

@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    saved_web_paths: List[str] = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower() or ".png"
        filename = f"{uuid.uuid4()}{ext}"
        dest = os.path.join(UPLOAD_DIR, filename)

        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_web_paths.append(f"/uploads/{filename}")

    return {
        "uploaded": len(saved_web_paths),
        "files": saved_web_paths
    }

# -------------------------------------------------
# BUILD SCENES (NO PANORAMA / NO ROOM ASSUMPTIONS)
# -------------------------------------------------

@app.post("/build-scenes")
def build_scenes(payload: dict = Body(...)):
    """
    payload:
    {
      "files": ["/uploads/xxx.png", ...]  (web paths)
    }

    response:
    {
      "tour_id": "...",
      "json_url": "/static/tours/<tour_id>.json",
      "scenes": [...]
    }
    """
    files = payload.get("files", [])
    if not files:
        empty = {"scenes": []}
        tour_id = str(uuid.uuid4())
        json_url = _save_tour_json(tour_id, empty)
        return {"tour_id": tour_id, "json_url": json_url, "scenes": []}

    # Ensure we operate on web paths (viewer needs these)
    web_files = []
    for p in files:
        # normalize legacy absolute paths -> web paths
        if isinstance(p, str) and p.startswith("/app/uploads/"):
            web_files.append(_to_web_upload_path(p))
        elif isinstance(p, str) and p.startswith("/uploads/"):
            web_files.append(p)
        else:
            # try to convert absolute -> web
            web_files.append(_to_web_upload_path(p))

    scenes = []
    SCENE_SIZE = 7  # stable chunking, no room assumptions

    for i in range(0, len(web_files), SCENE_SIZE):
        chunk_web = web_files[i:i + SCENE_SIZE]
        if not chunk_web:
            continue

        scene_id = f"scene_{len(scenes) + 1}"
        main_image_web = chunk_web[0]

        # For similarity: convert chunk to absolute FS paths
        chunk_abs = [_to_abs_upload_path(p) for p in chunk_web]

        hotspot_map_abs = build_hotspots(chunk_abs)

        # Convert hotspots back to web paths for the viewer
        main_abs = _to_abs_upload_path(main_image_web)
        abs_hotspots = hotspot_map_abs.get(main_abs, [])

        hotspots_web = []
        for h in abs_hotspots:
            hotspots_web.append({
                "id": h.get("id"),
                "x": h.get("x", 50),
                "y": h.get("y", 55),
                "target_image": _to_web_upload_path(h.get("target_image"))
            })

        scenes.append({
            "id": scene_id,
            "image": main_image_web,
            "hotspots": hotspots_web
        })

    tour = {"scenes": scenes}
    tour_id = str(uuid.uuid4())
    json_url = _save_tour_json(tour_id, tour)

    return {
        "tour_id": tour_id,
        "json_url": json_url,
        "scenes": scenes
    }

# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "AI Virtual Tour Engine running"}
