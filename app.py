from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import uuid
import os
import shutil
import cv2
import numpy as np

from services.grouping import group_images_into_rooms
from services.pipeline import build_panorama_pipeline
from services.storage import save_image, load_image

app = FastAPI(
    title="AI Virtual Tour Engine",
    version="0.1.0"
)

# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# -------------------------------------------------
# Root
# -------------------------------------------------

@app.get("/", summary="Root")
def root():
    return HTMLResponse(
        """
        <h1>AI Virtual Tour Engine</h1>
        <ul>
            <li>POST /group (files[]) -> rooms</li>
            <li>POST /panorama (files[]) -> stitched pano + inpainted pano</li>
            <li>GET /viewer?img=/outputs/xxx.jpg</li>
        </ul>
        """
    )

# -------------------------------------------------
# GROUP ENDPOINT
# -------------------------------------------------

@app.post("/group", summary="Group images into rooms")
async def group_endpoint(files: list[UploadFile] = File(...)):
    image_paths = []

    for f in files:
        path = save_image(f)
        image_paths.append(path)

    rooms = group_images_into_rooms(image_paths)

    return JSONResponse(
        {
            "rooms": rooms
        }
    )

# -------------------------------------------------
# PANORAMA ENDPOINT
# -------------------------------------------------

@app.post("/panorama", summary="Create panorama")
async def panorama_endpoint(files: list[UploadFile] = File(...)):
    images = []

    for f in files:
        path = save_image(f)
        img = cv2.imread(path)
        if img is None:
            return JSONResponse(
                {"error": f"Image could not be read: {f.filename}"},
                status_code=400
            )
        images.append(img)

    # 🔥 CORE PIPELINE
    pano = build_panorama_pipeline(images)

    pano_id = f"{uuid.uuid4().hex}.jpg"
    pano_path = os.path.join(OUTPUT_DIR, pano_id)

    cv2.imwrite(pano_path, pano)

    return JSONResponse(
        {
            "panorama": f"/outputs/{pano_id}",
            "viewer": f"/viewer?img=/outputs/{pano_id}"
        }
    )

# -------------------------------------------------
# VIEWER
# -------------------------------------------------

@app.get("/viewer", summary="360 Viewer")
def viewer(img: str = Query(...)):
    html_path = os.path.join(STATIC_DIR, "viewer.html")

    if not os.path.exists(html_path):
        return HTMLResponse("viewer.html not found", status_code=404)

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("{{IMAGE_URL}}", img)

    return HTMLResponse(html)
