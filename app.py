from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import os
import uuid
import cv2
import numpy as np

from services.grouping import group_images_into_rooms
from services.panorama import build_panorama_360
from services.storage import save_image, OUTPUT_DIR


app = FastAPI(
    title="AI Virtual Tour Engine",
    version="0.1.0"
)

# -------------------------------------------------
# Static files (viewer)
# -------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")


# -------------------------------------------------
# Root
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h1>AI Virtual Tour Engine</h1>
    <ul>
        <li>POST /group (files[]) -> rooms</li>
        <li>POST /panorama (files[]) -> stitched + inpainted panorama</li>
        <li>GET /viewer?img=/outputs/xxx.jpg</li>
    </ul>
    """


# -------------------------------------------------
# GROUP ENDPOINT (ÖNCELİK 1)
# -------------------------------------------------
@app.post("/group")
async def group_endpoint(files: List[UploadFile] = File(...)):
    images = []

    for f in files:
        data = await f.read()
        np_img = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid image: {f.filename}"}
            )

        images.append(img)

    rooms = group_images_into_rooms(images)

    return {
        "room_count": len(rooms),
        "rooms": [
            {
                "room_id": i,
                "image_count": len(room)
            }
            for i, room in enumerate(rooms)
        ]
    }


# -------------------------------------------------
# PANORAMA ENDPOINT (ÖNCELİK 2)
# -------------------------------------------------
@app.post("/panorama")
async def panorama_endpoint(files: List[UploadFile] = File(...)):
    images = []

    for f in files:
        data = await f.read()
        np_img = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid image: {f.filename}"}
            )

        images.append(img)

    try:
        pano = build_panorama_360(images)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    filename = f"{uuid.uuid4().hex}.jpg"
    output_path = save_image(pano, filename)

    return {
        "panorama_path": f"/outputs/{filename}",
        "viewer_url": f"/viewer?img=/outputs/{filename}"
    }


# -------------------------------------------------
# VIEWER (360 HTML)
# -------------------------------------------------
@app.get("/viewer", response_class=HTMLResponse)
def viewer(img: str):
    viewer_path = os.path.join("static", "viewer.html")

    if not os.path.exists(viewer_path):
        return HTMLResponse("<h2>viewer.html not found</h2>", status_code=500)

    with open(viewer_path, "r", encoding="utf-8") as f:
        html = f.read()

    return html.replace("{{IMAGE_URL}}", img)


# -------------------------------------------------
# Ensure output dir exists
# -------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
