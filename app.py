import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from services.pipeline import build_panorama_pipeline

# =========================================================
# BASE CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================================================
# FASTAPI INIT
# =========================================================

app = FastAPI(
    title="AI Virtual Tour Engine",
    description="360° panorama & virtual tour generation service",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

# =========================================================
# MODELS
# =========================================================

class RoomImages(BaseModel):
    room_id: int
    images: List[str]


class PanoramaRequest(BaseModel):
    rooms: List[RoomImages]


# =========================================================
# HEALTHCHECK
# =========================================================

@app.get("/health")
def healthcheck():
    return {"status": "ok"}


# =========================================================
# UPLOAD ENDPOINT
# =========================================================

@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    saved_files = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}"
            )

        filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        saved_files.append(file_path)

    return {
        "uploaded": len(saved_files),
        "files": saved_files
    }


# =========================================================
# PANORAMA PIPELINE
# =========================================================

@app.post("/build-panorama")
def build_panorama(data: PanoramaRequest):
    """
    Expects:
    {
      "rooms": [
        {
          "room_id": 0,
          "images": ["uploads/a.jpg", "uploads/b.jpg"]
        }
      ]
    }
    """

    if not data.rooms:
        raise HTTPException(status_code=400, detail="No rooms provided")

    try:
        result = build_panorama_pipeline(
            {
                "rooms": [room.dict() for room in data.rooms]
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=result)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "service": "AI Virtual Tour Engine",
        "docs": "/docs",
        "health": "/health"
    }
