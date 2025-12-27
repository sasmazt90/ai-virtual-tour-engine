import os
import uuid
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from services.pipeline import build_panorama_pipeline

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="AI Virtual Tour Engine",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

# =========================================================
# MODELS
# =========================================================

class PanoramaImagesRequest(BaseModel):
    images: List[str]

# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():
    return {"status": "ok"}

# =========================================================
# UPLOAD
# =========================================================

@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    saved = []

    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            raise HTTPException(status_code=400, detail=f"Unsupported file: {f.filename}")

        name = f"{uuid.uuid4()}{ext}"
        path = os.path.join(UPLOAD_DIR, name)

        with open(path, "wb") as out:
            out.write(await f.read())

        saved.append(path)

    return {"uploaded": len(saved), "files": saved}

# =========================================================
# BUILD PANORAMA (ROOMSIZ / NİHAİ)
# =========================================================

@app.post("/build-panorama")
def build_panorama(payload: PanoramaImagesRequest):
    if not payload.images:
        raise HTTPException(status_code=400, detail="No images provided")

    try:
        result = build_panorama_pipeline(payload.images)
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
        "health": "/health",
        "docs": "/docs"
    }
