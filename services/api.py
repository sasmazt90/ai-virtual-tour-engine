from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from services.storage import Storage
from services.tour import TourService, TourConfig

# =========================================================
# App & Router
# =========================================================

app = FastAPI(
    title="AI Virtual Tour Engine",
    description="Upload photos → AI groups rooms → builds 360 panoramas → serves manifest",
    version="1.0.0",
)

router = APIRouter()


# =========================================================
# Singletons (simple & explicit)
# =========================================================

storage = Storage()
tour_service = TourService(storage=storage, config=TourConfig())


# =========================================================
# Models (lightweight, inline)
# =========================================================

def _error(msg: str, status: int = 400):
    raise HTTPException(status_code=status, detail=msg)


# =========================================================
# Endpoints
# =========================================================

@router.get("/", tags=["health"])
def health():
    return {"status": "ok", "service": "ai-virtual-tour-engine"}


# ---------------------------------------------------------
# Upload images
# ---------------------------------------------------------

@router.post("/upload", tags=["upload"])
async def upload_images(files: List[UploadFile] = File(...)):
    """
    Upload raw room photos.
    Returns storage keys + public URLs.
    """
    if not files:
        _error("No files uploaded")

    uploaded = []

    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            continue

        data = await f.read()
        if not data:
            continue

        ext = (f.filename or "").split(".")[-1].lower()
        if ext not in {"jpg", "jpeg", "png", "webp"}:
            ext = "png"

        key = f"uploads/{uuid.uuid4().hex}.{ext}"
        storage.put_bytes(key, data, content_type=f.content_type)

        uploaded.append(
            {
                "key": key,
                "url": storage.public_url(key),
                "filename": f.filename,
            }
        )

    if not uploaded:
        _error("No valid image files found")

    return {"count": len(uploaded), "images": uploaded}


# ---------------------------------------------------------
# Create tour
# ---------------------------------------------------------

@router.post("/tour/create", tags=["tour"])
def create_tour(
    imageKeys: List[str],
    tourId: Optional[str] = None,
):
    """
    imageKeys = storage keys returned from /upload

    Creates:
      - room clustering
      - panoramas (AI-filled, 360-ready)
      - tour_manifest.json
    """
    if not imageKeys:
        _error("imageKeys is required")

    try:
        manifest = tour_service.create_tour(
            image_keys=imageKeys,
            tour_id=tourId,
        )
        return manifest

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "tour_creation_failed", "detail": str(e)},
        )


# ---------------------------------------------------------
# Get manifest
# ---------------------------------------------------------

@router.get("/tour/{tour_id}/manifest", tags=["tour"])
def get_manifest(tour_id: str):
    """
    Returns the stored tour_manifest.json
    """
    key = f"{tour_id}/tour_manifest.json"
    try:
        data = storage.get_bytes(key)
    except Exception:
        _error("Tour manifest not found", status=404)

    return JSONResponse(content=data.decode("utf-8"))


# =========================================================
# Mount router
# =========================================================

app.include_router(router)
