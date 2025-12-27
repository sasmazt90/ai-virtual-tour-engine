from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import os
import uuid

from services.pipeline import build_panorama_pipeline

app = FastAPI(title="AI Virtual Tour Engine")

BASE_DIR = os.path.dirname(os.path.abspath(_file_))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/build-panorama-upload")
async def build_panorama_upload(
    files: List[UploadFile] = File(
        ..., description="Birden fazla görsel seçilebilir (multi-select aktif)"
    ),
    room_id: int = Form(0),
):
    """
    Swagger UI üzerinden MULTIPLE image upload destekler.
    Seçilen tüm görseller server'a kaydedilir ve pipeline'a path olarak gönderilir.
    """

    image_paths: List[str] = []

    for file in files:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext == "":
            ext = ".jpg"

        filename = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(UPLOAD_DIR, filename)

        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        image_paths.append(save_path)

    payload = {
        "rooms": [
            {
                "room_id": room_id,
                "images": image_paths,
            }
        ]
    }

    return build_panorama_pipeline(payload)


@app.post("/build-panorama")
def build_panorama(payload: dict):
    """
    JSON tabanlı endpoint (dosya upload yok).
    Örnek payload:
    {
      "rooms": [
        { "room_id": 0, "images": ["/app/uploads/a.jpg", "/app/uploads/b.jpg"] }
      ]
    }
    """
    return build_panorama_pipeline(payload)
