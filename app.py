from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.pipeline import build_panorama_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/build-panorama")
def build_panorama(payload: dict):
    """
    Expected payload:
    {
      "rooms": [
        {
          "room_id": 0,
          "images": ["path1.jpg", "path2.jpg", ...]
        }
      ]
    }
    """
    return build_panorama_pipeline(payload)
