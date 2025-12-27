import os
import uuid


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")


def persist_image_bytes(data: bytes, prefix: str = "image") -> str:
    """
    Saves image bytes under /static and returns public URL path.
    """

    os.makedirs(STATIC_DIR, exist_ok=True)

    filename = f"{prefix}_{uuid.uuid4().hex}.jpg"
    path = os.path.join(STATIC_DIR, filename)

    with open(path, "wb") as f:
        f.write(data)

    return f"/static/{filename}"
