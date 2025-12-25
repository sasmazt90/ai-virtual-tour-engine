import os
import io
import base64
import requests
import cv2
import numpy as np
from PIL import Image


def classical_inpaint(pano_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.inpaint(pano_bgr, mask, 3, cv2.INPAINT_TELEA)


def ai_inpaint_panorama(
    pano_bgr: np.ndarray,
    mask: np.ndarray,
    prompt: str
) -> np.ndarray:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return classical_inpaint(pano_bgr, mask)

    def to_png(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        return buf.getvalue()

    image_png = to_png(pano_bgr)
    mask_png = Image.fromarray(mask).convert("L")
    buf = io.BytesIO()
    mask_png.save(buf, format="PNG")

    resp = requests.post(
        "https://api.openai.com/v1/images/edits",
        headers={"Authorization": f"Bearer {api_key}"},
        files={
            "image": ("image.png", image_png),
            "mask": ("mask.png", buf.getvalue()),
        },
        data={"prompt": prompt, "model": "gpt-image-1"},
        timeout=180,
    )

    if resp.status_code >= 300:
        return classical_inpaint(pano_bgr, mask)

    b64 = resp.json()["data"][0]["b64_json"]
    out = base64.b64decode(b64)
    img = Image.open(io.BytesIO(out)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
