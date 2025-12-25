import os
import numpy as np
import cv2
import requests

def make_mask_from_black_pixels(image_path: str, mask_path: str) -> None:
    """
    OpenAI mask rule: transparent pixels indicate area to replace. :contentReference[oaicite:0]{index=0}
    We'll create a PNG mask where black regions become TRANSPARENT (replace),
    and non-black regions become OPAQUE (keep).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Cannot read image for mask.")

    # detect "black" pixels (tune threshold)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black = gray < 10  # boolean mask

    # Create RGBA mask: alpha=0 where replace, alpha=255 where keep
    h, w = gray.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0:3] = 255  # white
    rgba[..., 3] = np.where(black, 0, 255).astype(np.uint8)

    cv2.imwrite(mask_path, rgba)

def fill_black_with_openai_inpaint(image_path: str, mask_path: str, out_path: str) -> None:
    """
    Calls OpenAI image editing with a mask.
    - Mask must match original image dimensions. :contentReference[oaicite:1]{index=1}
    - Transparent pixels are replaced. :contentReference[oaicite:2]{index=2}
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing (set it in Render Env Vars).")

    # NOTE: We use the OpenAI Images API endpoint.
    # The exact endpoint/model name can evolve; keep this isolated here.
    url = "https://api.openai.com/v1/images/edits"

    prompt = (
        "Fill the missing black areas naturally to match the same room. "
        "Do not add new objects. Continue existing walls, ceiling, floor, windows, and lighting consistently. "
        "Photorealistic, seamless panorama continuation."
    )

    headers = {"Authorization": f"Bearer {api_key}"}

    with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
        files = {
            "image": img_f,
            "mask": mask_f,
        }
        data = {
            "model": "gpt-image-1",
            "prompt": prompt,
            "size": "1024x1024"
        }

        r = requests.post(url, headers=headers, files=files, data=data, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI inpaint failed: {r.status_code} {r.text}")

        payload = r.json()
        # Many responses return base64; handle both url/base64 conservatively
        if "data" not in payload or len(payload["data"]) == 0:
            raise RuntimeError("OpenAI returned no image data.")

        item = payload["data"][0]
        if "url" in item:
            img_bytes = requests.get(item["url"], timeout=120).content
            with open(out_path, "wb") as out:
                out.write(img_bytes)
        elif "b64_json" in item:
            import base64
            img_bytes = base64.b64decode(item["b64_json"])
            with open(out_path, "wb") as out:
                out.write(img_bytes)
        else:
            raise RuntimeError("Unknown OpenAI image response format.")
