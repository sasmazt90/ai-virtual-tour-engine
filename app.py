from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import os
import uuid
from services.grouping import group_images_by_room
from services.stitching import stitch_images_to_panorama
from services.inpaint import fill_black_with_openai_inpaint, make_mask_from_black_pixels

app = FastAPI(title="AI Virtual Tour Engine")

UPLOAD_DIR = "uploads"
OUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory=OUT_DIR), name="outputs")

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h2>AI Virtual Tour Engine</h2>
    <ul>
      <li>POST /group (files[]) -> rooms</li>
      <li>POST /panorama (files[]) -> stitched pano + inpainted pano</li>
      <li>GET  /viewer?img=/outputs/xxx.jpg</li>
    </ul>
    """

@app.post("/group")
async def group_endpoint(files: List[UploadFile] = File(...)):
    # save files
    paths = []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower() or ".jpg"
        name = f"{uuid.uuid4().hex}{ext}"
        p = os.path.join(UPLOAD_DIR, name)
        with open(p, "wb") as out:
            out.write(await f.read())
        paths.append(p)

    groups = group_images_by_room(paths)
    return JSONResponse(groups)

@app.post("/panorama")
async def panorama_endpoint(
    files: List[UploadFile] = File(...),
    do_inpaint: bool = Form(True),
):
    # 1) save
    paths = []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower() or ".jpg"
        name = f"{uuid.uuid4().hex}{ext}"
        p = os.path.join(UPLOAD_DIR, name)
        with open(p, "wb") as out:
            out.write(await f.read())
        paths.append(p)

    # 2) stitch
    pano_path = os.path.join(OUT_DIR, f"pano_{uuid.uuid4().hex}.jpg")
    stitch_images_to_panorama(paths, pano_path)

    result = {
        "stitched": f"/outputs/{os.path.basename(pano_path)}"
    }

    # 3) inpaint black regions (optional)
    if do_inpaint:
        mask_path = os.path.join(OUT_DIR, f"mask_{uuid.uuid4().hex}.png")
        make_mask_from_black_pixels(pano_path, mask_path)

        filled_path = os.path.join(OUT_DIR, f"pano_filled_{uuid.uuid4().hex}.jpg")
        fill_black_with_openai_inpaint(
            image_path=pano_path,
            mask_path=mask_path,
            out_path=filled_path
        )
        result["mask"] = f"/outputs/{os.path.basename(mask_path)}"
        result["filled"] = f"/outputs/{os.path.basename(filled_path)}"
        result["viewer"] = f"/viewer?img={result['filled']}"

    return JSONResponse(result)

@app.get("/viewer", response_class=HTMLResponse)
def viewer(img: str):
    # img should be like /outputs/xxx.jpg
    html = open("static/viewer.html", "r", encoding="utf-8").read()
    return html.replace("{{IMG_URL}}", img)
