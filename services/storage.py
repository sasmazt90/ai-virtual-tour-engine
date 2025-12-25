from fastapi import UploadFile
import shutil


async def save_image(file: UploadFile, path: str):
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
