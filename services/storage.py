import os
from typing import BinaryIO

BASE_DIR = os.getenv("STORAGE_DIR", "/tmp/virtual-tour-storage")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_file(file: BinaryIO, filename: str) -> str:
    ensure_dir(BASE_DIR)
    path = os.path.join(BASE_DIR, filename)
    with open(path, "wb") as f:
        f.write(file.read())
    return path


def load_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
