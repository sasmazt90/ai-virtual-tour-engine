from __future__ import annotations
from typing import List, Dict, Any
import os

def group_images_by_room(image_paths: List[str]) -> Dict[str, Any]:
    valid = [p for p in image_paths if p and os.path.exists(p)]
    return {
        "rooms": [
            {
                "room_id": 0,
                "images": valid
            }
        ]
    }
