from typing import List, Dict
import uuid


def build_scenes_from_groups(groups: List[Dict]) -> Dict:
    """
    Input:
    [
      {
        "scene_id": "scene_1",
        "images": [img1, img2, img3]
      },
      ...
    ]

    Output:
    {
      "scenes": [
        {
          "id": "scene_1",
          "image": img1,
          "hotspots": [...]
        }
      ]
    }
    """

    scenes = []

    for idx, group in enumerate(groups):
        scene_id = group.get("scene_id") or f"scene_{idx+1}"
        images = group.get("images", [])
        if not images:
            continue

        main_image = images[0]

        hotspots = []
        for other in images[1:]:
            hotspots.append({
                "id": str(uuid.uuid4()),
                "x": 50,
                "y": 55,
                "target_image": other
            })

        scenes.append({
            "id": scene_id,
            "image": main_image,
            "hotspots": hotspots
        })

    return {
        "scenes": scenes
    }
