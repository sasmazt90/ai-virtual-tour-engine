from dataclasses import dataclass
from typing import List, Optional
import cv2
import numpy as np

from services.stitching import stitch_panorama
from services.mask import compute_black_gap_mask
from services.inpaint import ai_inpaint_panorama
from services.seam import wrap_seam_blend
from services.utils import _cap_width


@dataclass
class PanoramaConfig:
    stitcher_mode: int = cv2.Stitcher_PANORAMA
    stitch_max_side: int = 1400
    output_max_width: int = 4096
    black_thresh: int = 10
    mask_dilate_px: int = 12
    max_black_ratio: float = 0.45
    seam_blend_band: int = 96
    use_ai_inpaint: bool = True
    inpaint_prompt: str = (
        "Fill missing areas realistically using the same room context, lighting, materials, and perspective. "
        "Do NOT add new objects or furniture. Keep architecture consistent. Photorealistic."
    )


def build_panorama_360(
    images_bgr: List[np.ndarray],
    *,
    config: Optional[PanoramaConfig] = None
) -> np.ndarray:

    if config is None:
        config = PanoramaConfig()

    pano = stitch_panorama(images_bgr, config=config)
    pano = _cap_width(pano, config.output_max_width)

    mask = compute_black_gap_mask(
        pano,
        thresh=config.black_thresh,
        dilate_px=config.mask_dilate_px
    )

    black_ratio = float((mask > 0).mean())
    if black_ratio > config.max_black_ratio:
        raise RuntimeError("Too many missing pixels after stitching")

    if config.use_ai_inpaint and black_ratio > 0.002:
        pano = ai_inpaint_panorama(pano, mask, prompt=config.inpaint_prompt)

    pano = wrap_seam_blend(pano, band=config.seam_blend_band)
    return pano
