from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_depth_png(
    depth_path: str | Path,
    depth_scale: float | None = None,
    resize_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    path = Path(depth_path)
    with Image.open(path) as image:
        depth = np.asarray(image)
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be single-channel, got shape {depth.shape}.")
    if depth.dtype != np.uint16:
        raise ValueError(f"Depth image must have dtype uint16, got {depth.dtype}.")
    if resize_hw is not None:
        resized = Image.fromarray(depth)
        depth = np.asarray(resized.resize((resize_hw[1], resize_hw[0]), resample=Image.Resampling.NEAREST))
    if depth_scale is None:
        return depth
    return depth.astype(np.float32) * float(depth_scale)

