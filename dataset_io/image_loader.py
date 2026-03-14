from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from calibration.camera_model import CameraModel


def load_rgb_image(
    image_path: str | Path,
    camera_model: CameraModel | None = None,
    use_undistort: bool = False,
    resize_hw: tuple[int, int] | None = None,
) -> tuple[np.ndarray, CameraModel | None]:
    path = Path(image_path)
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))

    updated_camera = camera_model
    if use_undistort:
        if camera_model is None:
            raise ValueError("camera_model is required when use_undistort=True.")
        rgb = cv2.undistort(rgb, camera_model.build_K(), camera_model.distortion)
    if resize_hw is not None:
        resized = Image.fromarray(rgb)
        rgb = np.asarray(resized.resize((resize_hw[1], resize_hw[0]), resample=Image.Resampling.BILINEAR))
        if updated_camera is not None:
            updated_camera = updated_camera.scale_intrinsics(resize_hw)
    return rgb, updated_camera

