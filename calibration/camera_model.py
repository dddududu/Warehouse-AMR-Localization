from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: np.ndarray
    model_type: str = "PinHole"

    def __post_init__(self) -> None:
        distortion = np.asarray(self.distortion, dtype=np.float64)
        if distortion.shape != (5,):
            raise ValueError(f"Expected 5 distortion parameters, got {distortion.shape}.")
        object.__setattr__(self, "distortion", distortion)

    def build_K(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def undistort_points(self, points_uv: np.ndarray) -> np.ndarray:
        points = np.asarray(points_uv, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Expected points with shape (N, 2), got {points.shape}.")
        undistorted = cv2.undistortPoints(
            points.reshape(-1, 1, 2),
            self.build_K(),
            self.distortion,
            P=self.build_K(),
        )
        return undistorted.reshape(-1, 2)

    def scale_intrinsics(self, new_size_hw: tuple[int, int]) -> "CameraModel":
        new_height, new_width = new_size_hw
        scale_x = new_width / self.width
        scale_y = new_height / self.height
        return CameraModel(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            width=int(new_width),
            height=int(new_height),
            distortion=self.distortion.copy(),
            model_type=self.model_type,
        )

