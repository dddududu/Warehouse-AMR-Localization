from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalCropConfig:
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    z_min: float = 0.13
    z_max: float = 4.73


def crop_local_lidar_points(points_xyz: np.ndarray, config: LocalCropConfig) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}.")
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    mask = (
        (points[:, 0] >= config.x_min)
        & (points[:, 0] <= config.x_max)
        & (points[:, 1] >= config.y_min)
        & (points[:, 1] <= config.y_max)
        & (points[:, 2] >= config.z_min)
        & (points[:, 2] <= config.z_max)
    )
    return points[mask].astype(np.float32, copy=False)

