from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BEVConfig:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float

    @property
    def width(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def height(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))


def points_to_bev(points_xyz: np.ndarray, config: BEVConfig) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}.")

    bev = np.zeros((4, config.height, config.width), dtype=np.float32)
    if points.size == 0:
        return bev

    # Map x/y to integer cell indices. Out-of-range points are discarded.
    cell_x = np.floor((points[:, 0] - config.x_min) / config.resolution).astype(np.int64)
    cell_y = np.floor((points[:, 1] - config.y_min) / config.resolution).astype(np.int64)
    valid = (
        (cell_x >= 0)
        & (cell_x < config.width)
        & (cell_y >= 0)
        & (cell_y < config.height)
    )
    if not np.any(valid):
        return bev

    points = points[valid]
    cell_x = cell_x[valid]
    cell_y = cell_y[valid]
    row = config.height - 1 - cell_y
    col = cell_x

    flat_index = row * config.width + col
    count = np.bincount(flat_index, minlength=config.height * config.width).astype(np.float32)
    height_sum = np.bincount(flat_index, weights=points[:, 2], minlength=config.height * config.width).astype(np.float32)
    max_height = np.full(config.height * config.width, -np.inf, dtype=np.float32)
    np.maximum.at(max_height, flat_index, points[:, 2])

    occupancy = count > 0
    mean_height = np.zeros_like(count)
    mean_height[occupancy] = height_sum[occupancy] / count[occupancy]
    max_height[~occupancy] = 0.0

    bev[0] = count.reshape(config.height, config.width)
    bev[1] = max_height.reshape(config.height, config.width)
    bev[2] = mean_height.reshape(config.height, config.width)
    bev[3] = occupancy.astype(np.float32).reshape(config.height, config.width)
    return bev

