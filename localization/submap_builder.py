from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from preprocess.bev_builder import BEVConfig, points_to_bev


@dataclass(frozen=True)
class LocalSubmap:
    center_xy: tuple[float, float]
    size_m: float
    points_xyz_world: np.ndarray
    points_xyz_local: np.ndarray
    bev: np.ndarray
    bev_config: BEVConfig


def build_local_submap(
    map_points_xyz: np.ndarray,
    center_xy: tuple[float, float],
    size_m: float,
    resolution: float,
) -> LocalSubmap:
    points = np.asarray(map_points_xyz, dtype=np.float32)
    half_size = float(size_m) / 2.0
    center = np.asarray(center_xy, dtype=np.float32)
    mask = (
        (points[:, 0] >= center[0] - half_size)
        & (points[:, 0] <= center[0] + half_size)
        & (points[:, 1] >= center[1] - half_size)
        & (points[:, 1] <= center[1] + half_size)
    )
    world_points = points[mask].astype(np.float32, copy=False)
    local_points = world_points.copy()
    local_points[:, 0] -= center[0]
    local_points[:, 1] -= center[1]
    bev_config = BEVConfig(
        x_min=-half_size,
        x_max=half_size,
        y_min=-half_size,
        y_max=half_size,
        resolution=float(resolution),
    )
    bev = points_to_bev(local_points, bev_config)
    return LocalSubmap(
        center_xy=(float(center[0]), float(center[1])),
        size_m=float(size_m),
        points_xyz_world=world_points.astype(np.float32, copy=False),
        points_xyz_local=local_points.astype(np.float32, copy=False),
        bev=bev.astype(np.float32, copy=False),
        bev_config=bev_config,
    )
