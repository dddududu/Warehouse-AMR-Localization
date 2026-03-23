from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from preprocess.bev_builder import BEVConfig, points_to_bev
from preprocess.bev_transforms import rotate_points_xy


@dataclass(frozen=True)
class BEVMatchResult:
    world_xy: tuple[float, float]
    yaw_rad: float
    score: float
    template_top_left: tuple[int, int]


def _extract_match_channel(bev_tensor: np.ndarray, channel: str) -> np.ndarray:
    tensor = np.asarray(bev_tensor, dtype=np.float32)
    key = str(channel).lower()
    if key == "occupancy":
        return tensor[3]
    if key == "count":
        return np.log1p(tensor[0])
    if key == "max_height":
        return tensor[1]
    if key == "mean_height":
        return tensor[2]
    if key == "count_height":
        return np.log1p(tensor[0]) + 0.25 * tensor[1]
    raise ValueError(f"Unsupported match_channel: {channel}")


def _match_template(search_image: np.ndarray, template_image: np.ndarray, method: str) -> tuple[float, tuple[int, int]]:
    match_method = str(method).lower()
    if match_method == "ccoeff":
        cv_method = cv2.TM_CCOEFF_NORMED
        maximize = True
    elif match_method == "sqdiff":
        cv_method = cv2.TM_SQDIFF_NORMED
        maximize = False
    else:
        raise ValueError(f"Unsupported coarse_match_method: {method}")
    response = cv2.matchTemplate(
        search_image.astype(np.float32),
        template_image.astype(np.float32),
        method=cv_method,
    )
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(response)
    if maximize:
        return float(max_val), (int(max_loc[0]), int(max_loc[1]))
    return float(1.0 - min_val), (int(min_loc[0]), int(min_loc[1]))


def match_query_points_to_submap_bev(
    query_points_xyz: np.ndarray,
    query_bev_config: BEVConfig,
    submap_bev: np.ndarray,
    submap_bev_config: BEVConfig,
    submap_center_xy: tuple[float, float],
    coarse_yaw_rad: float,
    yaw_half_range_deg: float,
    yaw_step_deg: float,
    match_channel: str = "occupancy",
    coarse_match_method: str = "ccoeff",
) -> BEVMatchResult:
    query_points = np.asarray(query_points_xyz, dtype=np.float32)
    search_image = _extract_match_channel(submap_bev, match_channel)
    best_result: BEVMatchResult | None = None
    half_range = float(yaw_half_range_deg)
    step = max(float(yaw_step_deg), 1.0e-3)
    yaw_offsets_deg = np.arange(-half_range, half_range + 0.5 * step, step, dtype=np.float32)
    if yaw_offsets_deg.size == 0:
        yaw_offsets_deg = np.array([0.0], dtype=np.float32)

    for yaw_offset_deg in yaw_offsets_deg:
        candidate_yaw_rad = float(coarse_yaw_rad + math.radians(float(yaw_offset_deg)))
        rotated_points = rotate_points_xy(query_points, candidate_yaw_rad)
        query_bev = points_to_bev(rotated_points, query_bev_config)
        template_image = _extract_match_channel(query_bev, match_channel)
        score, (left, top) = _match_template(search_image, template_image, coarse_match_method)

        template_center_col = left + query_bev_config.width / 2.0
        template_center_row = top + query_bev_config.height / 2.0
        local_x = submap_bev_config.x_min + template_center_col * submap_bev_config.resolution
        local_y = submap_bev_config.y_max - template_center_row * submap_bev_config.resolution
        world_x = float(submap_center_xy[0] + local_x)
        world_y = float(submap_center_xy[1] + local_y)
        candidate = BEVMatchResult(
            world_xy=(world_x, world_y),
            yaw_rad=candidate_yaw_rad,
            score=float(score),
            template_top_left=(int(left), int(top)),
        )
        if best_result is None or candidate.score > best_result.score:
            best_result = candidate

    if best_result is None:
        raise RuntimeError("BEV matching failed to produce any candidate pose.")
    return best_result
