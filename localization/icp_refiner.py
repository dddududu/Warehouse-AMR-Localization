from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from geometry.se3 import compose_transform, transform_points


@dataclass(frozen=True)
class ICPResult:
    pose_4x4: np.ndarray
    num_inliers: int
    inlier_ratio: float
    rmse: float
    converged: bool
    iterations: int


def _yaw_to_matrix(yaw_rad: float) -> np.ndarray:
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return matrix


def pose_from_xy_yaw_z(x: float, y: float, yaw_rad: float, z: float = 0.0) -> np.ndarray:
    matrix = _yaw_to_matrix(yaw_rad)
    matrix[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return matrix


def voxel_downsample(points_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    voxel = max(float(voxel_size), 1.0e-4)
    voxel_indices = np.floor(points / voxel).astype(np.int32)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    unique_indices.sort()
    return points[unique_indices].astype(np.float32, copy=False)


def _estimate_planar_delta(source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[np.ndarray, float]:
    source_xy = np.asarray(source_xyz[:, :2], dtype=np.float64)
    target_xy = np.asarray(target_xyz[:, :2], dtype=np.float64)
    source_mean = source_xy.mean(axis=0)
    target_mean = target_xy.mean(axis=0)
    source_centered = source_xy - source_mean
    target_centered = target_xy - target_mean
    covariance = source_centered.T @ target_centered
    u_mat, _, vh_mat = np.linalg.svd(covariance)
    rotation_2d = vh_mat.T @ u_mat.T
    if np.linalg.det(rotation_2d) < 0.0:
        vh_mat[-1, :] *= -1.0
        rotation_2d = vh_mat.T @ u_mat.T
    translation_xy = target_mean - rotation_2d @ source_mean
    delta = np.eye(4, dtype=np.float64)
    delta[:2, :2] = rotation_2d
    delta[:2, 3] = translation_xy
    delta[2, 3] = float(np.mean(target_xyz[:, 2] - source_xyz[:, 2]))
    yaw_delta = math.atan2(rotation_2d[1, 0], rotation_2d[0, 0])
    return delta, float(yaw_delta)


def refine_pose_with_icp(
    query_points_xyz_sensor: np.ndarray,
    map_points_xyz_world: np.ndarray,
    initial_pose_4x4: np.ndarray,
    voxel_size_m: float = 0.3,
    max_iterations: int = 20,
    max_correspondence_distance_m: float = 1.0,
    min_correspondences: int = 24,
    nearest_neighbor_backend: str = "flann",
) -> ICPResult:
    query_down = voxel_downsample(query_points_xyz_sensor, voxel_size_m)
    map_down = voxel_downsample(map_points_xyz_world, voxel_size_m)
    if query_down.shape[0] == 0 or map_down.shape[0] == 0:
        return ICPResult(
            pose_4x4=np.asarray(initial_pose_4x4, dtype=np.float64),
            num_inliers=0,
            inlier_ratio=0.0,
            rmse=float("inf"),
            converged=False,
            iterations=0,
        )

    pose = np.asarray(initial_pose_4x4, dtype=np.float64)
    max_corr_sq = float(max_correspondence_distance_m) ** 2
    best_num_inliers = 0
    best_rmse = float("inf")
    converged = False
    if nearest_neighbor_backend == "flann":
        nearest_index = cv2.flann_Index(
            np.asarray(map_down, dtype=np.float32),
            {"algorithm": 1, "trees": 4},
        )
    elif nearest_neighbor_backend == "ckdtree":
        nearest_index = cKDTree(np.asarray(map_down, dtype=np.float64))
    else:
        raise ValueError(f"Unsupported nearest-neighbor backend: {nearest_neighbor_backend}.")

    for iteration_idx in range(max(1, int(max_iterations))):
        transformed_query = transform_points(pose, query_down).astype(np.float64)
        if nearest_neighbor_backend == "flann":
            nearest_indices, nearest_sq = nearest_index.knnSearch(
                np.asarray(transformed_query, dtype=np.float32),
                1,
                params={},
            )
            nearest_sq = nearest_sq.reshape(-1).astype(np.float64)
            nearest_indices = nearest_indices.reshape(-1).astype(np.int64)
        else:
            nearest_distances, nearest_indices = nearest_index.query(
                transformed_query,
                k=1,
                workers=1,
            )
            nearest_sq = np.square(nearest_distances, dtype=np.float64)
            nearest_indices = np.asarray(nearest_indices, dtype=np.int64)
        inlier_mask = nearest_sq <= max_corr_sq
        num_inliers = int(np.count_nonzero(inlier_mask))
        if num_inliers < int(min_correspondences):
            break

        matched_source = transformed_query[inlier_mask]
        matched_target = map_down[nearest_indices[inlier_mask]].astype(np.float64)
        rmse = float(np.sqrt(np.mean(np.sum((matched_source - matched_target) ** 2, axis=1))))
        if num_inliers > best_num_inliers or (num_inliers == best_num_inliers and rmse < best_rmse):
            best_num_inliers = num_inliers
            best_rmse = rmse

        delta, yaw_delta = _estimate_planar_delta(matched_source, matched_target)
        pose = compose_transform(delta, pose)
        translation_step = float(np.linalg.norm(delta[:3, 3]))
        if abs(yaw_delta) < 1.0e-3 and translation_step < 1.0e-3:
            converged = True
            break

    return ICPResult(
        pose_4x4=pose,
        num_inliers=best_num_inliers,
        inlier_ratio=float(best_num_inliers / max(1, query_down.shape[0])),
        rmse=best_rmse,
        converged=converged,
        iterations=iteration_idx + 1,
    )
