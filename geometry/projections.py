from __future__ import annotations

import numpy as np


def project_points(points_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}.")
    if K.shape != (3, 3):
        raise ValueError(f"Expected intrinsics with shape (3, 3), got {K.shape}.")

    z = points[:, 2]
    valid = z > 0.0
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        normalized = points[valid, :2] / z[valid, None]
        uv[valid] = normalized @ K[:2, :2].T + K[:2, 2]
    return uv, valid


def unproject_pixels(pixels_uv: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    uv = np.asarray(pixels_uv, dtype=np.float64)
    depth_values = np.asarray(depth, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"Expected pixels with shape (N, 2), got {uv.shape}.")
    if depth_values.shape != (uv.shape[0],):
        raise ValueError("Depth shape must match the number of pixels.")
    if K.shape != (3, 3):
        raise ValueError(f"Expected intrinsics with shape (3, 3), got {K.shape}.")

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    x = (uv[:, 0] - cx) * depth_values / fx
    y = (uv[:, 1] - cy) * depth_values / fy
    return np.column_stack((x, y, depth_values)).astype(np.float32)

