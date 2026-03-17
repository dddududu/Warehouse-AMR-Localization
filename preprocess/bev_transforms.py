from __future__ import annotations

import math

import cv2
import numpy as np


def rotate_points_xy(points_xyz: np.ndarray, angle_rad: float) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}.")
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    rotation = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=np.float32)
    rotated_xy = points[:, :2] @ rotation.T
    rotated = points.copy()
    rotated[:, :2] = rotated_xy
    return rotated


def rotate_bev_tensor(bev_tensor: np.ndarray, angle_deg: float) -> np.ndarray:
    tensor = np.asarray(bev_tensor, dtype=np.float32)
    if tensor.ndim != 3:
        raise ValueError(f"Expected bev_tensor with shape (C, H, W), got {tensor.shape}.")
    _, height, width = tensor.shape
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_channels: list[np.ndarray] = []
    for channel in tensor:
        rotated_channels.append(
            cv2.warpAffine(
                channel,
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        )
    return np.stack(rotated_channels, axis=0).astype(np.float32)
