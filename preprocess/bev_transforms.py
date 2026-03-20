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


def translate_bev_tensor(bev_tensor: np.ndarray, shift_x_cells: int, shift_y_cells: int) -> np.ndarray:
    tensor = np.asarray(bev_tensor, dtype=np.float32)
    if tensor.ndim != 3:
        raise ValueError(f"Expected bev_tensor with shape (C, H, W), got {tensor.shape}.")
    if shift_x_cells == 0 and shift_y_cells == 0:
        return tensor.astype(np.float32, copy=True)

    channels, height, width = tensor.shape
    translated = np.zeros((channels, height, width), dtype=np.float32)

    src_x_start = max(0, -int(shift_x_cells))
    src_x_end = min(width, width - int(shift_x_cells))
    dst_x_start = max(0, int(shift_x_cells))
    dst_x_end = dst_x_start + max(0, src_x_end - src_x_start)

    src_y_start = max(0, -int(shift_y_cells))
    src_y_end = min(height, height - int(shift_y_cells))
    dst_y_start = max(0, int(shift_y_cells))
    dst_y_end = dst_y_start + max(0, src_y_end - src_y_start)

    if src_x_start >= src_x_end or src_y_start >= src_y_end:
        return translated

    translated[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = tensor[:, src_y_start:src_y_end, src_x_start:src_x_end]
    return translated
