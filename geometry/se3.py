from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _as_quaternion_xyzw(quaternion: Iterable[float]) -> np.ndarray:
    q = np.asarray(list(quaternion), dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion with shape (4,), got {q.shape}.")
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("Quaternion norm must be non-zero.")
    return q / norm


def quaternion_xyzw_to_rotation_matrix(quaternion: Iterable[float]) -> np.ndarray:
    qx, qy, qz, qw = _as_quaternion_xyzw(quaternion)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose7_to_matrix(pose7: Iterable[float]) -> np.ndarray:
    pose = np.asarray(list(pose7), dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"Expected pose7 with shape (7,), got {pose.shape}.")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_xyzw_to_rotation_matrix(pose[3:])
    matrix[:3, 3] = pose[:3]
    return matrix


def invert_transform(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected transform with shape (4, 4), got {matrix.shape}.")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def compose_transform(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_matrix = np.asarray(left, dtype=np.float64)
    right_matrix = np.asarray(right, dtype=np.float64)
    if left_matrix.shape != (4, 4) or right_matrix.shape != (4, 4):
        raise ValueError("Both transforms must have shape (4, 4).")
    return left_matrix @ right_matrix


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    xyz = np.asarray(points, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected transform with shape (4, 4), got {matrix.shape}.")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {xyz.shape}.")
    if xyz.size == 0:
        return xyz.astype(np.float32)
    rotated = (matrix[:3, :3] @ xyz.T).T
    translated = rotated + matrix[:3, 3]
    return translated.astype(np.float32)


def rotation_matrix_to_yaw(rotation: np.ndarray) -> float:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected rotation with shape (3, 3), got {matrix.shape}.")
    return math.atan2(matrix[1, 0], matrix[0, 0])

