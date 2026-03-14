from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from geometry.se3 import pose7_to_matrix, quaternion_xyzw_to_rotation_matrix, rotation_matrix_to_yaw


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion_xyzw(quaternion: Iterable[float]) -> float:
    rotation = quaternion_xyzw_to_rotation_matrix(quaternion)
    return rotation_matrix_to_yaw(rotation)


def yaw_from_pose_matrix(transform: np.ndarray) -> float:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected pose matrix with shape (4, 4), got {matrix.shape}.")
    return rotation_matrix_to_yaw(matrix[:3, :3])


def planar_pose_from_pose7(pose7: Iterable[float]) -> np.ndarray:
    pose = np.asarray(list(pose7), dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"Expected pose7 with shape (7,), got {pose.shape}.")
    yaw = yaw_from_pose_matrix(pose7_to_matrix(pose))
    return np.array([pose[0], pose[1], yaw], dtype=np.float64)
