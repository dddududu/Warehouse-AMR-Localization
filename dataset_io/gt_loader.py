from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geometry.se3 import pose7_to_matrix
from geometry.yaw_utils import yaw_from_quaternion_xyzw


@dataclass(frozen=True)
class GroundTruthTrajectory:
    timestamps: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray
    poses_4x4: np.ndarray


def load_ground_truth(traj_path: str | Path) -> GroundTruthTrajectory:
    path = Path(traj_path)
    data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if data.ndim != 2 or data.shape[1] != 8:
        raise ValueError(f"Expected trajectory file with shape (N, 8), got {data.shape}.")

    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quaternions = data[:, 4:8]
    poses = np.stack([pose7_to_matrix(np.concatenate((p, q))) for p, q in zip(positions, quaternions)], axis=0)
    return GroundTruthTrajectory(
        timestamps=timestamps,
        positions=positions,
        quaternions_xyzw=quaternions,
        poses_4x4=poses,
    )


def quaternion_to_pose_matrix(position_xyz: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    return pose7_to_matrix(np.concatenate((position_xyz, quaternion_xyzw)))


def quaternion_to_yaw(quaternion_xyzw: np.ndarray) -> float:
    return float(yaw_from_quaternion_xyzw(quaternion_xyzw))

