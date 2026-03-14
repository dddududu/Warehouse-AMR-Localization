from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from dataset_io.lidar_loader import load_pcd_xyz, read_pcd_header
from dataset_io.sequence_dataset import WarehouseSequenceDataset, load_dataset_config
from geometry.yaw_utils import wrap_to_pi, yaw_from_quaternion_xyzw


def _stat_dict(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def analyze_sequence(
    sequence_root: str | Path | None = None,
    calibration_path: str | Path | None = None,
    config: str | Path | dict[str, Any] | None = None,
    lidar_stride: int = 1,
    output_json: str | Path | None = None,
) -> dict:
    dataset_config = load_dataset_config(config)
    sequence_root = sequence_root or dataset_config.sequence_root
    calibration_path = calibration_path or dataset_config.calibration_path
    dataset = WarehouseSequenceDataset(sequence_root=sequence_root, calibration_path=calibration_path, config=dataset_config)

    timestamps = dataset.frame_times
    duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    frame_rate_hz = float((len(timestamps) - 1) / duration) if duration > 0.0 else None

    lidar_counts: list[int] = []
    lidar_xyz_min: np.ndarray | None = None
    lidar_xyz_max: np.ndarray | None = None
    sampled_frame_indices = list(range(0, len(dataset), max(1, int(lidar_stride))))
    for frame_idx in sampled_frame_indices:
        record = dataset.frame_index[frame_idx]
        header = read_pcd_header(record.lidar_path)
        lidar_counts.append(header.points)
        points = load_pcd_xyz(record.lidar_path)
        frame_min = points.min(axis=0)
        frame_max = points.max(axis=0)
        lidar_xyz_min = frame_min if lidar_xyz_min is None else np.minimum(lidar_xyz_min, frame_min)
        lidar_xyz_max = frame_max if lidar_xyz_max is None else np.maximum(lidar_xyz_max, frame_max)

    positions = dataset.ground_truth.positions
    yaws = np.array([yaw_from_quaternion_xyzw(q) for q in dataset.ground_truth.quaternions_xyzw], dtype=np.float64)
    relative_translation = np.linalg.norm(np.diff(positions, axis=0), axis=1) if len(positions) > 1 else np.array([])
    relative_yaw = wrap_to_pi(np.diff(yaws)) if len(yaws) > 1 else np.array([])

    report = {
        "sequence_root": str(dataset.sequence_root),
        "calibration_path": str(dataset.calibration_path),
        "num_frames": len(dataset),
        "time_range_sec": [float(timestamps[0]), float(timestamps[-1])] if len(timestamps) else [None, None],
        "duration_sec": duration,
        "average_frame_rate_hz": frame_rate_hz,
        "imu_frequency_hz": {
            "left": dataset.imu_left.estimate_frequency_hz(),
            "right": dataset.imu_right.estimate_frequency_hz(),
        },
        "lidar_analysis_frame_count": len(sampled_frame_indices),
        "lidar_point_count": _stat_dict(np.asarray(lidar_counts, dtype=np.float64)),
        "lidar_xyz_min": lidar_xyz_min.tolist() if lidar_xyz_min is not None else None,
        "lidar_xyz_max": lidar_xyz_max.tolist() if lidar_xyz_max is not None else None,
        "trajectory_position_min": positions.min(axis=0).tolist(),
        "trajectory_position_max": positions.max(axis=0).tolist(),
        "yaw_range_rad": [float(np.min(yaws)), float(np.max(yaws))],
        "relative_translation_m": _stat_dict(relative_translation),
        "relative_yaw_rad": _stat_dict(relative_yaw),
    }

    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a TorWIC sequence directory.")
    parser.add_argument("--sequence-root", default=None)
    parser.add_argument("--calibration-path", default=None)
    parser.add_argument("--config", default="configs/dataset_default.yaml")
    parser.add_argument("--lidar-stride", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    analyze_sequence(
        sequence_root=args.sequence_root,
        calibration_path=args.calibration_path,
        config=args.config,
        lidar_stride=args.lidar_stride,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()

