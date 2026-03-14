from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from calibration.calibration_parser import Calibration, parse_calibration_file
from calibration.camera_model import CameraModel
from dataset_io.depth_loader import load_depth_png
from dataset_io.frame_indexer import DatasetConsistencyError, FrameRecord, build_frame_index, read_frame_times
from dataset_io.gt_loader import GroundTruthTrajectory, load_ground_truth
from dataset_io.image_loader import load_rgb_image
from dataset_io.imu_loader import IMULoader
from dataset_io.lidar_loader import load_pcd_xyz


@dataclass(frozen=True)
class WarehouseSequenceConfig:
    sequence_root: str | None = None
    calibration_path: str | None = None
    use_undistort: bool = False
    image_resize: tuple[int, int] | None = None
    depth_scale: float | None = None
    imu_time_window_sec: float = 0.1
    cache_dir: str | None = None
    load_images: bool = False
    load_depth: bool = False
    load_lidar: bool = False
    validate_timestamps: bool = True
    timestamp_tolerance_sec: float = 1e-6

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "WarehouseSequenceConfig":
        if mapping is None:
            return cls()
        payload = dict(mapping)
        resize = payload.get("image_resize")
        if resize is not None:
            if len(resize) != 2:
                raise ValueError("image_resize must be [height, width] or null.")
            payload["image_resize"] = (int(resize[0]), int(resize[1]))
        return cls(**payload)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "WarehouseSequenceConfig":
        data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("Dataset config YAML must parse to a mapping.")
        return cls.from_mapping(data)


def load_dataset_config(config: WarehouseSequenceConfig | Mapping[str, Any] | str | Path | None) -> WarehouseSequenceConfig:
    if config is None:
        return WarehouseSequenceConfig()
    if isinstance(config, WarehouseSequenceConfig):
        return config
    if isinstance(config, Mapping):
        return WarehouseSequenceConfig.from_mapping(config)
    return WarehouseSequenceConfig.from_yaml(config)


class WarehouseSequenceDataset:
    def __init__(
        self,
        sequence_root: str | Path | None = None,
        calibration_path: str | Path | None = None,
        config: WarehouseSequenceConfig | Mapping[str, Any] | str | Path | None = None,
    ) -> None:
        self.config = load_dataset_config(config)
        sequence_root_value = sequence_root or self.config.sequence_root
        calibration_path_value = calibration_path or self.config.calibration_path
        if not sequence_root_value:
            raise ValueError("sequence_root must be provided directly or via config.")
        if not calibration_path_value:
            raise ValueError("calibration_path must be provided directly or via config.")
        self.sequence_root = Path(sequence_root_value)
        self.calibration_path = Path(calibration_path_value)

        self.calibration: Calibration = parse_calibration_file(self.calibration_path)
        self.frame_index: list[FrameRecord] = build_frame_index(self.sequence_root)
        self.frame_times = read_frame_times(self.sequence_root / "frame_times.txt")
        self.ground_truth: GroundTruthTrajectory = load_ground_truth(self.sequence_root / "traj_gt.txt")
        self.imu_left = IMULoader.from_file(self.sequence_root / "imu_left.txt")
        self.imu_right = IMULoader.from_file(self.sequence_root / "imu_right.txt")

        if len(self.frame_index) != len(self.ground_truth.timestamps):
            raise DatasetConsistencyError(
                f"Frame index length ({len(self.frame_index)}) does not match trajectory length "
                f"({len(self.ground_truth.timestamps)})."
            )
        if self.config.validate_timestamps and not np.allclose(
            self.frame_times,
            self.ground_truth.timestamps,
            atol=self.config.timestamp_tolerance_sec,
            rtol=0.0,
        ):
            raise DatasetConsistencyError("frame_times.txt and traj_gt.txt timestamps are inconsistent.")

    def __len__(self) -> int:
        return len(self.frame_index)

    def _resolve_index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Frame index out of range: {index}")
        return index

    def _build_sample(self, frame_idx: int, record: FrameRecord) -> dict[str, Any]:
        gt_pose_7 = np.concatenate(
            (
                self.ground_truth.positions[frame_idx],
                self.ground_truth.quaternions_xyzw[frame_idx],
            )
        ).astype(np.float64)
        sample = {
            "frame_idx": frame_idx,
            "timestamp": float(record.timestamp),
            "image_left_path": str(record.image_left_path),
            "image_right_path": str(record.image_right_path),
            "depth_left_path": str(record.depth_left_path),
            "depth_right_path": str(record.depth_right_path),
            "lidar_path": str(record.lidar_path),
            "segmentation_color_left_path": str(record.segmentation_color_left_path)
            if record.segmentation_color_left_path
            else None,
            "segmentation_color_right_path": str(record.segmentation_color_right_path)
            if record.segmentation_color_right_path
            else None,
            "segmentation_greyscale_left_path": str(record.segmentation_greyscale_left_path)
            if record.segmentation_greyscale_left_path
            else None,
            "segmentation_greyscale_right_path": str(record.segmentation_greyscale_right_path)
            if record.segmentation_greyscale_right_path
            else None,
            "gt_pose_7": gt_pose_7,
            "gt_pose_4x4": self.ground_truth.poses_4x4[frame_idx].copy(),
            "imu_left": self.imu_left.query_time_window(record.timestamp, self.config.imu_time_window_sec),
            "imu_right": self.imu_right.query_time_window(record.timestamp, self.config.imu_time_window_sec),
        }

        if self.config.load_images:
            left_image, left_camera = load_rgb_image(
                record.image_left_path,
                camera_model=self.calibration.camera_left,
                use_undistort=self.config.use_undistort,
                resize_hw=self.config.image_resize,
            )
            right_image, right_camera = load_rgb_image(
                record.image_right_path,
                camera_model=self.calibration.camera_right,
                use_undistort=self.config.use_undistort,
                resize_hw=self.config.image_resize,
            )
            sample["image_left"] = left_image
            sample["image_right"] = right_image
            sample["camera_left"] = left_camera
            sample["camera_right"] = right_camera
        if self.config.load_depth:
            sample["depth_left"] = load_depth_png(
                record.depth_left_path,
                depth_scale=self.config.depth_scale,
                resize_hw=self.config.image_resize,
            )
            sample["depth_right"] = load_depth_png(
                record.depth_right_path,
                depth_scale=self.config.depth_scale,
                resize_hw=self.config.image_resize,
            )
        if self.config.load_lidar:
            sample["lidar_points"] = load_pcd_xyz(record.lidar_path)
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        frame_idx = self._resolve_index(index)
        record = self.frame_index[frame_idx]
        return self._build_sample(frame_idx, record)

    @property
    def camera_left(self) -> CameraModel:
        return self.calibration.camera_left

    @property
    def camera_right(self) -> CameraModel:
        return self.calibration.camera_right
