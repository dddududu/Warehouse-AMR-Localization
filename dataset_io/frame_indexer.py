from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DatasetConsistencyError(ValueError):
    """Raised when the sequence directory contents are inconsistent."""


@dataclass(frozen=True)
class FrameRecord:
    frame_idx: int
    timestamp: float
    image_left_path: Path
    image_right_path: Path
    depth_left_path: Path
    depth_right_path: Path
    lidar_path: Path
    segmentation_color_left_path: Path | None = None
    segmentation_color_right_path: Path | None = None
    segmentation_greyscale_left_path: Path | None = None
    segmentation_greyscale_right_path: Path | None = None


def read_frame_times(frame_times_path: str | Path) -> np.ndarray:
    path = Path(frame_times_path)
    timestamps = np.loadtxt(path, dtype=np.float64, ndmin=1)
    if timestamps.ndim != 1:
        raise DatasetConsistencyError(f"Expected 1-D frame_times, got shape {timestamps.shape}.")
    return timestamps


def _count_lines(text_path: Path) -> int:
    with text_path.open("r", encoding="utf-8") as file_obj:
        return sum(1 for _ in file_obj)


def _enumerate_files(directory: Path, suffix: str, expected_count: int) -> list[Path]:
    if not directory.is_dir():
        raise DatasetConsistencyError(f"Missing required directory: {directory}")
    paths = sorted(directory.glob(f"*{suffix}"))
    if len(paths) != expected_count:
        raise DatasetConsistencyError(
            f"{directory.name} count mismatch: expected {expected_count}, found {len(paths)}."
        )
    for index, path in enumerate(paths):
        expected_name = f"{index:06d}{suffix}"
        if path.name != expected_name:
            raise DatasetConsistencyError(
                f"{directory.name} naming is not continuous: expected {expected_name}, found {path.name}."
            )
    return paths


def _enumerate_optional_files(directory: Path, suffix: str, expected_count: int) -> list[Path | None]:
    if not directory.exists():
        return [None] * expected_count
    return list(_enumerate_files(directory, suffix, expected_count))


def build_frame_index(sequence_root: str | Path) -> list[FrameRecord]:
    root = Path(sequence_root)
    frame_times_path = root / "frame_times.txt"
    traj_path = root / "traj_gt.txt"
    if not frame_times_path.is_file():
        raise DatasetConsistencyError(f"Missing frame_times file: {frame_times_path}")
    if not traj_path.is_file():
        raise DatasetConsistencyError(f"Missing ground-truth trajectory file: {traj_path}")

    timestamps = read_frame_times(frame_times_path)
    gt_count = _count_lines(traj_path)
    if len(timestamps) != gt_count:
        raise DatasetConsistencyError(
            f"frame_times count ({len(timestamps)}) does not match traj_gt count ({gt_count})."
        )

    image_left = _enumerate_files(root / "image_left", ".png", len(timestamps))
    image_right = _enumerate_files(root / "image_right", ".png", len(timestamps))
    depth_left = _enumerate_files(root / "depth_left", ".png", len(timestamps))
    depth_right = _enumerate_files(root / "depth_right", ".png", len(timestamps))
    lidar = _enumerate_files(root / "lidar", ".pcd", len(timestamps))
    seg_color_left = _enumerate_optional_files(root / "segmentation_color_left", ".png", len(timestamps))
    seg_color_right = _enumerate_optional_files(root / "segmentation_color_right", ".png", len(timestamps))
    seg_gray_left = _enumerate_optional_files(
        root / "segmentation_greyscale_left",
        ".png",
        len(timestamps),
    )
    seg_gray_right = _enumerate_optional_files(
        root / "segmentation_greyscale_right",
        ".png",
        len(timestamps),
    )

    records: list[FrameRecord] = []
    for idx, timestamp in enumerate(timestamps):
        records.append(
            FrameRecord(
                frame_idx=idx,
                timestamp=float(timestamp),
                image_left_path=image_left[idx],
                image_right_path=image_right[idx],
                depth_left_path=depth_left[idx],
                depth_right_path=depth_right[idx],
                lidar_path=lidar[idx],
                segmentation_color_left_path=seg_color_left[idx],
                segmentation_color_right_path=seg_color_right[idx],
                segmentation_greyscale_left_path=seg_gray_left[idx],
                segmentation_greyscale_right_path=seg_gray_right[idx],
            )
        )
    return records

