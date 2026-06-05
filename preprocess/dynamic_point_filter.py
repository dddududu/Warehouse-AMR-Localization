from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from calibration.calibration_parser import Calibration
from calibration.camera_model import CameraModel
from geometry.se3 import transform_points


def _load_dynamic_mask(
    label_path: str | Path | None,
    dynamic_labels: tuple[int, ...],
    dilation_px: int,
) -> np.ndarray | None:
    if label_path is None:
        return None
    label_image = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if label_image is None:
        return None
    mask = np.isin(label_image, dynamic_labels).astype(np.uint8)
    if dilation_px > 0:
        kernel_size = 2 * int(dilation_px) + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)


def semantic_label_ratio(
    segmentation_left_path: str | Path | None,
    segmentation_right_path: str | Path | None,
    labels: Iterable[int],
) -> float:
    dynamic_labels = tuple(int(label) for label in labels)
    ratios = []
    for path in (segmentation_left_path, segmentation_right_path):
        mask = _load_dynamic_mask(path, dynamic_labels, dilation_px=0)
        if mask is not None:
            ratios.append(float(mask.mean()))
    return max(ratios, default=0.0)


def _points_inside_mask(
    points_sensor: np.ndarray,
    camera: CameraModel,
    transform_camera_sensor: np.ndarray,
    dynamic_mask: np.ndarray | None,
) -> np.ndarray:
    inside = np.zeros(points_sensor.shape[0], dtype=bool)
    if dynamic_mask is None or points_sensor.size == 0:
        return inside
    points_camera = transform_points(transform_camera_sensor, points_sensor).astype(np.float64)
    finite = np.isfinite(points_camera).all(axis=1)
    projected = np.full((points_sensor.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(finite):
        image_points, _ = cv2.projectPoints(
            points_camera[finite],
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            camera.build_K(),
            camera.distortion,
        )
        projected[finite] = image_points.reshape(-1, 2)
    rounded = np.zeros_like(projected, dtype=np.int64)
    valid_numbers = np.isfinite(projected).all(axis=1) & (np.abs(projected).max(axis=1) < 1.0e9)
    rounded[valid_numbers] = np.rint(projected[valid_numbers]).astype(np.int64)
    valid = (
        finite
        & valid_numbers
        & (points_camera[:, 2] > 0.0)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < dynamic_mask.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < dynamic_mask.shape[0])
    )
    inside[valid] = dynamic_mask[rounded[valid, 1], rounded[valid, 0]]
    return inside


def filter_dynamic_points_by_semantics(
    points_sensor: np.ndarray,
    calibration: Calibration,
    segmentation_left_path: str | Path | None,
    segmentation_right_path: str | Path | None,
    dynamic_labels: Iterable[int] = (12, 13, 14, 15),
    dilation_px: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_sensor, dtype=np.float32)
    labels = tuple(int(label) for label in dynamic_labels)
    left_mask = _load_dynamic_mask(segmentation_left_path, labels, int(dilation_px))
    right_mask = _load_dynamic_mask(segmentation_right_path, labels, int(dilation_px))
    dynamic_left = _points_inside_mask(
        points,
        calibration.camera_left,
        calibration.T_cam1_os.matrix,
        left_mask,
    )
    dynamic_right = _points_inside_mask(
        points,
        calibration.camera_right,
        calibration.T_cam2_os.matrix,
        right_mask,
    )
    dynamic = dynamic_left | dynamic_right
    return points[~dynamic].astype(np.float32, copy=False), dynamic
