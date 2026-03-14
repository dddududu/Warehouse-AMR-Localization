from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from calibration.camera_model import CameraModel
from geometry.se3 import invert_transform, pose7_to_matrix


@dataclass(frozen=True)
class TransformSpec:
    translation: np.ndarray
    quaternion_xyzw: np.ndarray
    matrix: np.ndarray


@dataclass(frozen=True)
class Calibration:
    camera_left: CameraModel
    camera_right: CameraModel
    camera_left_parent_frame: str
    camera_right_parent_frame: str
    camera_left_frame: str
    camera_right_frame: str
    imu_left_parent_frame: str
    imu_right_parent_frame: str
    imu_left_frame: str
    imu_right_frame: str
    T_cam1_os: TransformSpec
    T_cam2_os: TransformSpec
    T_imu1_cam1: TransformSpec
    T_imu2_cam2: TransformSpec
    T_os_cam_left: np.ndarray
    T_os_cam_right: np.ndarray
    T_cam_left_imu_left: np.ndarray
    T_cam_right_imu_right: np.ndarray


_LINE_PATTERN = re.compile(r"^(?P<key>[^:=]+)\s*(?P<sep>[:=])\s*(?P<value>.+)$")


def _parse_value(raw_value: str) -> Any:
    value = raw_value.strip()
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        try:
            return float(value)
        except ValueError:
            return value.strip('"')


def _load_key_values(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _LINE_PATTERN.match(stripped)
        if not match:
            raise ValueError(f"Unsupported calibration line format: {line!r}")
        key = match.group("key").strip()
        values[key] = _parse_value(match.group("value"))
    return values


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing calibration field: {key}")
    return mapping[key]


def _build_camera(mapping: dict[str, Any], prefix: str) -> CameraModel:
    return CameraModel(
        fx=float(_require(mapping, f"{prefix}.fx")),
        fy=float(_require(mapping, f"{prefix}.fy")),
        cx=float(_require(mapping, f"{prefix}.cx")),
        cy=float(_require(mapping, f"{prefix}.cy")),
        width=int(_require(mapping, f"{prefix}.width")),
        height=int(_require(mapping, f"{prefix}.height")),
        distortion=np.asarray(_require(mapping, f"{prefix}.D"), dtype=np.float64),
        model_type=str(_require(mapping, f"{prefix}.type")).strip('"'),
    )


def _build_transform(mapping: dict[str, Any], prefix: str) -> TransformSpec:
    pose7 = np.array(
        [
            float(_require(mapping, f"{prefix}.px")),
            float(_require(mapping, f"{prefix}.py")),
            float(_require(mapping, f"{prefix}.pz")),
            float(_require(mapping, f"{prefix}.qx")),
            float(_require(mapping, f"{prefix}.qy")),
            float(_require(mapping, f"{prefix}.qz")),
            float(_require(mapping, f"{prefix}.qw")),
        ],
        dtype=np.float64,
    )
    return TransformSpec(
        translation=pose7[:3].copy(),
        quaternion_xyzw=pose7[3:].copy(),
        matrix=pose7_to_matrix(pose7),
    )


def parse_calibration_file(calibration_path: str | Path) -> Calibration:
    path = Path(calibration_path)
    values = _load_key_values(path)

    T_cam1_os = _build_transform(values, "T_cam1_os")
    T_cam2_os = _build_transform(values, "T_cam2_os")
    T_imu1_cam1 = _build_transform(values, "T_imu1_cam1")
    T_imu2_cam2 = _build_transform(values, "T_imu2_cam2")

    return Calibration(
        camera_left=_build_camera(values, "Camera1"),
        camera_right=_build_camera(values, "Camera2"),
        camera_left_parent_frame=str(_require(values, "Camera1.parent_frame_id")).strip('"'),
        camera_right_parent_frame=str(_require(values, "Camera2.parent_frame_id")).strip('"'),
        camera_left_frame=str(_require(values, "Camera1.frame_id")).strip('"'),
        camera_right_frame=str(_require(values, "Camera2.frame_id")).strip('"'),
        imu_left_parent_frame=str(_require(values, "Imu1.parent_frame_id")).strip('"'),
        imu_right_parent_frame=str(_require(values, "Imu2.parent_frame_id")).strip('"'),
        imu_left_frame=str(_require(values, "Imu1.frame_id")).strip('"'),
        imu_right_frame=str(_require(values, "Imu2.frame_id")).strip('"'),
        T_cam1_os=T_cam1_os,
        T_cam2_os=T_cam2_os,
        T_imu1_cam1=T_imu1_cam1,
        T_imu2_cam2=T_imu2_cam2,
        T_os_cam_left=invert_transform(T_cam1_os.matrix),
        T_os_cam_right=invert_transform(T_cam2_os.matrix),
        T_cam_left_imu_left=invert_transform(T_imu1_cam1.matrix),
        T_cam_right_imu_right=invert_transform(T_imu2_cam2.matrix),
    )
