from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


CALIBRATION_TEXT = """# Sensor Calibration
Camera1.parent_frame_id = "os_sensor"
Camera1.frame_id = "azure_left_camera"
Camera1.type: "PinHole"
Camera1.fx: 100.0
Camera1.cx: 3.0
Camera1.fy: 110.0
Camera1.cy: 2.0
Camera1.width: 6
Camera1.height: 4
Camera1.D = [0.1, -0.01, 0.001, 0.002, 0.0]
Camera2.parent_frame_id = "os_sensor"
Camera2.frame_id = "azure_right_camera"
Camera2.type: "PinHole"
Camera2.fx: 101.0
Camera2.cx: 3.5
Camera2.fy: 111.0
Camera2.cy: 2.5
Camera2.width: 6
Camera2.height: 4
Camera2.D = [0.09, -0.02, 0.001, 0.001, 0.0]
T_cam1_os.px: 0.1
T_cam1_os.py: 0.0
T_cam1_os.pz: -0.2
T_cam1_os.qx: 0.0
T_cam1_os.qy: 0.0
T_cam1_os.qz: 0.0
T_cam1_os.qw: 1.0
T_cam2_os.px: -0.1
T_cam2_os.py: 0.0
T_cam2_os.pz: -0.2
T_cam2_os.qx: 0.0
T_cam2_os.qy: 0.0
T_cam2_os.qz: 0.0
T_cam2_os.qw: 1.0
Imu1.parent_frame_id = "azure_left_camera"
Imu1.frame_id = "azure_left_imu"
T_imu1_cam1.px: 0.01
T_imu1_cam1.py: 0.0
T_imu1_cam1.pz: 0.0
T_imu1_cam1.qx: 0.0
T_imu1_cam1.qy: 0.0
T_imu1_cam1.qz: 0.0
T_imu1_cam1.qw: 1.0
Imu2.parent_frame_id = "azure_right_camera"
Imu2.frame_id = "azure_right_imu"
T_imu2_cam2.px: -0.01
T_imu2_cam2.py: 0.0
T_imu2_cam2.pz: 0.0
T_imu2_cam2.qx: 0.0
T_imu2_cam2.qy: 0.0
T_imu2_cam2.qz: 0.0
T_imu2_cam2.qw: 1.0
"""


def _write_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def _write_pcd(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {points.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {points.shape[0]}\n"
        "DATA binary\n"
    ).encode("ascii")
    with path.open("wb") as file_obj:
        file_obj.write(header)
        file_obj.write(points.tobytes())


def _write_ply(path: Path, xyz: np.ndarray, normals: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as file_obj:
        file_obj.write(header)
        for point, normal in zip(xyz.astype(np.float32), normals.astype(np.float32)):
            file_obj.write(struct.pack("<6f", *(point.tolist() + normal.tolist())))


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "sequence"
    for name in ("image_left", "image_right", "depth_left", "depth_right", "lidar"):
        (root / name).mkdir(parents=True, exist_ok=True)

    calibration_path = root / "calibrations.txt"
    calibration_path.write_text(CALIBRATION_TEXT, encoding="utf-8")

    frame_times = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    np.savetxt(root / "frame_times.txt", frame_times, fmt="%.6f")

    traj = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.04997917, 0.99875026],
            [0.2, 2.0, 0.5, 0.0, 0.0, 0.0, 0.09983342, 0.99500417],
        ],
        dtype=np.float64,
    )
    np.savetxt(root / "traj_gt.txt", traj, fmt="%.8f")

    imu = np.array(
        [
            [0.00, 0.1, 0.0, 0.0, 0.0, 0.0, 9.8],
            [0.05, 0.1, 0.0, 0.0, 0.0, 0.0, 9.8],
            [0.10, 0.1, 0.0, 0.0, 0.0, 0.0, 9.8],
            [0.15, 0.1, 0.0, 0.0, 0.0, 0.0, 9.8],
            [0.20, 0.1, 0.0, 0.0, 0.0, 0.0, 9.8],
        ],
        dtype=np.float64,
    )
    np.savetxt(root / "imu_left.txt", imu, fmt="%.6f")
    np.savetxt(root / "imu_right.txt", imu, fmt="%.6f")

    for frame_idx in range(3):
        rgb = np.full((4, 6, 3), frame_idx * 40, dtype=np.uint8)
        depth = np.full((4, 6), 1000 + frame_idx, dtype=np.uint16)
        points = np.array(
            [
                [frame_idx + 0.0, 0.0, 1.0],
                [frame_idx + 1.0, 1.0, 2.0],
                [frame_idx + 2.0, -1.0, 3.0],
            ],
            dtype=np.float32,
        )
        _write_png(root / "image_left" / f"{frame_idx:06d}.png", rgb)
        _write_png(root / "image_right" / f"{frame_idx:06d}.png", rgb + 1)
        _write_png(root / "depth_left" / f"{frame_idx:06d}.png", depth)
        _write_png(root / "depth_right" / f"{frame_idx:06d}.png", depth + 1)
        _write_pcd(root / "lidar" / f"{frame_idx:06d}.pcd", points)

    map_xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    map_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1))
    _write_ply(root / "groundtruth_map.ply", map_xyz, map_normals)

    return {
        "sequence_root": root,
        "calibration_path": calibration_path,
        "map_path": root / "groundtruth_map.ply",
    }

