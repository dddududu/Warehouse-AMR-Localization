from __future__ import annotations

import numpy as np

from calibration.calibration_parser import parse_calibration_file


def test_parse_calibration_file(synthetic_dataset) -> None:
    calibration = parse_calibration_file(synthetic_dataset["calibration_path"])

    assert calibration.camera_left.width == 6
    assert calibration.camera_right.height == 4
    assert calibration.camera_left.build_K()[0, 0] == 100.0
    assert np.allclose(calibration.T_os_cam_left @ calibration.T_cam1_os.matrix, np.eye(4))
    assert np.allclose(calibration.T_cam_left_imu_left @ calibration.T_imu1_cam1.matrix, np.eye(4))

