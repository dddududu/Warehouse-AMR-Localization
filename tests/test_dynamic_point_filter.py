from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from calibration.camera_model import CameraModel
from preprocess.dynamic_point_filter import filter_dynamic_points_by_semantics, semantic_label_ratio


def test_filter_dynamic_points_by_semantics_removes_projected_person(tmp_path: Path) -> None:
    camera = CameraModel(
        fx=10.0,
        fy=10.0,
        cx=10.0,
        cy=10.0,
        width=20,
        height=20,
        distortion=np.zeros(5, dtype=np.float64),
    )
    identity_spec = SimpleNamespace(matrix=np.eye(4, dtype=np.float64))
    calibration = SimpleNamespace(
        camera_left=camera,
        camera_right=camera,
        T_cam1_os=identity_spec,
        T_cam2_os=identity_spec,
    )
    labels = np.zeros((20, 20), dtype=np.uint8)
    labels[10, 10] = 13
    label_path = tmp_path / "labels.png"
    cv2.imwrite(str(label_path), labels)
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    filtered, dynamic_mask = filter_dynamic_points_by_semantics(
        points,
        calibration=calibration,
        segmentation_left_path=label_path,
        segmentation_right_path=None,
        dynamic_labels=(13,),
        dilation_px=0,
    )

    assert dynamic_mask.tolist() == [True, False, False]
    assert np.allclose(filtered, points[1:])


def test_semantic_label_ratio_uses_larger_camera_ratio(tmp_path: Path) -> None:
    left = np.zeros((10, 10), dtype=np.uint8)
    right = np.zeros((10, 10), dtype=np.uint8)
    left[:2] = 13
    right[:4] = 13
    left_path = tmp_path / "left.png"
    right_path = tmp_path / "right.png"
    cv2.imwrite(str(left_path), left)
    cv2.imwrite(str(right_path), right)

    ratio = semantic_label_ratio(left_path, right_path, labels=(13,))

    assert ratio == 0.4
