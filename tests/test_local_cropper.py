from __future__ import annotations

import numpy as np

from preprocess.local_lidar_cropper import LocalCropConfig, crop_local_lidar_points


def test_local_cropper_enforces_fixed_bounds() -> None:
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [10.0, 10.0, 4.73],
            [-10.0, -10.0, 0.13],
            [11.0, 0.0, 1.0],
            [0.0, -11.0, 1.0],
            [0.0, 0.0, 5.0],
            [np.nan, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cropped = crop_local_lidar_points(points, LocalCropConfig())
    assert np.all(cropped[:, 0] >= -10.0)
    assert np.all(cropped[:, 0] <= 10.0)
    assert np.all(cropped[:, 1] >= -10.0)
    assert np.all(cropped[:, 1] <= 10.0)
    assert np.all(cropped[:, 2] >= 0.13)
    assert np.all(cropped[:, 2] <= 4.73)
    assert cropped.shape[0] == 3

