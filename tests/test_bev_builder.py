from __future__ import annotations

import numpy as np

from preprocess.bev_builder import BEVConfig, points_to_bev


def test_bev_builder_uses_consistent_cell_mapping() -> None:
    config = BEVConfig(x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0, resolution=0.1)
    points = np.array(
        [
            [-9.95, -9.95, 1.0],
            [-9.95, -9.95, 3.0],
            [9.95, 9.95, 2.0],
        ],
        dtype=np.float32,
    )
    bev = points_to_bev(points, config)
    assert bev.shape == (4, 200, 200)
    assert bev[0, 199, 0] == 2.0
    assert bev[1, 199, 0] == 3.0
    assert bev[2, 199, 0] == 2.0
    assert bev[3, 199, 0] == 1.0
    assert bev[0, 0, 199] == 1.0
    assert bev[1, 10, 10] == 0.0
    assert bev[2, 10, 10] == 0.0
