from __future__ import annotations

import numpy as np

from geometry.projections import project_points, unproject_pixels
from geometry.se3 import compose_transform, invert_transform, pose7_to_matrix, transform_points
from geometry.yaw_utils import yaw_from_quaternion_xyzw


def test_geometry_helpers_round_trip() -> None:
    pose = pose7_to_matrix([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    inverse = invert_transform(pose)
    assert np.allclose(compose_transform(pose, inverse), np.eye(4))

    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    transformed = transform_points(pose, points)
    assert np.allclose(transformed[0], [1.0, 2.0, 3.0])
    assert yaw_from_quaternion_xyzw([0.0, 0.0, 0.0, 1.0]) == 0.0


def test_projection_helpers() -> None:
    K = np.array([[100.0, 0.0, 3.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]])
    points = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 2.0]], dtype=np.float64)
    uv, valid = project_points(points, K)
    assert valid.tolist() == [True, True]
    back_projected = unproject_pixels(uv, points[:, 2], K)
    assert np.allclose(back_projected, points.astype(np.float32))

