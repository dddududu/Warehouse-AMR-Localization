import numpy as np

from localization.icp_refiner import refine_pose_with_icp


def test_ckdtree_backend_recovers_planar_translation() -> None:
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    target = source + np.array([0.2, -0.1, 0.0], dtype=np.float32)
    result = refine_pose_with_icp(
        source,
        target,
        np.eye(4, dtype=np.float64),
        voxel_size_m=0.01,
        max_correspondence_distance_m=1.0,
        min_correspondences=3,
        nearest_neighbor_backend="ckdtree",
    )

    assert result.num_inliers == 4
    assert np.allclose(result.pose_4x4[:2, 3], [0.2, -0.1], atol=1.0e-5)
