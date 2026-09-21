import numpy as np


def test_resume_state_round_trip_preserves_pose_bits(tmp_path) -> None:
    state_path = tmp_path / "state.npz"
    poses = np.array(
        [
            [
                [1.0, 0.0, 0.0, 0.12345678901234567],
                [0.0, 1.0, 0.0, -0.9876543210987654],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float64,
    )
    np.savez_compressed(state_path, frame_indices=np.array([7], dtype=np.int64), accepted_poses=poses)

    with np.load(state_path) as state:
        restored = np.asarray(state["accepted_poses"], dtype=np.float64)

    assert np.array_equal(restored, poses)
