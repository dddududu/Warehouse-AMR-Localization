from __future__ import annotations

from dataset_io.sequence_dataset import WarehouseSequenceDataset


def test_sequence_dataset_getitem_loads_modalities(synthetic_dataset) -> None:
    dataset = WarehouseSequenceDataset(
        sequence_root=synthetic_dataset["sequence_root"],
        calibration_path=synthetic_dataset["calibration_path"],
        config={
            "load_images": True,
            "load_depth": True,
            "load_lidar": True,
            "image_resize": [2, 3],
            "imu_time_window_sec": 0.11,
        },
    )

    sample = dataset[1]
    assert len(dataset) == 3
    assert sample["frame_idx"] == 1
    assert sample["image_left"].shape == (2, 3, 3)
    assert sample["depth_left"].shape == (2, 3)
    assert sample["lidar_points"].shape == (3, 3)
    assert sample["gt_pose_7"].shape == (7,)
    assert sample["gt_pose_4x4"].shape == (4, 4)
    assert sample["camera_left"].width == 3
    assert sample["imu_left"].timestamps.size >= 1

