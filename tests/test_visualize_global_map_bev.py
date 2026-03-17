from __future__ import annotations

from pathlib import Path

from analysis.visualize_global_map_bev import visualize_global_map_bev


def test_visualize_global_map_bev_writes_four_complete_channel_images(synthetic_dataset, tmp_path) -> None:
    report = visualize_global_map_bev(
        {
            "sequence_root": str(synthetic_dataset["sequence_root"]),
            "calibration_path": str(synthetic_dataset["calibration_path"]),
            "map_path": str(synthetic_dataset["map_path"]),
            "cache_dir": str(tmp_path / "cache"),
        },
        output_dir=tmp_path / "global_bev",
        prefix="synthetic_global",
    )
    assert report["global_bev_shape"][0] == 4
    assert len(report["saved_files"]) == 4
    for image_path in report["saved_files"]:
        assert Path(image_path).is_file()
