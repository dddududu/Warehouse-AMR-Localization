from __future__ import annotations

from pathlib import Path

from analysis.visualize_query_bevs import visualize_query_bevs


def test_visualize_query_bevs_writes_first_two_frames(synthetic_dataset, tmp_path) -> None:
    report = visualize_query_bevs(
        {
            "sequence_root": str(synthetic_dataset["sequence_root"]),
            "calibration_path": str(synthetic_dataset["calibration_path"]),
            "map_path": str(synthetic_dataset["map_path"]),
            "cache_dir": str(tmp_path / "cache"),
        },
        frame_indices=[0, 1],
        output_dir=tmp_path / "query_images",
        prefix="query",
    )
    assert report["num_frames"] == 2
    assert report["saved_file_count"] == 8
    for frame in report["frames"]:
        assert len(frame["saved_files"]) == 4
        for image_path in frame["saved_files"]:
            assert Path(image_path).is_file()
