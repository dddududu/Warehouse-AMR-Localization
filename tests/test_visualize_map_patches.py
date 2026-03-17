from __future__ import annotations

from pathlib import Path

from analysis.visualize_map_patches import visualize_map_patches


def test_visualize_map_patches_writes_four_channel_images(synthetic_dataset, tmp_path) -> None:
    report = visualize_map_patches(
        {
            "sequence_root": str(synthetic_dataset["sequence_root"]),
            "calibration_path": str(synthetic_dataset["calibration_path"]),
            "map_path": str(synthetic_dataset["map_path"]),
            "cache_dir": str(tmp_path / "cache"),
        },
        output_dir=tmp_path / "images",
        prefix="synthetic",
    )
    assert len(report["saved_files"]) == 4
    for image_path in report["saved_files"]:
        assert Path(image_path).is_file()


def test_visualize_map_patches_writes_individual_patch_images(synthetic_dataset, tmp_path) -> None:
    report = visualize_map_patches(
        {
            "sequence_root": str(synthetic_dataset["sequence_root"]),
            "calibration_path": str(synthetic_dataset["calibration_path"]),
            "map_path": str(synthetic_dataset["map_path"]),
            "cache_dir": str(tmp_path / "cache"),
        },
        output_dir=tmp_path / "individual",
        prefix="synthetic_patch",
        export_individual=True,
    )
    assert report["saved_file_count"] == 4
    for image_path in report["saved_files"]:
        assert Path(image_path).is_file()
