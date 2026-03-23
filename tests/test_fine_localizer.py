from __future__ import annotations

from pathlib import Path

import yaml

from localization.fine_localizer import localize_sequence


def test_fine_localizer_runs_on_synthetic_dataset(synthetic_dataset, tmp_path: Path) -> None:
    coarse_config = {
        "sequence_root": str(synthetic_dataset["sequence_root"]),
        "calibration_path": str(synthetic_dataset["calibration_path"]),
        "map_path": str(synthetic_dataset["map_path"]),
        "cache_dir": str(tmp_path / "coarse_cache"),
        "topk": 3,
        "patch_size_m": 20.0,
        "patch_stride_m": 10.0,
        "device": "cpu",
        "use_query_rotation_search": True,
        "query_rotation_search_step_deg": 10.0,
    }
    coarse_config_path = tmp_path / "coarse_config.yaml"
    coarse_config_path.write_text(yaml.safe_dump(coarse_config, sort_keys=False), encoding="utf-8")

    report = localize_sequence(
        config={
            "coarse_config_path": str(coarse_config_path),
            "coarse_checkpoint_path": None,
            "descriptor_bank_path": str(tmp_path / "descriptor_bank.npz"),
            "topk_candidates": 3,
            "local_submap_size_m": 20.0,
            "local_submap_resolution": 0.5,
            "coarse_yaw_half_range_deg": 10.0,
            "coarse_yaw_step_deg": 5.0,
            "voxel_size_m": 0.5,
            "icp_max_iterations": 5,
            "icp_max_correspondence_distance_m": 2.0,
            "icp_min_correspondences": 1,
            "output_json": str(tmp_path / "fine_report.json"),
        },
        frame_start=0,
        num_frames=1,
    )
    assert report["num_eval_frames"] == 1
    assert report["mean_position_error_m"] is not None
    assert report["frame_results"][0]["candidate_results"]
