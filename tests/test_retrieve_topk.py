from __future__ import annotations

from retrieval.build_patch_database import build_patch_database
from retrieval.retrieve_topk import retrieve_topk_for_frame


def test_retrieve_topk_contains_gt_patch_for_matching_query(synthetic_dataset, tmp_path) -> None:
    config = {
        "sequence_root": str(synthetic_dataset["sequence_root"]),
        "calibration_path": str(synthetic_dataset["calibration_path"]),
        "map_path": str(synthetic_dataset["map_path"]),
        "cache_dir": str(tmp_path / "cache"),
        "topk": 8,
    }
    database = build_patch_database(config, output_path=tmp_path / "descriptor_bank.npz")
    result = retrieve_topk_for_frame(
        frame_idx=0,
        config=config,
        descriptor_bank_path=database["descriptor_bank_path"],
    )
    assert result["gt_patch_id"] == 0
    assert result["gt_in_topk"] is True
