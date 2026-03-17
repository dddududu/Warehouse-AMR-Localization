from __future__ import annotations

from retrieval.build_patch_database import build_patch_database
from retrieval.evaluate_retrieval import evaluate_retrieval


def test_evaluate_retrieval_reports_accuracy(synthetic_dataset, tmp_path) -> None:
    config = {
        "sequence_root": str(synthetic_dataset["sequence_root"]),
        "calibration_path": str(synthetic_dataset["calibration_path"]),
        "map_path": str(synthetic_dataset["map_path"]),
        "cache_dir": str(tmp_path / "cache"),
        "topk": 1,
        "use_query_rotation_search": True,
        "query_rotation_search_step_deg": 10.0,
        "device": "cpu",
    }
    database = build_patch_database(config, output_path=tmp_path / "descriptor_bank.npz")
    report = evaluate_retrieval(
        config=config,
        descriptor_bank_path=database["descriptor_bank_path"],
        frame_start=0,
        num_frames=2,
    )
    assert report["num_eval_frames"] == 2
    assert report["topk"] == 1
    assert report["top1_accuracy"] is not None
