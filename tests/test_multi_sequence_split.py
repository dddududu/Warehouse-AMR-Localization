from __future__ import annotations

from dataset_io.retrieval_dataset import CoarseRetrievalDataset
from retrieval.config import load_coarse_retrieval_config


def test_multi_sequence_config_splits_without_leakage(synthetic_multi_sequence_dataset, tmp_path) -> None:
    config = load_coarse_retrieval_config(
        {
            "dataset_parent_root": str(synthetic_multi_sequence_dataset["dataset_parent_root"]),
            "sequence_names": synthetic_multi_sequence_dataset["sequence_names"],
            "val_sequence_names": synthetic_multi_sequence_dataset["val_sequence_names"],
            "shared_calibration_path": str(
                synthetic_multi_sequence_dataset["dataset_parent_root"] / "Seq_A" / "Seq_A" / "calibrations.txt"
            ),
            "shared_map_path": str(
                synthetic_multi_sequence_dataset["dataset_parent_root"] / "Seq_A" / "Seq_A" / "groundtruth_map.ply"
            ),
            "cache_dir": str(tmp_path / "cache"),
            "topk": 3,
            "query_rotation_search_step_deg": 30.0,
        }
    )
    train_entries, val_entries = config.split_sequence_entries()
    assert [entry["sequence_name"] for entry in train_entries] == ["Seq_A", "Seq_B"]
    assert [entry["sequence_name"] for entry in val_entries] == ["Seq_C"]
    assert len({entry["calibration_path"] for entry in train_entries + val_entries}) == 1
    assert len({entry["map_path"] for entry in train_entries + val_entries}) == 1

    train_dataset = CoarseRetrievalDataset(config, sequence_entries=train_entries, align_query_to_gt_yaw=True)
    val_dataset = CoarseRetrievalDataset(config, sequence_entries=val_entries, align_query_to_gt_yaw=False)
    assert len(train_dataset.sequence_resources) == 2
    assert len(val_dataset.sequence_resources) == 1
    assert {item["sequence_name"] for item in (train_dataset[0], train_dataset[len(train_dataset) - 1])}.issubset(
        {"Seq_A", "Seq_B"}
    )
    assert val_dataset[0]["sequence_name"] == "Seq_C"
