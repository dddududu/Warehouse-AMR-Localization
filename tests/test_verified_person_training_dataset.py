import pytest

from experiments.target_motion_20260804.build_verified_person_training_dataset import build


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "manual_person_label": "person",
        "manual_motion_label": "moving",
        "manual_instance_id": "train_001",
        "manual_box_json": "[1, 2, 11, 22]",
        "window_image_paths_json": "[\"frame.png\"]",
        "split": "train",
        "sequence": "aisle_cw_run_1",
        "anchor_frame": "10",
    }
    row.update(overrides)
    return row


def test_build_exports_only_verified_person_records() -> None:
    coco, motions, incomplete = build([_row(), _row(manual_person_label="non_person")])
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 1
    assert motions[0]["motion_label"] == "moving"
    assert incomplete == []


def test_build_rejects_pending_record() -> None:
    with pytest.raises(ValueError, match="仍有 1 条未核验记录"):
        build([_row(manual_person_label="pending")])
