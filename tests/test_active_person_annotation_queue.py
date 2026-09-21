# -*- coding: utf-8 -*-
from experiments.target_motion_20260812.build_active_person_annotation_queue import select_balanced, score_row


def _row(sequence: str, split: str, hint: str, frame: int) -> dict[str, str]:
    return {
        "sequence": sequence,
        "split": split,
        "motion_hint": hint,
        "anchor_frame": str(frame),
        "track_observation_count": "5",
        "track_scores_json": "[0.8, 0.8, 1.0, 0.8, 0.8]",
        "track_link_iou_json": "[0.7, 0.7, 0.0, 0.7, 0.7]",
    }


def test_motion_evidence_is_prioritized_without_becoming_a_label() -> None:
    moving = score_row(_row("aisle_ccw_run_1", "train", "potentially_moving", 10))
    static = score_row(_row("aisle_ccw_run_1", "train", "likely_static_relative_to_map", 20))

    assert moving.tier == "A"
    assert moving.priority_score > static.priority_score
    assert moving.reason


def test_balanced_selection_limits_single_sequence() -> None:
    items = [
        score_row(_row("aisle_ccw_run_1", "train", "potentially_moving", index))
        for index in range(10)
    ] + [
        score_row(_row("aisle_cw_run_2", "validation", "ambiguous_motion", 100 + index))
        for index in range(4)
    ]

    selected = select_balanced(items, max_items=6)

    assert len(selected) == 6
    assert sum(item.row["sequence"] == "aisle_cw_run_2" for item in selected) >= 2
