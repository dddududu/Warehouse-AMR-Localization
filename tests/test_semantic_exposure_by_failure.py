# -*- coding: utf-8 -*-
from experiments.target_motion_20260829.analyze_semantic_exposure_by_failure import _group_rows, _statistics


def test_groups_keep_normal_and_candidate_failure_rows_separate() -> None:
    rows = [
        {"selected_position_error_m": "0.05", "failure_category": "normal", "semi_dynamic_point_ratio": "0.02"},
        {"selected_position_error_m": "0.25", "failure_category": "topk_selection_failure", "semi_dynamic_point_ratio": "0.04"},
    ]

    normal = _group_rows(rows, "normal")
    selection = _group_rows(rows, "topk_selection_failure")

    assert len(normal) == 1
    assert len(selection) == 1
    assert _statistics(selection)["mean_ratio"] == 0.04
