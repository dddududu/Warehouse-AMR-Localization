# -*- coding: utf-8 -*-
from experiments.target_motion_20260812.audit_candidate_error_mechanisms import category_for_error


def test_error_categories_distinguish_candidate_selection_from_absence() -> None:
    assert category_for_error(0.19, 0.05) == "normal"
    assert category_for_error(0.35, 0.15) == "topk_selection_failure"
    assert category_for_error(0.35, 0.30) == "topk_partial_recovery_only"
    assert category_for_error(0.35, 0.55) == "topk_candidate_absence"
