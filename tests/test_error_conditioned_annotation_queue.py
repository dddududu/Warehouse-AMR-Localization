# -*- coding: utf-8 -*-
from experiments.target_motion_20260812.build_error_conditioned_annotation_queue import (
    ErrorSegment,
    choose_control,
    choose_segments,
)


def _segment(sequence: str, split: str, frame: int, error: float) -> ErrorSegment:
    return ErrorSegment(sequence, split, frame, frame, frame, error, error, 6, {"sequence": sequence, "split": split})


def test_selection_balances_sequences_within_split() -> None:
    selected = choose_segments(
        [_segment("a", "train", 10, 0.8), _segment("a", "train", 20, 0.7), _segment("b", "train", 30, 0.6)],
        max_per_split=2,
    )
    assert {item.sequence for item in selected} == {"a", "b"}


def test_control_prefers_same_patch_and_excludes_high_error_anchor() -> None:
    segment = _segment("a", "train", 100, 0.5)
    frames = [
        {"frame_idx": 20, "position_error_m": 0.05, "best_patch_id": 3},
        {"frame_idx": 40, "position_error_m": 0.06, "best_patch_id": 6},
        {"frame_idx": 100, "position_error_m": 0.05, "best_patch_id": 6},
    ]
    chosen = choose_control(segment, frames, {100})
    assert chosen is not None
    assert chosen["frame_idx"] == 40
