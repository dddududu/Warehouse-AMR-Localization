import numpy as np
import pytest

from experiments.target_motion_20260804.build_temporal_person_review_packets import _link_box, _motion_hint


def test_link_box_requires_positive_overlap() -> None:
    previous = np.asarray([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    boxes = np.asarray([[1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]], dtype=np.float32)
    scores = np.asarray([0.8, 0.99], dtype=np.float32)
    box, score, overlap = _link_box(previous, boxes, scores)
    assert box is not None
    assert score == pytest.approx(0.8)
    assert overlap > 0.6


def test_motion_hint_is_only_a_review_signal() -> None:
    stable = [np.asarray([0.0, 0.0, 0.0]), np.asarray([0.04, 0.0, 0.0]), np.asarray([0.08, 0.0, 0.0])]
    moving = [np.asarray([0.0, 0.0, 0.0]), np.asarray([0.3, 0.0, 0.0]), np.asarray([0.7, 0.0, 0.0])]
    assert _motion_hint(stable)[0] == "likely_static_relative_to_map"
    assert _motion_hint(moving)[0] == "potentially_moving"
    assert _motion_hint([None, np.zeros(3), None])[0] == "insufficient_3d_evidence"
