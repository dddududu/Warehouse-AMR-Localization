import numpy as np
import pytest

from experiments.target_motion_20260804.rank_person_candidates_by_detector_agreement import _agreement


def test_agreement_prefers_overlap_and_confidence() -> None:
    semantic = np.asarray([0.0, 0.0, 100.0, 100.0], dtype=np.float32)
    boxes = np.asarray([[0.0, 0.0, 90.0, 90.0], [200.0, 200.0, 300.0, 300.0]], dtype=np.float32)
    scores = np.asarray([0.8, 0.99], dtype=np.float32)
    iou, score, box = _agreement(semantic, boxes, scores)
    assert iou == pytest.approx(0.81)
    assert score == pytest.approx(0.8)
    assert box == [0.0, 0.0, 90.0, 90.0]


def test_agreement_handles_no_person_predictions() -> None:
    iou, score, box = _agreement(np.zeros(4, dtype=np.float32), np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))
    assert (iou, score, box) == (0.0, 0.0, None)
