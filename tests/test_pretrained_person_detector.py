import numpy as np

from experiments.target_motion_20260803.evaluate_pretrained_person_detector import _box_iou, _match_count


def test_box_iou_and_greedy_matching() -> None:
    predictions = np.asarray([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], dtype=np.float32)
    references = np.asarray([[1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]], dtype=np.float32)
    overlaps = _box_iou(predictions, references)
    assert overlaps.shape == (2, 2)
    assert np.isclose(overlaps[0, 0], 0.64)
    assert _match_count(predictions, references, 0.3) == 2
