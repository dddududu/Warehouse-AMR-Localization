import numpy as np

from experiments.semantic_class_prior_20260808.evaluate_structural_update_safety import _accepts


def _current(label: int, support: int = 4) -> np.ndarray:
    value = np.zeros(17, dtype=np.int32)
    value[0] = support
    value[label + 1] = support
    return value


def test_structural_policy_accepts_persistent_rack() -> None:
    accepted = _accepts("structural_persistent", _current(6), {"dominant_label": 6, "stability": 0.9, "evidence": 6}, {4, 6, 8}, {1, 4, 6, 8}, 3, 0.7, 3)
    assert accepted


def test_structural_policy_rejects_goods_and_dynamic_objects() -> None:
    map_info = {"dominant_label": 7, "stability": 0.9, "evidence": 6}
    assert not _accepts("structural_persistent", _current(7), map_info, {4, 6, 8}, {1, 4, 6, 8}, 3, 0.7, 3)
    assert not _accepts("structural_persistent", _current(13), {"dominant_label": 13, "stability": 0.9, "evidence": 6}, {4, 6, 8}, {1, 4, 6, 8}, 3, 0.7, 3)
