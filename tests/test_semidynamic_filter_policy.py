import numpy as np

from localization.config import FineLocalizationConfig
from preprocess import dynamic_point_filter


def test_semidynamic_trigger_keeps_dynamic_only_filter_below_threshold(monkeypatch) -> None:
    points = np.zeros((4, 3), dtype=np.float32)
    calls = []

    def fake_filter(points_sensor, **kwargs):
        labels = tuple(kwargs["dynamic_labels"])
        calls.append(labels)
        mask = np.array([False, True, False, False]) if labels == (5, 7) else np.zeros(4, dtype=bool)
        return points_sensor[~mask], mask

    monkeypatch.setattr(dynamic_point_filter, "filter_dynamic_points_by_semantics", fake_filter)
    filtered, decision = dynamic_point_filter.filter_points_with_semidynamic_trigger(
        points,
        calibration=None,
        segmentation_left_path=None,
        segmentation_right_path=None,
        dynamic_labels=(12, 13),
        semi_dynamic_labels=(5, 7),
        semi_dynamic_ratio_threshold=0.30,
    )

    assert calls == [(5, 7), (12, 13)]
    assert filtered.shape == (4, 3)
    assert decision.semi_dynamic_point_ratio == 0.25
    assert not decision.use_conservative_filter
    assert decision.filtered_labels == (12, 13)


def test_semidynamic_trigger_adds_semidynamic_labels_above_threshold(monkeypatch) -> None:
    points = np.zeros((4, 3), dtype=np.float32)
    calls = []

    def fake_filter(points_sensor, **kwargs):
        labels = tuple(kwargs["dynamic_labels"])
        calls.append(labels)
        mask = np.array([False, True, False, False]) if labels == (5, 7) else np.array([False, True, True, False])
        return points_sensor[~mask], mask

    monkeypatch.setattr(dynamic_point_filter, "filter_dynamic_points_by_semantics", fake_filter)
    filtered, decision = dynamic_point_filter.filter_points_with_semidynamic_trigger(
        points,
        calibration=None,
        segmentation_left_path=None,
        segmentation_right_path=None,
        dynamic_labels=(12, 13),
        semi_dynamic_labels=(5, 7),
        semi_dynamic_ratio_threshold=0.20,
    )

    assert calls == [(5, 7), (12, 13, 5, 7)]
    assert filtered.shape == (2, 3)
    assert decision.use_conservative_filter
    assert decision.filtered_labels == (12, 13, 5, 7)


def test_semidynamic_filter_config_parses_label_lists() -> None:
    config = FineLocalizationConfig.from_mapping(
        {
            "coarse_config_path": "coarse.yaml",
            "semantic_semi_dynamic_labels": [5, 7, 9],
        }
    )
    assert config.semantic_semi_dynamic_labels == (5, 7, 9)
