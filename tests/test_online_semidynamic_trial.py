from pathlib import Path

from experiments.semantic_class_prior_20260808.run_online_semidynamic_trial import build_trial_config


def test_trial_modes_only_change_semantic_filter_policy(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("coarse_config_path: coarse.yaml\noutput_json: old.json\n", encoding="utf-8")

    original = build_trial_config(base, "original", tmp_path / "original.json")
    dynamic = build_trial_config(base, "dynamic_only", tmp_path / "dynamic.json")
    adaptive = build_trial_config(base, "adaptive_semidynamic", tmp_path / "adaptive.json")

    assert not original.use_semantic_dynamic_filter
    assert dynamic.use_semantic_dynamic_filter
    assert not dynamic.use_semidynamic_filter_trigger
    assert dynamic.semantic_filter_apply_dynamic_filter_without_trigger
    assert adaptive.use_semantic_dynamic_filter
    assert adaptive.use_semidynamic_filter_trigger
    assert not adaptive.semantic_filter_apply_dynamic_filter_without_trigger
    assert adaptive.use_semantic_dual_geometry_gate
    assert adaptive.use_semantic_shadow_tracker
    assert adaptive.semantic_semi_dynamic_ratio_threshold == 0.10
