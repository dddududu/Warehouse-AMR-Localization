from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class FineLocalizationConfig:
    coarse_config_path: str
    coarse_checkpoint_path: str | None = None
    descriptor_bank_path: str | None = None
    output_json: str | None = None
    topk_candidates: int = 5
    local_submap_size_m: float = 40.0
    local_submap_resolution: float = 0.1
    match_channel: str = "occupancy"
    coarse_yaw_half_range_deg: float = 15.0
    coarse_yaw_step_deg: float = 1.0
    coarse_match_method: str = "ccoeff"
    voxel_size_m: float = 0.3
    icp_max_iterations: int = 20
    icp_max_correspondence_distance_m: float = 1.0
    icp_min_correspondences: int = 24
    retrieval_score_weight: float = 0.2
    bev_score_weight: float = 0.35
    icp_inlier_weight: float = 0.35
    icp_rmse_weight: float = 0.10
    invalid_icp_penalty: float = 1.0
    temporal_weight: float = 0.0
    temporal_position_sigma_m: float = 1.5
    temporal_yaw_sigma_deg: float = 20.0
    use_constant_velocity_prediction: bool = True
    fallback_to_predicted_pose_on_invalid: bool = True
    max_temporal_position_jump_m: float | None = None
    max_temporal_yaw_jump_deg: float | None = None
    use_sequence_smoothing: bool = True
    sequence_smoothing_position_sigma_m: float = 2.0
    sequence_smoothing_yaw_sigma_deg: float = 30.0
    sequence_smoothing_stay_bonus: float = 0.3
    deep_matcher_checkpoint_path: str | None = None
    deep_matcher_score_weight: float = 0.35
    deep_pose_confidence_weight: float = 0.10
    use_deep_pose_init_hypothesis: bool = True
    use_tracker_pose_init_hypothesis: bool = False
    tracker_pose_init_consistency_max_xy_m: float | None = None
    tracker_pose_init_consistency_max_yaw_deg: float | None = None
    tracker_pose_init_min_inlier_ratio: float | None = None
    tracker_pose_init_max_rmse: float | None = None
    deep_pose_init_xy_consistency_weight: float = 0.05
    deep_pose_init_yaw_consistency_weight: float = 0.01
    deep_pose_hypothesis_valid_bonus: float = 0.5
    deep_geometry_gate_min_bev_score: float = 0.10
    deep_geometry_gate_min_inlier_ratio: float = 0.60
    use_online_pose_stabilizer: bool = True
    online_stabilizer_max_position_jump_m: float = 2.5
    online_stabilizer_max_yaw_jump_deg: float = 20.0
    online_stabilizer_min_icp_inlier_ratio: float = 0.55
    online_stabilizer_min_bev_score: float = 0.08
    online_stabilizer_score_margin: float = 0.20
    online_stabilizer_temporal_bonus: float = 0.10
    online_stabilizer_fallback_to_tracker: bool = True
    use_online_patch_hysteresis: bool = True
    online_patch_switch_margin: float = 0.05
    use_persistent_patch_override: bool = False
    persistent_patch_override_pairs: tuple[tuple[int, int], ...] = ()
    persistent_patch_override_topn: int = 4
    persistent_patch_override_min_run_length: int = 6
    persistent_patch_override_min_geometry_score: float = 1.10
    use_top2_confusion_geometry_override: bool = False
    top2_confusion_pairs: tuple[tuple[int, int], ...] = ()
    top2_confusion_final_gap_max: float = 0.30
    top2_confusion_bev_weight: float = 0.80
    top2_confusion_inlier_weight: float = 1.20
    top2_confusion_rmse_weight: float = 0.30
    top2_confusion_geometry_margin: float = 0.0
    candidate_reranker_checkpoint_path: str | None = None
    candidate_reranker_score_weight: float = 0.20
    use_candidate_reranker_top2_override: bool = False
    candidate_reranker_top2_probability_min: float = 0.80
    candidate_reranker_top2_margin_min: float = 0.20
    candidate_reranker_top2_final_gap_max: float = 0.25
    candidate_reranker_top2_pairs: tuple[tuple[int, int], ...] = ()
    confusion_pair_resolver_checkpoint_path: str | None = None
    confusion_pair_resolver_score_weight: float = 0.20
    confusion_pair_resolver_top2_only: bool = True
    confusion_pair_resolver_final_gap_max: float = 0.30
    confusion_pair_resolver_skip_if_top1_bev_score_ge: float | None = None
    confusion_pair_resolver_skip_if_top1_inlier_ge: float | None = None
    confusion_pair_resolver_skip_if_winner_deep_match_lower_by: float | None = None
    confusion_pair_resolver_pair_weight_overrides: tuple[tuple[int, int, float], ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FineLocalizationConfig":
        payload = dict(mapping)
        if "confusion_pair_resolver_pair_weight_overrides" in payload:
            payload["confusion_pair_resolver_pair_weight_overrides"] = tuple(
                (int(item[0]), int(item[1]), float(item[2]))
                for item in payload["confusion_pair_resolver_pair_weight_overrides"]
            )
        if "candidate_reranker_top2_pairs" in payload:
            payload["candidate_reranker_top2_pairs"] = tuple(
                (int(item[0]), int(item[1]))
                for item in payload["candidate_reranker_top2_pairs"]
            )
        if "persistent_patch_override_pairs" in payload:
            payload["persistent_patch_override_pairs"] = tuple(
                (int(item[0]), int(item[1]))
                for item in payload["persistent_patch_override_pairs"]
            )
        return cls(**payload)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "FineLocalizationConfig":
        data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("Fine localization config YAML must parse to a mapping.")
        return cls.from_mapping(data)


def load_fine_localization_config(
    config: FineLocalizationConfig | Mapping[str, Any] | str | Path,
) -> FineLocalizationConfig:
    if isinstance(config, FineLocalizationConfig):
        return config
    if isinstance(config, Mapping):
        return FineLocalizationConfig.from_mapping(config)
    return FineLocalizationConfig.from_yaml(config)
