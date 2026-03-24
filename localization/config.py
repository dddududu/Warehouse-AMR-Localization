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

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FineLocalizationConfig":
        return cls(**dict(mapping))

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
