from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class DeepFineMatcherTrainConfig:
    coarse_config_path: str
    coarse_checkpoint_path: str | None = None
    cache_dir: str = "./cache/deep_fine_matcher"
    local_submap_size_m: float = 30.0
    local_submap_resolution: float = 0.1
    num_hard_negative_patches: int = 4
    num_random_negative_patches: int = 4
    hard_negative_min_distance_m: float = 5.0
    hard_negative_max_distance_m: float = 30.0
    descriptor_dim: int = 128
    hidden_dim: int = 128
    num_xy_bins: int = 31
    num_yaw_bins: int = 72
    use_query_image: bool = True
    use_stereo_query_image: bool = True
    use_query_depth: bool = True
    use_stereo_geometry: bool = True
    image_resize_hw: tuple[int, int] = (128, 192)
    depth_scale: float = 0.001
    depth_min_m: float = 0.1
    depth_max_m: float = 20.0
    topk_candidates: int = 5
    train_batch_size: int = 8
    train_num_workers: int = 0
    train_epochs: int = 4
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-4
    lr_decay_gamma: float = 1.0
    model_seed: int = 0
    device: str = "cpu"
    use_amp: bool = False
    save_best_only: bool = True
    eval_every_epochs: int = 1
    match_loss_weight: float = 1.0
    pose_loss_weight: float = 1.0
    init_checkpoint_path: str | None = None
    max_train_samples: int | None = None
    max_val_samples: int | None = None

    @property
    def num_negative_candidates(self) -> int:
        return int(self.num_hard_negative_patches + self.num_random_negative_patches)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "DeepFineMatcherTrainConfig":
        payload = dict(mapping)
        resize_hw = payload.get("image_resize_hw")
        if resize_hw is not None:
            payload["image_resize_hw"] = (int(resize_hw[0]), int(resize_hw[1]))
        return cls(**payload)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "DeepFineMatcherTrainConfig":
        data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("Deep fine matcher config YAML must parse to a mapping.")
        return cls.from_mapping(data)


def load_deep_fine_matcher_train_config(
    config: DeepFineMatcherTrainConfig | Mapping[str, Any] | str | Path,
) -> DeepFineMatcherTrainConfig:
    if isinstance(config, DeepFineMatcherTrainConfig):
        return config
    if isinstance(config, Mapping):
        return DeepFineMatcherTrainConfig.from_mapping(config)
    return DeepFineMatcherTrainConfig.from_yaml(config)
