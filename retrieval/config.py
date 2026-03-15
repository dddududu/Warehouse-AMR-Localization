from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class CoarseRetrievalConfig:
    sequence_root: str
    calibration_path: str
    map_path: str
    crop_x_min: float = -10.0
    crop_x_max: float = 10.0
    crop_y_min: float = -10.0
    crop_y_max: float = 10.0
    crop_z_min: float = 0.13
    crop_z_max: float = 4.73
    bev_resolution: float = 0.1
    bev_size_xy_m: float = 20.0
    patch_size_m: float = 20.0
    patch_stride_m: float = 5.0
    descriptor_dim: int = 256
    topk: int = 8
    cache_dir: str = "./cache/coarse_retrieval"
    num_negative_patches: int = 4
    train_batch_size: int = 2
    train_epochs: int = 1
    learning_rate: float = 1.0e-3
    temperature: float = 0.07
    model_seed: int = 0
    device: str = "cpu"

    @property
    def bev_num_cells(self) -> int:
        return int(round(self.bev_size_xy_m / self.bev_resolution))

    @property
    def patch_num_cells(self) -> int:
        return int(round(self.patch_size_m / self.bev_resolution))

    @property
    def patch_stride_cells(self) -> int:
        return int(round(self.patch_stride_m / self.bev_resolution))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "CoarseRetrievalConfig":
        return cls(**dict(mapping))

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "CoarseRetrievalConfig":
        data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("Coarse retrieval config YAML must parse to a mapping.")
        return cls.from_mapping(data)


def load_coarse_retrieval_config(
    config: CoarseRetrievalConfig | Mapping[str, Any] | str | Path,
) -> CoarseRetrievalConfig:
    if isinstance(config, CoarseRetrievalConfig):
        return config
    if isinstance(config, Mapping):
        return CoarseRetrievalConfig.from_mapping(config)
    return CoarseRetrievalConfig.from_yaml(config)

