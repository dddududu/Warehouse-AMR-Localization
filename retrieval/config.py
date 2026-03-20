from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class CoarseRetrievalConfig:
    sequence_root: str | None = None
    calibration_path: str | None = None
    map_path: str | None = None
    dataset_parent_root: str | None = None
    shared_calibration_path: str | None = None
    shared_map_path: str | None = None
    sequence_entries: list[dict[str, Any]] | None = None
    sequence_names: list[str] | None = None
    val_sequence_names: list[str] | None = None
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
    align_query_to_gt_yaw_train: bool = True
    use_query_rotation_search: bool = True
    query_rotation_search_step_deg: float = 45.0
    num_hard_negative_patches: int = 4
    num_random_negative_patches: int = 4
    hard_negative_min_distance_m: float = 5.0
    hard_negative_max_distance_m: float = 25.0
    train_batch_size: int = 2
    train_num_workers: int = 0
    train_epochs: int = 10
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    lr_decay_gamma: float = 1.0
    init_checkpoint_path: str | None = None
    temperature: float = 0.07
    model_seed: int = 0
    device: str = "cpu"
    use_amp: bool = False
    val_ratio: float = 0.1
    eval_every_epochs: int = 1
    save_best_only: bool = True
    backbone_variant: str = "legacy"
    share_query_patch_encoder: bool = False
    freeze_query_encoder: bool = False
    freeze_patch_encoder: bool = False
    ensemble_checkpoint_paths: list[str] | None = None
    ensemble_model_weights: list[float] | None = None
    ensemble_classifier_score_weights: list[float] | None = None
    train_random_rotation_deg: float = 0.0
    train_random_xy_shift_cells: int = 0
    train_dropout_prob: float = 0.0
    train_count_noise_std: float = 0.0
    train_height_noise_std: float = 0.0
    use_patch_classification_loss: bool = False
    classification_loss_weight: float = 0.0
    classifier_score_weight: float = 0.0

    @property
    def bev_num_cells(self) -> int:
        return int(round(self.bev_size_xy_m / self.bev_resolution))

    @property
    def patch_num_cells(self) -> int:
        return int(round(self.patch_size_m / self.bev_resolution))

    @property
    def patch_stride_cells(self) -> int:
        return int(round(self.patch_stride_m / self.bev_resolution))

    @property
    def num_negative_patches(self) -> int:
        return int(self.num_hard_negative_patches + self.num_random_negative_patches)

    @property
    def query_rotation_search_angles_deg(self) -> list[float]:
        if not self.use_query_rotation_search:
            return [0.0]
        step = float(self.query_rotation_search_step_deg)
        if step <= 0.0:
            raise ValueError("query_rotation_search_step_deg must be positive.")
        angles: list[float] = []
        current = 0.0
        while current < 360.0 - 1e-6:
            angles.append(current)
            current += step
        if 0.0 not in angles:
            angles.insert(0, 0.0)
        return angles

    def resolved_sequence_names(self) -> list[str]:
        if self.sequence_entries:
            return [str(entry["sequence_name"]) for entry in self.resolved_sequence_entries()]
        if self.sequence_names:
            return list(self.sequence_names)
        if self.sequence_root:
            return [Path(self.sequence_root).name]
        if self.dataset_parent_root:
            root = Path(self.dataset_parent_root)
            return sorted([path.name for path in root.iterdir() if path.is_dir()])
        raise ValueError("Either sequence_root or dataset_parent_root must be configured.")

    def resolved_sequence_entries(self) -> list[dict[str, str]]:
        if self.sequence_entries:
            entries: list[dict[str, str]] = []
            shared_calibration_path = str(Path(self.shared_calibration_path)) if self.shared_calibration_path else None
            shared_map_path = str(Path(self.shared_map_path)) if self.shared_map_path else None
            for raw_entry in self.sequence_entries:
                if not isinstance(raw_entry, Mapping):
                    raise ValueError("Each item in sequence_entries must be a mapping.")
                sequence_name = str(raw_entry.get("sequence_name") or "")
                if not sequence_name:
                    if raw_entry.get("sequence_root"):
                        sequence_name = Path(str(raw_entry["sequence_root"])).name
                    else:
                        raise ValueError("sequence_entries items require sequence_name or sequence_root.")
                sequence_root = raw_entry.get("sequence_root")
                if sequence_root is None:
                    dataset_parent_root = raw_entry.get("dataset_parent_root") or self.dataset_parent_root
                    if dataset_parent_root is None:
                        raise ValueError(
                            "sequence_entries items require sequence_root, or dataset_parent_root must be provided."
                        )
                    sequence_root = Path(str(dataset_parent_root)) / sequence_name / sequence_name
                calibration_path = raw_entry.get("calibration_path") or shared_calibration_path
                map_path = raw_entry.get("map_path") or shared_map_path
                if calibration_path is None:
                    calibration_path = str(Path(str(sequence_root)) / "calibrations.txt")
                if map_path is None:
                    map_path = str(Path(str(sequence_root)) / "groundtruth_map.ply")
                entry = {
                    "sequence_name": sequence_name,
                    "sequence_root": str(Path(str(sequence_root))),
                    "calibration_path": str(Path(str(calibration_path))),
                    "map_path": str(Path(str(map_path))),
                }
                if "split" in raw_entry and raw_entry["split"] is not None:
                    entry["split"] = str(raw_entry["split"])
                entries.append(entry)
            return entries
        if self.sequence_root and self.calibration_path and self.map_path:
            sequence_name = Path(self.sequence_root).name
            return [
                {
                    "sequence_name": sequence_name,
                    "sequence_root": str(Path(self.sequence_root)),
                    "calibration_path": str(Path(self.calibration_path)),
                    "map_path": str(Path(self.map_path)),
                }
            ]
        if not self.dataset_parent_root:
            raise ValueError(
                "Multi-sequence mode requires dataset_parent_root, or single-sequence mode requires "
                "sequence_root/calibration_path/map_path."
            )
        entries: list[dict[str, str]] = []
        parent_root = Path(self.dataset_parent_root)
        shared_calibration_path = str(Path(self.shared_calibration_path)) if self.shared_calibration_path else None
        shared_map_path = str(Path(self.shared_map_path)) if self.shared_map_path else None
        fallback_calibration_path = str(Path(self.calibration_path)) if self.calibration_path else None
        fallback_map_path = str(Path(self.map_path)) if self.map_path else None
        for sequence_name in self.resolved_sequence_names():
            sequence_dir = parent_root / sequence_name / sequence_name
            entries.append(
                {
                    "sequence_name": sequence_name,
                    "sequence_root": str(sequence_dir),
                    "calibration_path": shared_calibration_path
                    or fallback_calibration_path
                    or str(sequence_dir / "calibrations.txt"),
                    "map_path": shared_map_path or fallback_map_path or str(sequence_dir / "groundtruth_map.ply"),
                }
            )
        return entries

    def split_sequence_entries(self) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        entries = self.resolved_sequence_entries()
        if len(entries) == 1:
            return entries, []

        explicit_split_entries = [entry for entry in entries if entry.get("split") is not None]
        if explicit_split_entries:
            train_entries = [entry for entry in entries if str(entry.get("split", "train")).lower() != "val"]
            val_entries = [entry for entry in entries if str(entry.get("split", "train")).lower() == "val"]
            if not train_entries:
                raise ValueError("Explicit split configuration requires at least one train entry.")
            if not val_entries:
                raise ValueError("Explicit split configuration requires at least one val entry.")
            return train_entries, val_entries

        val_names = set(self.val_sequence_names or [])
        if not val_names:
            val_names = {entries[-1]["sequence_name"]}
        train_entries = [entry for entry in entries if entry["sequence_name"] not in val_names]
        val_entries = [entry for entry in entries if entry["sequence_name"] in val_names]
        if not train_entries:
            raise ValueError("Validation split consumed all sequences; at least one train sequence is required.")
        if not val_entries:
            raise ValueError("No validation sequences were selected.")
        return train_entries, val_entries

    def resolve_single_sequence_entry(self, sequence_name: str | None = None, prefer_validation: bool = False) -> dict[str, str]:
        train_entries, val_entries = self.split_sequence_entries()
        if sequence_name is not None:
            for entry in train_entries + val_entries:
                if entry["sequence_name"] == sequence_name:
                    return entry
            raise ValueError(f"Unknown sequence_name: {sequence_name}")
        if prefer_validation and val_entries:
            return val_entries[0]
        return (train_entries + val_entries)[0]

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
