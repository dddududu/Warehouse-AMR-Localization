from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from runtime_compat import ensure_runtime_compatibility

ensure_runtime_compatibility()

import torch
from torch.utils.data import Dataset

from analysis.analyze_map import load_map_vertices
from dataset_io.lidar_loader import load_pcd_xyz
from dataset_io.sequence_dataset import WarehouseSequenceDataset
from geometry.yaw_utils import wrap_to_pi, yaw_from_quaternion_xyzw
from localization.deep_config import DeepFineMatcherTrainConfig, load_deep_fine_matcher_train_config
from localization.submap_builder import build_local_submap
from preprocess.bev_builder import BEVConfig, points_to_bev
from preprocess.local_lidar_cropper import LocalCropConfig, crop_local_lidar_points
from preprocess.map_patch_builder import PatchMetadata, build_or_load_patch_cache, choose_gt_patch_id
from retrieval.config import CoarseRetrievalConfig, load_coarse_retrieval_config


@dataclass(frozen=True)
class SequenceFineLocalizationResources:
    sequence_name: str
    sequence_dataset: WarehouseSequenceDataset
    patch_metadata: list[PatchMetadata]
    candidate_patch_bevs: np.ndarray
    candidate_centers_xy: np.ndarray


class DeepFineLocalizationDataset(Dataset):
    def __init__(
        self,
        config: DeepFineMatcherTrainConfig | dict | str | Path,
        sequence_entries: list[dict[str, str]] | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.config = load_deep_fine_matcher_train_config(config)
        self.coarse_cfg = load_coarse_retrieval_config(self.config.coarse_config_path)
        self.sequence_entries = (
            list(sequence_entries)
            if sequence_entries is not None
            else list(self.coarse_cfg.resolved_sequence_entries())
        )
        self.crop_config = LocalCropConfig(
            x_min=self.coarse_cfg.crop_x_min,
            x_max=self.coarse_cfg.crop_x_max,
            y_min=self.coarse_cfg.crop_y_min,
            y_max=self.coarse_cfg.crop_y_max,
            z_min=self.coarse_cfg.crop_z_min,
            z_max=self.coarse_cfg.crop_z_max,
        )
        self.query_bev_config = BEVConfig(
            x_min=self.coarse_cfg.crop_x_min,
            x_max=self.coarse_cfg.crop_x_max,
            y_min=self.coarse_cfg.crop_y_min,
            y_max=self.coarse_cfg.crop_y_max,
            resolution=self.coarse_cfg.bev_resolution,
        )
        self.sequence_resources = [self._build_sequence_resources(entry) for entry in self.sequence_entries]
        self.sample_index: list[tuple[int, int]] = []
        for sequence_idx, resources in enumerate(self.sequence_resources):
            self.sample_index.extend((sequence_idx, frame_idx) for frame_idx in range(len(resources.sequence_dataset)))
        if max_samples is not None and len(self.sample_index) > int(max_samples):
            self.sample_index = self.sample_index[: int(max_samples)]

    def _build_sequence_resources(self, entry: dict[str, str]) -> SequenceFineLocalizationResources:
        sequence_dataset = WarehouseSequenceDataset(
            sequence_root=entry["sequence_root"],
            calibration_path=entry["calibration_path"],
            config={"load_lidar": False},
        )
        map_cache_key = hashlib.md5(str(Path(entry["map_path"]).resolve()).encode("utf-8")).hexdigest()[:12]
        coarse_cache_dir = Path(self.coarse_cfg.cache_dir) / "maps" / map_cache_key
        patch_cache = build_or_load_patch_cache(
            entry["map_path"],
            CoarseRetrievalConfig.from_mapping(
                {
                    **self.coarse_cfg.__dict__,
                    "sequence_root": entry["sequence_root"],
                    "calibration_path": entry["calibration_path"],
                    "map_path": entry["map_path"],
                    "cache_dir": str(coarse_cache_dir),
                }
            ),
        )
        submap_cache_dir = Path(self.config.cache_dir) / "fine_submaps" / map_cache_key
        submap_cache_dir.mkdir(parents=True, exist_ok=True)
        submap_cache_file = submap_cache_dir / (
            f"submaps_s{self.config.local_submap_size_m:.1f}_r{self.config.local_submap_resolution:.2f}.npz"
        )
        if submap_cache_file.is_file():
            cache = np.load(submap_cache_file)
            candidate_patch_bevs = np.asarray(cache["candidate_patch_bevs"], dtype=np.float32)
        else:
            map_xyz, _ = load_map_vertices(
                entry["map_path"],
                cache_dir=submap_cache_dir / "map_vertices",
            )
            candidate_patch_bevs = []
            for patch_meta in patch_cache["metadata"]:
                submap = build_local_submap(
                    map_xyz,
                    center_xy=patch_meta.center_xy,
                    size_m=self.config.local_submap_size_m,
                    resolution=self.config.local_submap_resolution,
                )
                candidate_patch_bevs.append(np.asarray(submap.bev, dtype=np.float32))
            candidate_patch_bevs = np.stack(candidate_patch_bevs, axis=0).astype(np.float32, copy=False)
            np.savez_compressed(submap_cache_file, candidate_patch_bevs=candidate_patch_bevs)
        centers = np.asarray([item.center_xy for item in patch_cache["metadata"]], dtype=np.float32)
        return SequenceFineLocalizationResources(
            sequence_name=entry["sequence_name"],
            sequence_dataset=sequence_dataset,
            patch_metadata=list(patch_cache["metadata"]),
            candidate_patch_bevs=candidate_patch_bevs,
            candidate_centers_xy=centers,
        )

    def __len__(self) -> int:
        return len(self.sample_index)

    def _select_negative_patch_ids(
        self,
        patch_metadata: list[PatchMetadata],
        gt_patch_id: int,
        frame_seed: int,
    ) -> np.ndarray:
        gt_center = np.asarray(patch_metadata[gt_patch_id].center_xy, dtype=np.float32)
        centers = np.asarray([item.center_xy for item in patch_metadata], dtype=np.float32)
        distances = np.linalg.norm(centers - gt_center[None, :], axis=1)
        hard_mask = (
            (np.arange(len(patch_metadata)) != gt_patch_id)
            & (distances >= self.config.hard_negative_min_distance_m)
            & (distances <= self.config.hard_negative_max_distance_m)
        )
        hard_candidates = np.flatnonzero(hard_mask)
        if hard_candidates.size > 0:
            hard_order = np.argsort(distances[hard_candidates])
            hard_ids = hard_candidates[hard_order[: self.config.num_hard_negative_patches]]
        else:
            hard_ids = np.empty((0,), dtype=np.int64)
        used = np.zeros(len(patch_metadata), dtype=bool)
        used[gt_patch_id] = True
        used[hard_ids] = True
        random_candidates = np.flatnonzero(~used)
        if random_candidates.size > 0:
            rng = np.random.default_rng(seed=int(frame_seed))
            random_ids = rng.permutation(random_candidates)[: self.config.num_random_negative_patches]
        else:
            random_ids = np.empty((0,), dtype=np.int64)
        negative_ids = np.concatenate((hard_ids, random_ids), axis=0).astype(np.int64)
        target_count = int(self.config.num_negative_candidates)
        if negative_ids.shape[0] < target_count:
            rng = np.random.default_rng(seed=int(frame_seed) + 13)
            fallback_candidates = np.flatnonzero(np.arange(len(patch_metadata)) != gt_patch_id)
            extra_ids = rng.choice(
                fallback_candidates,
                size=target_count - negative_ids.shape[0],
                replace=fallback_candidates.size < (target_count - negative_ids.shape[0]),
            ).astype(np.int64)
            negative_ids = np.concatenate((negative_ids, extra_ids), axis=0)
        return negative_ids[:target_count].astype(np.int64)

    def _build_query_bev(self, resources: SequenceFineLocalizationResources, frame_idx: int) -> np.ndarray:
        lidar_path = resources.sequence_dataset.frame_index[frame_idx].lidar_path
        points = load_pcd_xyz(lidar_path)
        cropped = crop_local_lidar_points(points, self.crop_config)
        return points_to_bev(cropped, self.query_bev_config).astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        sequence_idx, frame_idx = self.sample_index[index]
        resources = self.sequence_resources[sequence_idx]
        sample = resources.sequence_dataset[frame_idx]
        gt_position = resources.sequence_dataset.ground_truth.positions[frame_idx]
        gt_quaternion = resources.sequence_dataset.ground_truth.quaternions_xyzw[frame_idx]
        gt_yaw = float(wrap_to_pi(yaw_from_quaternion_xyzw(gt_quaternion)))
        gt_patch_id = choose_gt_patch_id(resources.patch_metadata, float(gt_position[0]), float(gt_position[1]))
        negative_patch_ids = self._select_negative_patch_ids(
            resources.patch_metadata,
            gt_patch_id,
            frame_seed=sequence_idx * 1_000_000 + frame_idx,
        )
        positive_center_xy = resources.candidate_centers_xy[gt_patch_id]
        positive_pose_target = np.asarray(
            [
                float(gt_position[0] - positive_center_xy[0]),
                float(gt_position[1] - positive_center_xy[1]),
                gt_yaw,
            ],
            dtype=np.float32,
        )
        return {
            "sequence_name": resources.sequence_name,
            "frame_idx": int(sample["frame_idx"]),
            "query_bev": torch.from_numpy(self._build_query_bev(resources, frame_idx)),
            "positive_candidate_bev": torch.from_numpy(resources.candidate_patch_bevs[gt_patch_id]),
            "negative_candidate_bevs": torch.from_numpy(resources.candidate_patch_bevs[negative_patch_ids]),
            "positive_pose_target": torch.from_numpy(positive_pose_target),
            "gt_patch_id": int(gt_patch_id),
            "negative_patch_ids": torch.from_numpy(negative_patch_ids),
            "positive_candidate_center_xy": torch.tensor(positive_center_xy, dtype=torch.float32),
        }
