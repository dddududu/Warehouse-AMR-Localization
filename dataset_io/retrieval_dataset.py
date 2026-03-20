from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from runtime_compat import ensure_runtime_compatibility

ensure_runtime_compatibility()

import torch
from torch.utils.data import Dataset

from dataset_io.lidar_loader import load_pcd_xyz
from dataset_io.sequence_dataset import WarehouseSequenceDataset
from geometry.yaw_utils import yaw_from_quaternion_xyzw
from preprocess.bev_builder import BEVConfig, points_to_bev
from preprocess.bev_transforms import rotate_bev_tensor, rotate_points_xy, translate_bev_tensor
from preprocess.local_lidar_cropper import LocalCropConfig, crop_local_lidar_points
from preprocess.map_patch_builder import PatchMetadata, build_or_load_patch_cache, choose_gt_patch_id
from retrieval.config import CoarseRetrievalConfig, load_coarse_retrieval_config


@dataclass(frozen=True)
class SequenceRetrievalResources:
    sequence_name: str
    sequence_dataset: WarehouseSequenceDataset
    patch_tensors: np.ndarray
    patch_metadata: list[PatchMetadata]


class CoarseRetrievalDataset(Dataset):
    def __init__(
        self,
        config: CoarseRetrievalConfig | dict | str | Path,
        sequence_entries: list[dict[str, str]] | None = None,
        align_query_to_gt_yaw: bool | None = None,
    ) -> None:
        self.config = load_coarse_retrieval_config(config)
        self.sequence_entries = list(sequence_entries or self.config.resolved_sequence_entries())
        self.crop_config = LocalCropConfig(
            x_min=self.config.crop_x_min,
            x_max=self.config.crop_x_max,
            y_min=self.config.crop_y_min,
            y_max=self.config.crop_y_max,
            z_min=self.config.crop_z_min,
            z_max=self.config.crop_z_max,
        )
        self.query_bev_config = BEVConfig(
            x_min=self.config.crop_x_min,
            x_max=self.config.crop_x_max,
            y_min=self.config.crop_y_min,
            y_max=self.config.crop_y_max,
            resolution=self.config.bev_resolution,
        )
        self.align_query_to_gt_yaw = (
            self.config.align_query_to_gt_yaw_train if align_query_to_gt_yaw is None else bool(align_query_to_gt_yaw)
        )
        self.enable_query_augmentation = self.align_query_to_gt_yaw and any(
            (
                float(self.config.train_random_rotation_deg) > 0.0,
                int(self.config.train_random_xy_shift_cells) > 0,
                float(self.config.train_dropout_prob) > 0.0,
                float(self.config.train_count_noise_std) > 0.0,
                float(self.config.train_height_noise_std) > 0.0,
            )
        )
        self.sequence_resources = [self._build_sequence_resources(entry) for entry in self.sequence_entries]

        self.sample_index: list[tuple[int, int]] = []
        for sequence_idx, resources in enumerate(self.sequence_resources):
            self.sample_index.extend((sequence_idx, frame_idx) for frame_idx in range(len(resources.sequence_dataset)))

    def _build_sequence_resources(self, entry: dict[str, str]) -> SequenceRetrievalResources:
        sequence_dataset = WarehouseSequenceDataset(
            sequence_root=entry["sequence_root"],
            calibration_path=entry["calibration_path"],
            config={"load_lidar": False},
        )
        map_cache_key = hashlib.md5(str(Path(entry["map_path"]).resolve()).encode("utf-8")).hexdigest()[:12]
        sequence_cache_dir = Path(self.config.cache_dir) / "maps" / map_cache_key
        patch_cache = build_or_load_patch_cache(
            entry["map_path"],
            CoarseRetrievalConfig.from_mapping(
                {
                    **self.config.__dict__,
                    "sequence_root": entry["sequence_root"],
                    "calibration_path": entry["calibration_path"],
                    "map_path": entry["map_path"],
                    "cache_dir": str(sequence_cache_dir),
                }
            ),
        )
        return SequenceRetrievalResources(
            sequence_name=entry["sequence_name"],
            sequence_dataset=sequence_dataset,
            patch_tensors=np.asarray(patch_cache["patch_tensors"], dtype=np.float32),
            patch_metadata=list(patch_cache["metadata"]),
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
        return np.concatenate((hard_ids, random_ids), axis=0).astype(np.int64)

    def _augment_query_bev(self, bev_tensor: np.ndarray, frame_seed: int) -> np.ndarray:
        augmented = np.asarray(bev_tensor, dtype=np.float32).copy()
        rng = np.random.default_rng(seed=int(frame_seed))

        max_rotation_deg = float(self.config.train_random_rotation_deg)
        if max_rotation_deg > 0.0:
            angle_deg = float(rng.uniform(-max_rotation_deg, max_rotation_deg))
            augmented = rotate_bev_tensor(augmented, angle_deg)

        max_shift_cells = int(self.config.train_random_xy_shift_cells)
        if max_shift_cells > 0:
            shift_x = int(rng.integers(-max_shift_cells, max_shift_cells + 1))
            shift_y = int(rng.integers(-max_shift_cells, max_shift_cells + 1))
            augmented = translate_bev_tensor(augmented, shift_x_cells=shift_x, shift_y_cells=shift_y)

        dropout_prob = float(self.config.train_dropout_prob)
        if dropout_prob > 0.0:
            keep_mask = (rng.random(size=augmented[3].shape) >= dropout_prob).astype(np.float32)
            augmented *= keep_mask[None, ...]

        count_noise_std = float(self.config.train_count_noise_std)
        if count_noise_std > 0.0:
            count_scale = rng.normal(loc=1.0, scale=count_noise_std, size=augmented[0].shape).astype(np.float32)
            augmented[0] = np.clip(augmented[0] * count_scale, a_min=0.0, a_max=None)

        height_noise_std = float(self.config.train_height_noise_std)
        if height_noise_std > 0.0:
            height_noise = rng.normal(loc=0.0, scale=height_noise_std, size=augmented[1:3].shape).astype(np.float32)
            augmented[1:3] = augmented[1:3] + height_noise * augmented[3:4]

        return augmented.astype(np.float32, copy=False)

    def _build_query_bev(self, resources: SequenceRetrievalResources, frame_idx: int) -> np.ndarray:
        lidar_path = resources.sequence_dataset.frame_index[frame_idx].lidar_path
        points = load_pcd_xyz(lidar_path)
        cropped = crop_local_lidar_points(points, self.crop_config)
        if self.align_query_to_gt_yaw:
            yaw = yaw_from_quaternion_xyzw(resources.sequence_dataset.ground_truth.quaternions_xyzw[frame_idx])
            cropped = rotate_points_xy(cropped, yaw)
        query_bev = points_to_bev(cropped, self.query_bev_config)
        if self.enable_query_augmentation:
            frame_seed = int(frame_idx)
            query_bev = self._augment_query_bev(query_bev, frame_seed=frame_seed)
        return query_bev

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        sequence_idx, frame_idx = self.sample_index[index]
        resources = self.sequence_resources[sequence_idx]
        sample = resources.sequence_dataset[frame_idx]
        query_bev = self._build_query_bev(resources, frame_idx)
        gt_position = resources.sequence_dataset.ground_truth.positions[frame_idx]
        gt_quaternion = resources.sequence_dataset.ground_truth.quaternions_xyzw[frame_idx]
        gt_patch_id = choose_gt_patch_id(resources.patch_metadata, float(gt_position[0]), float(gt_position[1]))

        frame_seed = sequence_idx * 1_000_000 + frame_idx
        negative_ids = self._select_negative_patch_ids(resources.patch_metadata, gt_patch_id, frame_seed=frame_seed)
        negative_patch_bevs = resources.patch_tensors[negative_ids]

        return {
            "sequence_name": resources.sequence_name,
            "query_bev": torch.from_numpy(query_bev),
            "positive_patch_bev": torch.from_numpy(resources.patch_tensors[gt_patch_id]),
            "negative_patch_bevs": torch.from_numpy(negative_patch_bevs),
            "gt_patch_id": int(gt_patch_id),
            "negative_patch_ids": torch.from_numpy(negative_ids.astype(np.int64)),
            "gt_pose_xyyaw": torch.tensor(
                [
                    float(gt_position[0]),
                    float(gt_position[1]),
                    float(yaw_from_quaternion_xyzw(gt_quaternion)),
                ],
                dtype=torch.float32,
            ),
            "frame_idx": int(sample["frame_idx"]),
        }
