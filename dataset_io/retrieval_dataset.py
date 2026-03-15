from __future__ import annotations

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
from preprocess.local_lidar_cropper import LocalCropConfig, crop_local_lidar_points
from preprocess.map_patch_builder import build_or_load_patch_cache, choose_gt_patch_id
from retrieval.config import CoarseRetrievalConfig, load_coarse_retrieval_config


class CoarseRetrievalDataset(Dataset):
    def __init__(self, config: CoarseRetrievalConfig | dict | str | Path) -> None:
        self.config = load_coarse_retrieval_config(config)
        self.sequence_dataset = WarehouseSequenceDataset(
            sequence_root=self.config.sequence_root,
            calibration_path=self.config.calibration_path,
            config={"load_lidar": False},
        )
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
        patch_cache = build_or_load_patch_cache(self.config.map_path, self.config)
        self.patch_tensors = np.asarray(patch_cache["patch_tensors"], dtype=np.float32)
        self.patch_metadata = patch_cache["metadata"]

    def __len__(self) -> int:
        return len(self.sequence_dataset)

    def _build_query_bev(self, frame_idx: int) -> np.ndarray:
        lidar_path = self.sequence_dataset.frame_index[frame_idx].lidar_path
        points = load_pcd_xyz(lidar_path)
        cropped = crop_local_lidar_points(points, self.crop_config)
        return points_to_bev(cropped, self.query_bev_config)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        sample = self.sequence_dataset[index]
        frame_idx = int(sample["frame_idx"])
        query_bev = self._build_query_bev(frame_idx)
        gt_position = self.sequence_dataset.ground_truth.positions[frame_idx]
        gt_quaternion = self.sequence_dataset.ground_truth.quaternions_xyzw[frame_idx]
        gt_patch_id = choose_gt_patch_id(self.patch_metadata, float(gt_position[0]), float(gt_position[1]))

        negative_pool = np.setdiff1d(np.arange(len(self.patch_tensors)), np.array([gt_patch_id]), assume_unique=False)
        num_negatives = min(self.config.num_negative_patches, negative_pool.size)
        negative_ids = negative_pool[:num_negatives]
        negative_patch_bevs = self.patch_tensors[negative_ids]

        return {
            "query_bev": torch.from_numpy(query_bev),
            "positive_patch_bev": torch.from_numpy(self.patch_tensors[gt_patch_id]),
            "negative_patch_bevs": torch.from_numpy(negative_patch_bevs),
            "gt_patch_id": int(gt_patch_id),
            "gt_pose_xyyaw": torch.tensor(
                [
                    float(gt_position[0]),
                    float(gt_position[1]),
                    float(yaw_from_quaternion_xyzw(gt_quaternion)),
                ],
                dtype=torch.float32,
            ),
        }
