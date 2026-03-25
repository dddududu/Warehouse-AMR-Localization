from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from runtime_compat import ensure_runtime_compatibility

ensure_runtime_compatibility()

import torch
from torch.utils.data import Dataset

from analysis.analyze_map import load_map_vertices
from dataset_io.depth_loader import load_depth_png
from dataset_io.image_loader import load_rgb_image
from dataset_io.lidar_loader import load_pcd_xyz
from dataset_io.sequence_dataset import WarehouseSequenceDataset
from geometry.yaw_utils import wrap_to_pi, yaw_from_quaternion_xyzw
from localization.deep_config import DeepFineMatcherTrainConfig, load_deep_fine_matcher_train_config
from localization.submap_builder import build_local_submap
from preprocess.bev_builder import BEVConfig, points_to_bev
from preprocess.local_lidar_cropper import LocalCropConfig, crop_local_lidar_points
from preprocess.map_patch_builder import PatchMetadata, build_or_load_patch_cache, choose_gt_patch_id
from retrieval.build_patch_database import build_patch_database
from retrieval.config import CoarseRetrievalConfig, load_coarse_retrieval_config
from retrieval.retrieve_topk import _build_retrieval_model, build_patch_search_bank, load_descriptor_bank, score_query_bev_against_bank


@dataclass(frozen=True)
class SequenceFineLocalizationResources:
    sequence_name: str
    sequence_dataset: WarehouseSequenceDataset
    patch_metadata: list[PatchMetadata]
    candidate_patch_bevs: np.ndarray
    candidate_centers_xy: np.ndarray
    stereo_baseline_m: float
    query_fx_px: float
    coarse_candidate_patch_ids: np.ndarray
    coarse_gt_candidate_indices: np.ndarray


def build_stereo_geometry_features(
    depth_raw: np.ndarray,
    fx_px: float,
    baseline_m: float,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
    disparity_normalizer_px: float,
) -> np.ndarray:
    depth_m = depth_raw.astype(np.float32) * float(depth_scale)
    valid = depth_m > 0.0
    depth_clipped = np.clip(depth_m, float(depth_min_m), float(depth_max_m))
    inv_depth = np.zeros_like(depth_clipped, dtype=np.float32)
    inv_depth[valid] = 1.0 / np.maximum(depth_clipped[valid], 1.0e-6)
    disparity = np.zeros_like(depth_clipped, dtype=np.float32)
    disparity[valid] = float(fx_px) * float(baseline_m) / np.maximum(depth_clipped[valid], 1.0e-6)
    max_inv_depth = 1.0 / max(float(depth_min_m), 1.0e-6)
    inv_depth = np.clip(inv_depth / max(max_inv_depth, 1.0e-6), 0.0, 1.0)
    disparity = np.clip(disparity / max(float(disparity_normalizer_px), 1.0), 0.0, 1.0)
    valid_mask = valid.astype(np.float32)
    return np.stack((inv_depth, disparity, valid_mask), axis=0).astype(np.float32, copy=False)


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
        resize_height, resize_width = self.config.image_resize_hw
        width_scale = float(resize_width) / float(sequence_dataset.camera_left.width)
        stereo_baseline_m = float(
            np.linalg.norm(
                sequence_dataset.calibration.T_cam1_os.translation
                - sequence_dataset.calibration.T_cam2_os.translation
            )
        )
        descriptor_bank_path = Path(self.coarse_cfg.cache_dir) / f"descriptor_bank_{entry['sequence_name']}.npz"
        if not descriptor_bank_path.is_file():
            build_patch_database(
                {
                    **self.coarse_cfg.__dict__,
                    "sequence_entries": [dict(entry)],
                },
                checkpoint_path=self.config.coarse_checkpoint_path,
                output_path=descriptor_bank_path,
                sequence_name=entry["sequence_name"],
            )
        descriptor_bank, patch_ids, _, patch_tensors = load_descriptor_bank(
            descriptor_bank_path,
            include_patch_tensors=True,
        )
        coarse_device = torch.device(self.coarse_cfg.device)
        retrieval_model, _ = _build_retrieval_model(
            self.coarse_cfg,
            checkpoint_path=self.config.coarse_checkpoint_path,
            num_patch_classes=int(patch_tensors.shape[0]),
            device=coarse_device,
        )
        patch_search_bank, local_feature_bank = build_patch_search_bank(
            retrieval_model,
            patch_tensors,
            device=coarse_device,
        )
        candidate_cache_file = submap_cache_dir / (
            f"coarse_topk_{entry['sequence_name']}_k{int(self.config.topk_candidates)}_"
            f"{Path(self.config.coarse_checkpoint_path).stem if self.config.coarse_checkpoint_path else 'no_ckpt'}.npz"
        )
        if candidate_cache_file.is_file():
            coarse_cache = np.load(candidate_cache_file)
            coarse_candidate_patch_ids = np.asarray(coarse_cache["candidate_patch_ids"], dtype=np.int64)
            coarse_gt_candidate_indices = np.asarray(coarse_cache["gt_candidate_indices"], dtype=np.int64)
        else:
            coarse_candidate_patch_ids, coarse_gt_candidate_indices = self._build_coarse_candidate_cache(
                sequence_dataset=sequence_dataset,
                patch_metadata=list(patch_cache["metadata"]),
                patch_ids=np.asarray(patch_ids, dtype=np.int64),
                retrieval_model=retrieval_model,
                patch_search_bank=patch_search_bank,
                local_feature_bank=local_feature_bank,
                device=coarse_device,
            )
            np.savez_compressed(
                candidate_cache_file,
                candidate_patch_ids=coarse_candidate_patch_ids,
                gt_candidate_indices=coarse_gt_candidate_indices,
            )
        return SequenceFineLocalizationResources(
            sequence_name=entry["sequence_name"],
            sequence_dataset=sequence_dataset,
            patch_metadata=list(patch_cache["metadata"]),
            candidate_patch_bevs=candidate_patch_bevs,
            candidate_centers_xy=centers,
            stereo_baseline_m=stereo_baseline_m,
            query_fx_px=float(sequence_dataset.camera_left.fx) * width_scale,
            coarse_candidate_patch_ids=coarse_candidate_patch_ids,
            coarse_gt_candidate_indices=coarse_gt_candidate_indices,
        )

    def _build_coarse_candidate_cache(
        self,
        sequence_dataset: WarehouseSequenceDataset,
        patch_metadata: list[PatchMetadata],
        patch_ids: np.ndarray,
        retrieval_model: Any,
        patch_search_bank: np.ndarray,
        local_feature_bank: np.ndarray | None,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidate_patch_ids: list[np.ndarray] = []
        gt_candidate_indices: list[int] = []
        topk = int(self.config.topk_candidates)
        for frame_idx in range(len(sequence_dataset)):
            query_bev = self._build_query_bev_from_lidar_path(sequence_dataset.frame_index[frame_idx].lidar_path)
            scores, _ = score_query_bev_against_bank(
                query_bev=query_bev,
                encoder=retrieval_model.query_encoder,
                descriptor_bank=patch_search_bank,
                rotation_angles_deg=self.coarse_cfg.query_rotation_search_angles_deg,
                device=device,
                classifier=retrieval_model.query_classifier,
                classifier_score_weight=self.coarse_cfg.classifier_score_weight,
                local_matcher=retrieval_model.local_matcher,
                local_feature_bank=local_feature_bank,
                local_feature_level=retrieval_model.local_matcher_feature_level,
                local_matcher_score_weight=self.coarse_cfg.local_matcher_score_weight,
                local_matcher_rerank_topk=self.coarse_cfg.local_matcher_rerank_topk,
            )
            ranking = np.argsort(scores)[::-1][:topk]
            frame_patch_ids = patch_ids[ranking].astype(np.int64, copy=True)
            gt_position = sequence_dataset.ground_truth.positions[frame_idx]
            gt_patch_id = choose_gt_patch_id(patch_metadata, float(gt_position[0]), float(gt_position[1]))
            gt_match = np.flatnonzero(frame_patch_ids == int(gt_patch_id))
            if gt_match.size == 0:
                frame_patch_ids[-1] = int(gt_patch_id)
                gt_candidate_index = int(topk - 1)
            else:
                gt_candidate_index = int(gt_match[0])
            candidate_patch_ids.append(frame_patch_ids)
            gt_candidate_indices.append(gt_candidate_index)
        return (
            np.stack(candidate_patch_ids, axis=0).astype(np.int64, copy=False),
            np.asarray(gt_candidate_indices, dtype=np.int64),
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
        return self._build_query_bev_from_lidar_path(lidar_path)

    def _build_query_bev_from_lidar_path(self, lidar_path: str | Path) -> np.ndarray:
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
        candidate_patch_ids = resources.coarse_candidate_patch_ids[frame_idx].astype(np.int64, copy=True)
        gt_candidate_index = int(resources.coarse_gt_candidate_indices[frame_idx])
        rng = np.random.default_rng(seed=sequence_idx * 1_000_000 + frame_idx + 97)
        permutation = rng.permutation(candidate_patch_ids.shape[0]).astype(np.int64)
        candidate_patch_ids = candidate_patch_ids[permutation]
        gt_candidate_index = int(np.flatnonzero(permutation == gt_candidate_index)[0])
        candidate_centers_xy = resources.candidate_centers_xy[candidate_patch_ids]
        candidate_pose_targets = np.stack(
            [
                np.asarray(
                    [
                        float(gt_position[0] - center_xy[0]),
                        float(gt_position[1] - center_xy[1]),
                        gt_yaw,
                    ],
                    dtype=np.float32,
                )
                for center_xy in candidate_centers_xy
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        payload: dict[str, torch.Tensor | int | str] = {
            "sequence_name": resources.sequence_name,
            "frame_idx": int(sample["frame_idx"]),
            "query_bev": torch.from_numpy(self._build_query_bev(resources, frame_idx)),
            "candidate_bevs": torch.from_numpy(resources.candidate_patch_bevs[candidate_patch_ids]),
            "candidate_pose_targets": torch.from_numpy(candidate_pose_targets),
            "gt_patch_id": int(gt_patch_id),
            "gt_candidate_index": int(gt_candidate_index),
            "candidate_patch_ids": torch.from_numpy(candidate_patch_ids),
            "candidate_centers_xy": torch.from_numpy(candidate_centers_xy.astype(np.float32)),
        }
        if bool(self.config.use_query_image):
            image_left, _ = load_rgb_image(
                sample["image_left_path"],
                camera_model=resources.sequence_dataset.camera_left,
                use_undistort=False,
                resize_hw=self.config.image_resize_hw,
            )
            image_left = np.asarray(image_left, dtype=np.float32) / 255.0
            image_left = np.transpose(image_left, (2, 0, 1)).astype(np.float32, copy=False)
            payload["query_image"] = torch.from_numpy(image_left)
            if bool(self.config.use_stereo_query_image):
                image_right, _ = load_rgb_image(
                    sample["image_right_path"],
                    camera_model=resources.sequence_dataset.camera_right,
                    use_undistort=False,
                    resize_hw=self.config.image_resize_hw,
                )
                image_right = np.asarray(image_right, dtype=np.float32) / 255.0
                image_right = np.transpose(image_right, (2, 0, 1)).astype(np.float32, copy=False)
                payload["query_image_right"] = torch.from_numpy(image_right)
        if bool(self.config.use_query_depth):
            depth_left = load_depth_png(
                sample["depth_left_path"],
                depth_scale=None,
                resize_hw=self.config.image_resize_hw,
            )
            geometry_features = build_stereo_geometry_features(
                depth_raw=depth_left,
                fx_px=resources.query_fx_px,
                baseline_m=resources.stereo_baseline_m,
                depth_scale=float(self.config.depth_scale),
                depth_min_m=float(self.config.depth_min_m),
                depth_max_m=float(self.config.depth_max_m),
                disparity_normalizer_px=float(self.config.image_resize_hw[1]),
            )
            payload["query_depth_features"] = torch.from_numpy(geometry_features)
        return payload
