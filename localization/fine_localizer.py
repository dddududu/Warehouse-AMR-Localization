from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from runtime_compat import ensure_runtime_compatibility

ensure_runtime_compatibility()

import torch

from analysis.analyze_map import load_map_vertices
from dataset_io.lidar_loader import load_pcd_xyz
from dataset_io.sequence_dataset import WarehouseSequenceDataset
from geometry.se3 import transform_points
from geometry.yaw_utils import wrap_to_pi, yaw_from_pose_matrix
from localization.bev_matcher import match_query_points_to_submap_bev
from localization.config import load_fine_localization_config
from localization.icp_refiner import pose_from_xy_yaw_z, refine_pose_with_icp
from localization.submap_builder import build_local_submap
from preprocess.bev_builder import BEVConfig
from preprocess.dynamic_point_filter import filter_dynamic_points_by_semantics, semantic_label_ratio
from preprocess.local_lidar_cropper import LocalCropConfig, crop_local_lidar_points
from preprocess.map_patch_builder import PatchMetadata, build_or_load_patch_cache
from retrieval.build_patch_database import build_patch_database
from retrieval.config import load_coarse_retrieval_config
from retrieval.retrieve_topk import (
    _build_retrieval_model,
    build_patch_search_bank,
    load_descriptor_bank,
    score_query_bev_against_bank,
)


@dataclass(frozen=True)
class CandidateLocalizationResult:
    patch_id: int
    coarse_score: float
    coarse_query_rotation_deg: float
    initial_world_xy: tuple[float, float]
    initial_yaw_deg: float
    bev_score: float
    icp_inlier_ratio: float
    icp_rmse: float
    icp_valid: bool
    temporal_position_jump_m: float | None
    temporal_yaw_jump_deg: float | None
    temporal_gate_valid: bool
    temporal_score: float
    final_score: float
    final_pose_4x4: np.ndarray


class FineLocalizer:
    def __init__(self, config) -> None:
        self.config = load_fine_localization_config(config)
        self.coarse_cfg = load_coarse_retrieval_config(self.config.coarse_config_path)
        self.entry = self.coarse_cfg.resolve_single_sequence_entry(prefer_validation=True)
        self.sequence_dataset = WarehouseSequenceDataset(
            sequence_root=self.entry["sequence_root"],
            calibration_path=self.entry["calibration_path"],
            config={"load_lidar": False},
        )
        self.device = torch.device(self.coarse_cfg.device)
        descriptor_bank_path = self.config.descriptor_bank_path
        if descriptor_bank_path is None:
            descriptor_bank_path = str(Path(self.coarse_cfg.cache_dir) / f"descriptor_bank_{self.entry['sequence_name']}.npz")
        descriptor_bank_file = Path(descriptor_bank_path)
        if not descriptor_bank_file.is_file():
            build_patch_database(
                self.config.coarse_config_path,
                checkpoint_path=self.config.coarse_checkpoint_path,
                output_path=descriptor_bank_file,
            )
        self.descriptor_bank_path = str(descriptor_bank_path)
        self.descriptor_bank, self.patch_ids, metadata_raw, self.patch_tensors = load_descriptor_bank(
            self.descriptor_bank_path,
            include_patch_tensors=True,
        )
        self.patch_metadata = [PatchMetadata(**item) for item in metadata_raw]
        retrieval_model, _ = _build_retrieval_model(
            self.coarse_cfg,
            checkpoint_path=self.config.coarse_checkpoint_path,
            num_patch_classes=int(self.patch_tensors.shape[0]),
            device=self.device,
        )
        self.retrieval_model = retrieval_model
        self.patch_search_bank, self.local_feature_bank = build_patch_search_bank(
            self.retrieval_model,
            self.patch_tensors,
            device=self.device,
        )
        self.map_xyz, _ = load_map_vertices(
            self.entry["map_path"],
            cache_dir=Path(self.coarse_cfg.cache_dir) / "map_vertices",
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
        self._semantic_occlusion_ratio_cache: dict[int, float] = {}

    def _semantic_occlusion_ratio(self, frame_idx: int) -> float:
        cached = self._semantic_occlusion_ratio_cache.get(int(frame_idx))
        if cached is not None:
            return cached
        record = self.sequence_dataset.frame_index[int(frame_idx)]
        ratio = semantic_label_ratio(
            record.segmentation_greyscale_left_path,
            record.segmentation_greyscale_right_path,
            self.config.semantic_occlusion_labels,
        )
        self._semantic_occlusion_ratio_cache[int(frame_idx)] = float(ratio)
        return float(ratio)

    def _build_query_points(self, frame_idx: int) -> np.ndarray:
        record = self.sequence_dataset.frame_index[frame_idx]
        lidar_path = record.lidar_path
        points = load_pcd_xyz(lidar_path)
        points = crop_local_lidar_points(points, self.crop_config)
        if self.config.use_semantic_dynamic_filter:
            points, _ = filter_dynamic_points_by_semantics(
                points,
                calibration=self.sequence_dataset.calibration,
                segmentation_left_path=record.segmentation_greyscale_left_path,
                segmentation_right_path=record.segmentation_greyscale_right_path,
                dynamic_labels=self.config.semantic_dynamic_labels,
                dilation_px=self.config.semantic_dynamic_mask_dilation_px,
            )
        return points

    def _retrieve_topk_candidates(self, frame_idx: int) -> list[dict[str, Any]]:
        query_points = self._build_query_points(frame_idx)
        from preprocess.bev_builder import points_to_bev

        query_bev = points_to_bev(query_points, self.query_bev_config)
        scores, best_angles = score_query_bev_against_bank(
            query_bev=query_bev,
            encoder=self.retrieval_model.query_encoder,
            descriptor_bank=self.patch_search_bank,
            rotation_angles_deg=self.coarse_cfg.query_rotation_search_angles_deg,
            device=self.device,
            classifier=self.retrieval_model.query_classifier,
            classifier_score_weight=self.coarse_cfg.classifier_score_weight,
            local_matcher=self.retrieval_model.local_matcher,
            local_feature_bank=self.local_feature_bank,
            local_feature_level=self.retrieval_model.local_matcher_feature_level,
            local_matcher_score_weight=self.coarse_cfg.local_matcher_score_weight,
            local_matcher_rerank_topk=self.coarse_cfg.local_matcher_rerank_topk,
        )
        ranking = np.argsort(scores)[::-1][: int(self.config.topk_candidates)]
        return [
            {
                "patch_id": int(self.patch_ids[idx]),
                "score": float(scores[idx]),
                "query_rotation_deg": float(best_angles[idx]),
                "metadata": self.patch_metadata[int(self.patch_ids[idx])],
            }
            for idx in ranking
        ]

    def _predict_motion_prior_4x4(self, accepted_poses: list[np.ndarray]) -> np.ndarray | None:
        if not accepted_poses:
            return None
        if len(accepted_poses) < 2 or not self.config.use_constant_velocity_prediction:
            return np.asarray(accepted_poses[-1], dtype=np.float64).copy()
        prev_pose = np.asarray(accepted_poses[-1], dtype=np.float64)
        prev_prev_pose = np.asarray(accepted_poses[-2], dtype=np.float64)
        prev_xy = prev_pose[:2, 3]
        prev_prev_xy = prev_prev_pose[:2, 3]
        predicted_xy = prev_xy + (prev_xy - prev_prev_xy)
        prev_yaw = yaw_from_pose_matrix(prev_pose)
        prev_prev_yaw = yaw_from_pose_matrix(prev_prev_pose)
        predicted_yaw = float(prev_yaw + wrap_to_pi(prev_yaw - prev_prev_yaw))
        predicted_pose = prev_pose.copy()
        predicted_pose[:2, 3] = predicted_xy
        predicted_pose[:3, :3] = pose_from_xy_yaw_z(0.0, 0.0, predicted_yaw)[:3, :3]
        return predicted_pose

    def _predict_pose_4x4(
        self,
        accepted_poses: list[np.ndarray],
        accepted_frame_results: list[Mapping[str, Any]] | None = None,
    ) -> np.ndarray | None:
        if not accepted_poses:
            return None
        disable_hysteresis_streak = self.config.tracker_pose_init_disable_hysteresis_streak
        disable_hysteresis_min_deep_prob = self.config.tracker_pose_init_disable_hysteresis_min_deep_prob
        if (
            disable_hysteresis_streak is not None
            and disable_hysteresis_streak > 0
            and accepted_frame_results is not None
        ):
            hysteresis_run_length = 0
            challenger_patch_id: int | None = None
            retained_patch_id: int | None = None
            for item in reversed(accepted_frame_results):
                if str(item.get("selection_mode")) != "online_patch_hysteresis":
                    break
                state = item.get("online_patch_hysteresis_state")
                if not isinstance(state, Mapping):
                    break
                if str(state.get("previous_selected_init_source") or "") != "tracker_init":
                    break
                proposed_deep = float(state.get("proposed_deep_match_probability", 0.0))
                if (
                    disable_hysteresis_min_deep_prob is not None
                    and proposed_deep < float(disable_hysteresis_min_deep_prob)
                ):
                    break
                current_challenger = int(state.get("proposed_patch_id", -1))
                current_retained = int(state.get("previous_patch_id", -1))
                if challenger_patch_id is None:
                    challenger_patch_id = current_challenger
                    retained_patch_id = current_retained
                elif challenger_patch_id != current_challenger or retained_patch_id != current_retained:
                    break
                hysteresis_run_length += 1
            if hysteresis_run_length >= int(disable_hysteresis_streak):
                return None
        disable_after_override_frames = self.config.tracker_pose_init_disable_after_override_frames
        disable_after_override_min_deep_prob = self.config.tracker_pose_init_disable_after_override_min_deep_prob
        if (
            disable_after_override_frames is not None
            and disable_after_override_frames > 0
            and accepted_frame_results is not None
        ):
            for item in reversed(accepted_frame_results[-int(disable_after_override_frames):]):
                override = item.get("online_patch_hysteresis_override")
                if not isinstance(override, Mapping):
                    continue
                if str(override.get("previous_selected_init_source") or "") != "tracker_init":
                    continue
                proposed_deep = float(override.get("proposed_deep_match_probability", 0.0))
                if (
                    disable_after_override_min_deep_prob is not None
                    and proposed_deep < float(disable_after_override_min_deep_prob)
                ):
                    continue
                return None
        stable_min_frames = self.config.tracker_pose_init_patch_stable_min_frames
        if (
            stable_min_frames is not None
            and stable_min_frames > 1
            and accepted_frame_results is not None
        ):
            if len(accepted_frame_results) < int(stable_min_frames):
                return None
            recent_patch_ids = [
                int(item.get("best_patch_id", -1))
                for item in accepted_frame_results[-int(stable_min_frames):]
            ]
            if any(patch_id != recent_patch_ids[-1] for patch_id in recent_patch_ids[:-1]):
                return None
        return self._predict_motion_prior_4x4(accepted_poses)

    def _temporal_score(self, pose_4x4: np.ndarray, predicted_pose_4x4: np.ndarray | None) -> float:
        if predicted_pose_4x4 is None or float(self.config.temporal_weight) <= 0.0:
            return 0.0
        predicted_xy = np.asarray(predicted_pose_4x4[:2, 3], dtype=np.float64)
        current_xy = np.asarray(pose_4x4[:2, 3], dtype=np.float64)
        xy_error = float(np.linalg.norm(current_xy - predicted_xy))
        current_yaw = yaw_from_pose_matrix(pose_4x4)
        predicted_yaw = yaw_from_pose_matrix(predicted_pose_4x4)
        yaw_error_deg = abs(math.degrees(float(wrap_to_pi(current_yaw - predicted_yaw))))
        sigma_xy = max(float(self.config.temporal_position_sigma_m), 1.0e-6)
        sigma_yaw = max(float(self.config.temporal_yaw_sigma_deg), 1.0e-6)
        return float(
            math.exp(
                -0.5 * (xy_error / sigma_xy) ** 2
                -0.5 * (yaw_error_deg / sigma_yaw) ** 2
            )
        )

    def _temporal_jump_metrics(
        self,
        pose_4x4: np.ndarray,
        predicted_pose_4x4: np.ndarray | None,
    ) -> tuple[float | None, float | None]:
        if predicted_pose_4x4 is None:
            return None, None
        predicted_xy = np.asarray(predicted_pose_4x4[:2, 3], dtype=np.float64)
        current_xy = np.asarray(pose_4x4[:2, 3], dtype=np.float64)
        xy_jump_m = float(np.linalg.norm(current_xy - predicted_xy))
        current_yaw = yaw_from_pose_matrix(pose_4x4)
        predicted_yaw = yaw_from_pose_matrix(predicted_pose_4x4)
        yaw_jump_deg = abs(math.degrees(float(wrap_to_pi(current_yaw - predicted_yaw))))
        return xy_jump_m, yaw_jump_deg

    def _is_temporal_gate_valid(
        self,
        xy_jump_m: float | None,
        yaw_jump_deg: float | None,
    ) -> bool:
        max_xy_jump = self.config.max_temporal_position_jump_m
        max_yaw_jump = self.config.max_temporal_yaw_jump_deg
        if xy_jump_m is not None and max_xy_jump is not None and xy_jump_m > float(max_xy_jump):
            return False
        if yaw_jump_deg is not None and max_yaw_jump is not None and yaw_jump_deg > float(max_yaw_jump):
            return False
        return True

    def localize_frame(self, frame_idx: int, predicted_pose_4x4: np.ndarray | None = None) -> dict[str, Any]:
        query_points = self._build_query_points(frame_idx)
        candidates = self._retrieve_topk_candidates(frame_idx)
        candidate_results: list[CandidateLocalizationResult] = []
        for candidate in candidates:
            patch_meta = candidate["metadata"]
            submap = build_local_submap(
                self.map_xyz,
                center_xy=patch_meta.center_xy,
                size_m=self.config.local_submap_size_m,
                resolution=self.config.local_submap_resolution,
            )
            bev_match = match_query_points_to_submap_bev(
                query_points_xyz=query_points,
                query_bev_config=self.query_bev_config,
                submap_bev=submap.bev,
                submap_bev_config=submap.bev_config,
                submap_center_xy=submap.center_xy,
                coarse_yaw_rad=math.radians(float(candidate["query_rotation_deg"])),
                yaw_half_range_deg=self.config.coarse_yaw_half_range_deg,
                yaw_step_deg=self.config.coarse_yaw_step_deg,
                match_channel=self.config.match_channel,
                coarse_match_method=self.config.coarse_match_method,
            )
            initial_pose = pose_from_xy_yaw_z(
                x=bev_match.world_xy[0],
                y=bev_match.world_xy[1],
                yaw_rad=bev_match.yaw_rad,
                z=0.0,
            )
            icp_result = refine_pose_with_icp(
                query_points_xyz_sensor=query_points,
                map_points_xyz_world=submap.points_xyz_world,
                initial_pose_4x4=initial_pose,
                voxel_size_m=self.config.voxel_size_m,
                max_iterations=self.config.icp_max_iterations,
                max_correspondence_distance_m=self.config.icp_max_correspondence_distance_m,
                min_correspondences=self.config.icp_min_correspondences,
            )
            icp_valid = bool(
                icp_result.num_inliers >= int(self.config.icp_min_correspondences)
                and np.isfinite(icp_result.rmse)
            )
            candidate_pose = icp_result.pose_4x4 if icp_valid else initial_pose
            temporal_position_jump_m, temporal_yaw_jump_deg = self._temporal_jump_metrics(
                candidate_pose,
                predicted_pose_4x4,
            )
            temporal_gate_valid = self._is_temporal_gate_valid(
                temporal_position_jump_m,
                temporal_yaw_jump_deg,
            )
            temporal_score = self._temporal_score(candidate_pose, predicted_pose_4x4)
            final_score = (
                float(self.config.retrieval_score_weight) * float(candidate["score"])
                + float(self.config.bev_score_weight) * float(bev_match.score)
                + float(self.config.icp_inlier_weight) * float(icp_result.inlier_ratio)
                - float(self.config.icp_rmse_weight) * float(icp_result.rmse if np.isfinite(icp_result.rmse) else 10.0)
                + float(self.config.temporal_weight) * float(temporal_score)
                - (0.0 if icp_valid else float(self.config.invalid_icp_penalty))
            )
            if predicted_pose_4x4 is not None and not temporal_gate_valid:
                final_score = -1.0e9
            candidate_results.append(
                CandidateLocalizationResult(
                    patch_id=int(candidate["patch_id"]),
                    coarse_score=float(candidate["score"]),
                    coarse_query_rotation_deg=float(candidate["query_rotation_deg"]),
                    initial_world_xy=(float(bev_match.world_xy[0]), float(bev_match.world_xy[1])),
                    initial_yaw_deg=float(math.degrees(bev_match.yaw_rad)),
                    bev_score=float(bev_match.score),
                    icp_inlier_ratio=float(icp_result.inlier_ratio),
                    icp_rmse=float(icp_result.rmse),
                    icp_valid=icp_valid,
                    temporal_position_jump_m=temporal_position_jump_m,
                    temporal_yaw_jump_deg=temporal_yaw_jump_deg,
                    temporal_gate_valid=temporal_gate_valid,
                    temporal_score=float(temporal_score),
                    final_score=float(final_score),
                    final_pose_4x4=candidate_pose.copy(),
                )
            )

        best_candidate = max(candidate_results, key=lambda item: item.final_score)
        gt_pose = self.sequence_dataset.ground_truth.poses_4x4[frame_idx]
        if (
            predicted_pose_4x4 is not None
            and bool(self.config.fallback_to_predicted_pose_on_invalid)
            and not any(item.icp_valid and item.temporal_gate_valid for item in candidate_results)
        ):
            fallback_pose = np.asarray(predicted_pose_4x4, dtype=np.float64).copy()
            fallback_yaw_rad = float(
                wrap_to_pi(yaw_from_pose_matrix(fallback_pose) - yaw_from_pose_matrix(gt_pose))
            )
            pred_xy = fallback_pose[:2, 3]
            gt_xy = gt_pose[:2, 3]
            return {
                "frame_idx": int(frame_idx),
                "timestamp": float(self.sequence_dataset.frame_index[frame_idx].timestamp),
                "best_patch_id": int(best_candidate.patch_id),
                "pred_pose_4x4": fallback_pose.tolist(),
                "predicted_pose_4x4_from_tracker": predicted_pose_4x4.tolist(),
                "position_error_m": float(np.linalg.norm(pred_xy - gt_xy)),
                "yaw_error_deg": float(math.degrees(abs(fallback_yaw_rad))),
                "used_tracker_fallback": True,
                "selection_mode": "online_best",
                "selected_candidate_index": 0,
                "candidate_results": [
                    {
                        **{
                            key: value
                            for key, value in asdict(item).items()
                            if key != "final_pose_4x4"
                        },
                        "final_pose_4x4": item.final_pose_4x4.tolist(),
                    }
                    for item in sorted(candidate_results, key=lambda item: item.final_score, reverse=True)
                ],
            }

        gt_xy = gt_pose[:2, 3]
        pred_xy = best_candidate.final_pose_4x4[:2, 3]
        position_error_m = float(np.linalg.norm(pred_xy - gt_xy))
        yaw_error_rad = float(
            wrap_to_pi(yaw_from_pose_matrix(best_candidate.final_pose_4x4) - yaw_from_pose_matrix(gt_pose))
        )
        result = {
            "frame_idx": int(frame_idx),
            "timestamp": float(self.sequence_dataset.frame_index[frame_idx].timestamp),
            "best_patch_id": int(best_candidate.patch_id),
            "pred_pose_4x4": best_candidate.final_pose_4x4.tolist(),
            "predicted_pose_4x4_from_tracker": predicted_pose_4x4.tolist() if predicted_pose_4x4 is not None else None,
            "position_error_m": position_error_m,
            "yaw_error_deg": float(math.degrees(abs(yaw_error_rad))),
            "used_tracker_fallback": False,
            "selection_mode": "online_best",
            "selected_candidate_index": 0,
            "candidate_results": [
                {
                    **{
                        key: value
                        for key, value in asdict(item).items()
                        if key != "final_pose_4x4"
                    },
                    "final_pose_4x4": item.final_pose_4x4.tolist(),
                }
                for item in sorted(candidate_results, key=lambda item: item.final_score, reverse=True)
            ],
        }
        result = self._apply_top2_confusion_geometry_override(result)
        return self._apply_online_pose_stabilizer(result, predicted_pose_4x4)

    def _candidate_unary_score(self, candidate: Mapping[str, Any]) -> float:
        if "final_score" in candidate:
            return float(candidate["final_score"])
        bev_score = float(candidate.get("selected_bev_score", candidate["bev_score"]))
        rmse = float(candidate["icp_rmse"]) if np.isfinite(candidate["icp_rmse"]) else 10.0
        score = (
            float(self.config.retrieval_score_weight) * float(candidate["coarse_score"])
            + float(self.config.bev_score_weight) * bev_score
            + float(self.config.icp_inlier_weight) * float(candidate["icp_inlier_ratio"])
            - float(self.config.icp_rmse_weight) * rmse
        )
        if "deep_match_probability" in candidate:
            score += (
                float(self.config.deep_matcher_score_weight)
                * float(candidate["deep_match_probability"])
                * self._deep_geometry_gate(candidate, bev_score=bev_score)
            )
        if "deep_pose_confidence" in candidate:
            score += (
                float(self.config.deep_pose_confidence_weight)
                * float(candidate["deep_pose_confidence"])
                * self._deep_geometry_gate(candidate, bev_score=bev_score)
            )
        if not bool(candidate.get("icp_valid", True)):
            score -= float(self.config.invalid_icp_penalty)
        return float(score)

    @staticmethod
    def _candidate_pose_from_mapping(candidate: Mapping[str, Any]) -> np.ndarray:
        return np.asarray(candidate["final_pose_4x4"], dtype=np.float64)

    def _deep_geometry_gate(
        self,
        candidate: Mapping[str, Any],
        *,
        bev_score: float | None = None,
    ) -> float:
        if not bool(candidate.get("icp_valid", True)):
            return 0.0
        selected_bev_score = float(
            bev_score if bev_score is not None else candidate.get("selected_bev_score", candidate["bev_score"])
        )
        inlier_ratio = float(candidate.get("icp_inlier_ratio", 0.0))
        bev_threshold = max(float(self.config.deep_geometry_gate_min_bev_score), 1.0e-6)
        inlier_threshold = min(max(float(self.config.deep_geometry_gate_min_inlier_ratio), 0.0), 0.999999)
        bev_support = min(max(selected_bev_score / bev_threshold, 0.0), 1.0)
        inlier_support = min(max((inlier_ratio - inlier_threshold) / (1.0 - inlier_threshold), 0.0), 1.0)
        return float(bev_support * inlier_support)

    def _online_candidate_support(self, candidate: Mapping[str, Any]) -> float:
        if not bool(candidate.get("icp_valid", True)):
            return 0.0
        bev_score = float(candidate.get("selected_bev_score", candidate["bev_score"]))
        inlier_ratio = float(candidate.get("icp_inlier_ratio", 0.0))
        bev_threshold = max(float(self.config.online_stabilizer_min_bev_score), 1.0e-6)
        inlier_threshold = max(float(self.config.online_stabilizer_min_icp_inlier_ratio), 1.0e-6)
        bev_support = min(max(bev_score / bev_threshold, 0.0), 1.0)
        inlier_support = min(max(inlier_ratio / inlier_threshold, 0.0), 1.0)
        return float(bev_support * inlier_support)

    def _candidate_jump_to_prediction(
        self,
        candidate: Mapping[str, Any],
        predicted_pose_4x4: np.ndarray | None,
    ) -> tuple[float | None, float | None]:
        if predicted_pose_4x4 is None:
            return None, None
        if "temporal_position_jump_m" in candidate and "temporal_yaw_jump_deg" in candidate:
            return (
                None if candidate["temporal_position_jump_m"] is None else float(candidate["temporal_position_jump_m"]),
                None if candidate["temporal_yaw_jump_deg"] is None else float(candidate["temporal_yaw_jump_deg"]),
            )
        return self._temporal_jump_metrics(
            self._candidate_pose_from_mapping(candidate),
            predicted_pose_4x4,
        )

    def _online_stabilizer_candidate_score(
        self,
        candidate: Mapping[str, Any],
        predicted_pose_4x4: np.ndarray | None,
    ) -> float:
        temporal_score = float(candidate.get("temporal_score", 0.0))
        if predicted_pose_4x4 is not None and temporal_score <= 0.0:
            temporal_score = self._temporal_score(
                self._candidate_pose_from_mapping(candidate),
                predicted_pose_4x4,
            )
        return float(
            self._candidate_unary_score(candidate)
            + float(self.config.online_stabilizer_temporal_bonus) * temporal_score
        )

    def _apply_online_pose_stabilizer(
        self,
        frame_result: dict[str, Any],
        predicted_pose_4x4: np.ndarray | None,
    ) -> dict[str, Any]:
        if (
            predicted_pose_4x4 is None
            or not bool(self.config.use_online_pose_stabilizer)
            or not frame_result.get("candidate_results")
        ):
            return frame_result
        candidate_results = list(frame_result["candidate_results"])
        selected_idx = int(frame_result.get("selected_candidate_index", 0))
        selected_candidate = candidate_results[selected_idx]
        selected_score = self._online_stabilizer_candidate_score(selected_candidate, predicted_pose_4x4)
        max_xy_jump = float(self.config.online_stabilizer_max_position_jump_m)
        max_yaw_jump = float(self.config.online_stabilizer_max_yaw_jump_deg)
        admissible_candidates: list[tuple[int, float]] = []
        for candidate_idx, candidate in enumerate(candidate_results):
            xy_jump_m, yaw_jump_deg = self._candidate_jump_to_prediction(candidate, predicted_pose_4x4)
            if xy_jump_m is None or yaw_jump_deg is None:
                continue
            if xy_jump_m <= max_xy_jump and yaw_jump_deg <= max_yaw_jump:
                admissible_candidates.append(
                    (candidate_idx, self._online_stabilizer_candidate_score(candidate, predicted_pose_4x4))
                )
        if admissible_candidates:
            best_admissible_idx, best_admissible_score = max(admissible_candidates, key=lambda item: item[1])
            if (
                best_admissible_idx != selected_idx
                and best_admissible_score >= selected_score - float(self.config.online_stabilizer_score_margin)
            ):
                return self._replace_selected_candidate(
                    frame_result,
                    candidate_results,
                    best_admissible_idx,
                    selection_mode="online_stabilized",
                    selected_candidate_score=best_admissible_score,
                )
            return frame_result
        selected_support = self._online_candidate_support(selected_candidate)
        if (
            bool(self.config.online_stabilizer_fallback_to_tracker)
            and selected_support < 1.0
        ):
            return self._replace_with_tracker_prediction(frame_result, predicted_pose_4x4)
        return frame_result

    def _top2_confusion_geometry_score(self, candidate: Mapping[str, Any]) -> float:
        if not bool(candidate.get("icp_valid", True)):
            return -1.0e9
        bev_score = float(candidate.get("selected_bev_score", candidate["bev_score"]))
        inlier_ratio = float(candidate.get("icp_inlier_ratio", 0.0))
        rmse = float(candidate["icp_rmse"]) if np.isfinite(candidate["icp_rmse"]) else 10.0
        return float(
            float(self.config.top2_confusion_bev_weight) * bev_score
            + float(self.config.top2_confusion_inlier_weight) * inlier_ratio
            - float(self.config.top2_confusion_rmse_weight) * rmse
        )

    def _apply_top2_confusion_geometry_override(
        self,
        frame_result: dict[str, Any],
    ) -> dict[str, Any]:
        if (
            not bool(self.config.use_top2_confusion_geometry_override)
            or not frame_result.get("candidate_results")
            or len(frame_result["candidate_results"]) < 2
        ):
            return frame_result
        configured_pairs = {
            tuple(sorted((int(pair[0]), int(pair[1]))))
            for pair in self.config.top2_confusion_pairs
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        }
        if not configured_pairs:
            return frame_result
        candidate_results = list(frame_result["candidate_results"])
        first_candidate = candidate_results[0]
        second_candidate = candidate_results[1]
        pair = tuple(sorted((int(first_candidate["patch_id"]), int(second_candidate["patch_id"]))))
        if pair not in configured_pairs:
            return frame_result
        final_score_gap = float(first_candidate["final_score"]) - float(second_candidate["final_score"])
        if final_score_gap > float(self.config.top2_confusion_final_gap_max):
            return frame_result
        first_geometry_score = self._top2_confusion_geometry_score(first_candidate)
        second_geometry_score = self._top2_confusion_geometry_score(second_candidate)
        geometry_margin = float(self.config.top2_confusion_geometry_margin)
        if second_geometry_score <= first_geometry_score + geometry_margin:
            return frame_result
        reordered_candidates = list(candidate_results)
        reordered_candidates[0], reordered_candidates[1] = reordered_candidates[1], reordered_candidates[0]
        updated_frame_result = dict(frame_result)
        updated_frame_result["candidate_results"] = reordered_candidates
        updated_frame_result = self._replace_selected_candidate(
            updated_frame_result,
            reordered_candidates,
            0,
            selection_mode="top2_confusion_geometry_override",
            selected_candidate_score=float(reordered_candidates[0]["final_score"]),
        )
        updated_frame_result["top2_confusion_geometry_override"] = {
            "pair": [int(pair[0]), int(pair[1])],
            "final_score_gap": float(final_score_gap),
            "selected_geometry_score": float(second_geometry_score),
            "rejected_geometry_score": float(first_geometry_score),
        }
        return updated_frame_result

    def _replace_selected_candidate(
        self,
        frame_result: dict[str, Any],
        candidate_results: list[Mapping[str, Any]],
        selected_candidate_idx: int,
        *,
        selection_mode: str,
        selected_candidate_score: float | None = None,
    ) -> dict[str, Any]:
        selected_candidate = candidate_results[selected_candidate_idx]
        gt_pose = self.sequence_dataset.ground_truth.poses_4x4[frame_result["frame_idx"]]
        selected_pose = self._candidate_pose_from_mapping(selected_candidate)
        position_error_m = float(np.linalg.norm(selected_pose[:2, 3] - gt_pose[:2, 3]))
        yaw_error_rad = float(
            wrap_to_pi(yaw_from_pose_matrix(selected_pose) - yaw_from_pose_matrix(gt_pose))
        )
        updated_frame_result = dict(frame_result)
        updated_frame_result["pred_pose_4x4"] = selected_pose.tolist()
        updated_frame_result["best_patch_id"] = int(selected_candidate["patch_id"])
        updated_frame_result["position_error_m"] = position_error_m
        updated_frame_result["yaw_error_deg"] = float(math.degrees(abs(yaw_error_rad)))
        updated_frame_result["selection_mode"] = selection_mode
        updated_frame_result["selected_candidate_index"] = int(selected_candidate_idx)
        if selected_candidate_score is not None:
            updated_frame_result["selected_candidate_score"] = float(selected_candidate_score)
        return updated_frame_result

    def _replace_with_tracker_prediction(
        self,
        frame_result: dict[str, Any],
        predicted_pose_4x4: np.ndarray,
    ) -> dict[str, Any]:
        gt_pose = self.sequence_dataset.ground_truth.poses_4x4[frame_result["frame_idx"]]
        pred_xy = np.asarray(predicted_pose_4x4[:2, 3], dtype=np.float64)
        gt_xy = np.asarray(gt_pose[:2, 3], dtype=np.float64)
        pred_yaw = yaw_from_pose_matrix(predicted_pose_4x4)
        gt_yaw = yaw_from_pose_matrix(gt_pose)
        updated_frame_result = dict(frame_result)
        updated_frame_result["pred_pose_4x4"] = np.asarray(predicted_pose_4x4, dtype=np.float64).tolist()
        updated_frame_result["position_error_m"] = float(np.linalg.norm(pred_xy - gt_xy))
        updated_frame_result["yaw_error_deg"] = float(math.degrees(abs(wrap_to_pi(pred_yaw - gt_yaw))))
        updated_frame_result["selection_mode"] = "online_tracker_fallback"
        updated_frame_result["used_tracker_fallback"] = True
        return updated_frame_result

    @staticmethod
    def _transition_score(
        prev_candidate: Mapping[str, Any],
        current_candidate: Mapping[str, Any],
        position_sigma_m: float,
        yaw_sigma_deg: float,
        stay_bonus: float,
    ) -> float:
        prev_pose = np.asarray(prev_candidate["final_pose_4x4"], dtype=np.float64)
        current_pose = np.asarray(current_candidate["final_pose_4x4"], dtype=np.float64)
        xy_jump_m = float(np.linalg.norm(current_pose[:2, 3] - prev_pose[:2, 3]))
        yaw_jump_deg = abs(
            math.degrees(
                float(
                    wrap_to_pi(
                        yaw_from_pose_matrix(current_pose) - yaw_from_pose_matrix(prev_pose)
                    )
                )
            )
        )
        score = (
            -0.5 * (xy_jump_m / max(position_sigma_m, 1.0e-6)) ** 2
            -0.5 * (yaw_jump_deg / max(yaw_sigma_deg, 1.0e-6)) ** 2
        )
        if int(prev_candidate["patch_id"]) == int(current_candidate["patch_id"]):
            score += float(stay_bonus)
        return float(score)

    def _apply_sequence_smoothing(self, frame_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not frame_results or not bool(self.config.use_sequence_smoothing):
            return frame_results
        dp_scores: list[np.ndarray] = []
        backpointers: list[np.ndarray] = []
        first_candidates = frame_results[0]["candidate_results"]
        dp_scores.append(
            np.asarray(
                [self._candidate_unary_score(candidate) for candidate in first_candidates],
                dtype=np.float64,
            )
        )
        backpointers.append(np.full(len(first_candidates), -1, dtype=np.int32))
        position_sigma_m = float(self.config.sequence_smoothing_position_sigma_m)
        yaw_sigma_deg = float(self.config.sequence_smoothing_yaw_sigma_deg)
        stay_bonus = float(self.config.sequence_smoothing_stay_bonus)
        for frame_idx in range(1, len(frame_results)):
            previous_candidates = frame_results[frame_idx - 1]["candidate_results"]
            current_candidates = frame_results[frame_idx]["candidate_results"]
            current_dp = np.full(len(current_candidates), -1.0e18, dtype=np.float64)
            current_backpointer = np.full(len(current_candidates), -1, dtype=np.int32)
            for current_idx, current_candidate in enumerate(current_candidates):
                unary_score = self._candidate_unary_score(current_candidate)
                best_score = -1.0e18
                best_prev_idx = -1
                for prev_idx, prev_candidate in enumerate(previous_candidates):
                    transition_score = self._transition_score(
                        prev_candidate,
                        current_candidate,
                        position_sigma_m=position_sigma_m,
                        yaw_sigma_deg=yaw_sigma_deg,
                        stay_bonus=stay_bonus,
                    )
                    candidate_score = float(dp_scores[-1][prev_idx]) + transition_score
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_prev_idx = prev_idx
                current_dp[current_idx] = unary_score + best_score
                current_backpointer[current_idx] = best_prev_idx
            dp_scores.append(current_dp)
            backpointers.append(current_backpointer)

        selected_indices = [0] * len(frame_results)
        selected_indices[-1] = int(np.argmax(dp_scores[-1]))
        for frame_idx in range(len(frame_results) - 1, 0, -1):
            selected_indices[frame_idx - 1] = int(
                backpointers[frame_idx][selected_indices[frame_idx]]
            )

        smoothed_results: list[dict[str, Any]] = []
        for frame_idx, (frame_result, selected_candidate_idx) in enumerate(zip(frame_results, selected_indices)):
            selected_candidate = frame_result["candidate_results"][selected_candidate_idx]
            gt_pose = self.sequence_dataset.ground_truth.poses_4x4[frame_result["frame_idx"]]
            selected_pose = np.asarray(selected_candidate["final_pose_4x4"], dtype=np.float64)
            position_error_m = float(np.linalg.norm(selected_pose[:2, 3] - gt_pose[:2, 3]))
            yaw_error_rad = float(
                wrap_to_pi(yaw_from_pose_matrix(selected_pose) - yaw_from_pose_matrix(gt_pose))
            )
            updated_frame_result = dict(frame_result)
            updated_frame_result["online_best_patch_id"] = int(frame_result["best_patch_id"])
            updated_frame_result["online_pred_pose_4x4"] = frame_result["pred_pose_4x4"]
            updated_frame_result["best_patch_id"] = int(selected_candidate["patch_id"])
            updated_frame_result["pred_pose_4x4"] = selected_candidate["final_pose_4x4"]
            updated_frame_result["position_error_m"] = position_error_m
            updated_frame_result["yaw_error_deg"] = float(math.degrees(abs(yaw_error_rad)))
            updated_frame_result["selection_mode"] = "sequence_smoothing"
            updated_frame_result["selected_candidate_index"] = int(selected_candidate_idx)
            updated_frame_result["selected_candidate_score"] = float(dp_scores[frame_idx][selected_candidate_idx])
            smoothed_results.append(updated_frame_result)
        return smoothed_results

    def _apply_online_patch_hysteresis(self, frame_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not frame_results or not bool(self.config.use_online_patch_hysteresis):
            return frame_results
        switch_margin = float(self.config.online_patch_switch_margin)
        force_prev_max = self.config.online_patch_switch_force_prev_deep_prob_max
        force_proposed_min = self.config.online_patch_switch_force_proposed_deep_prob_min
        force_min_advantage = float(self.config.online_patch_switch_force_min_score_advantage)
        challenger_min_deep_prob = self.config.online_patch_challenger_min_deep_prob
        challenger_min_support = self.config.online_patch_challenger_min_support
        challenger_min_streak = (
            None
            if self.config.online_patch_challenger_min_streak is None
            else max(1, int(self.config.online_patch_challenger_min_streak))
        )
        tracker_release_min_streak = (
            None
            if self.config.online_patch_tracker_release_min_streak is None
            else max(1, int(self.config.online_patch_tracker_release_min_streak))
        )
        tracker_release_prev_deep_prob_max = self.config.online_patch_tracker_release_prev_deep_prob_max
        tracker_release_prev_support_max = self.config.online_patch_tracker_release_prev_support_max
        dynamic_margin_min_scale = max(0.0, float(self.config.online_patch_dynamic_margin_min_scale))
        dynamic_margin_max_scale = max(dynamic_margin_min_scale, float(self.config.online_patch_dynamic_margin_max_scale))
        dynamic_margin_tracker_factor = max(0.0, float(self.config.online_patch_dynamic_margin_tracker_factor))
        stabilized_results: list[dict[str, Any]] = []
        previous_patch_id: int | None = None
        challenger_patch_id: int | None = None
        challenger_run_length = 0

        def _candidate_support(candidate: Mapping[str, Any]) -> float:
            return float(self._online_candidate_support(candidate))

        for frame_result in frame_results:
            candidate_results = list(frame_result.get("candidate_results", []))
            if not candidate_results:
                stabilized_results.append(frame_result)
                previous_patch_id = int(frame_result["best_patch_id"])
                challenger_patch_id = None
                challenger_run_length = 0
                continue
            proposed_idx = 0
            proposed_candidate = candidate_results[proposed_idx]
            if previous_patch_id is not None and int(proposed_candidate["patch_id"]) != previous_patch_id:
                previous_patch_candidate = next(
                    (candidate for candidate in candidate_results if int(candidate["patch_id"]) == previous_patch_id),
                    None,
                )
                if (
                    previous_patch_candidate is not None
                ):
                    proposed_score = float(proposed_candidate["final_score"])
                    previous_score = float(previous_patch_candidate["final_score"])
                    proposed_deep = float(proposed_candidate.get("deep_match_probability", 0.0))
                    previous_deep = float(previous_patch_candidate.get("deep_match_probability", 0.0))
                    proposed_support = _candidate_support(proposed_candidate)
                    previous_support = _candidate_support(previous_patch_candidate)
                    previous_selected_init = str(previous_patch_candidate.get("selected_init_source") or "")

                    strong_challenger = (
                        (challenger_min_deep_prob is None or proposed_deep >= float(challenger_min_deep_prob))
                        and (challenger_min_support is None or proposed_support >= float(challenger_min_support))
                    )
                    proposed_patch_id = int(proposed_candidate["patch_id"])
                    if strong_challenger:
                        if challenger_patch_id == proposed_patch_id:
                            challenger_run_length += 1
                        else:
                            challenger_patch_id = proposed_patch_id
                            challenger_run_length = 1
                    else:
                        challenger_patch_id = None
                        challenger_run_length = 0

                    previous_retention = 0.5 * previous_support + 0.5 * previous_deep
                    challenger_pressure = 0.5 * proposed_support + 0.5 * proposed_deep
                    if dynamic_margin_max_scale > dynamic_margin_min_scale:
                        raw_scale = 1.0 + (previous_retention - challenger_pressure)
                        dynamic_scale = float(
                            np.clip(raw_scale, dynamic_margin_min_scale, dynamic_margin_max_scale)
                        )
                    else:
                        dynamic_scale = dynamic_margin_min_scale
                    if previous_selected_init == "tracker_init":
                        dynamic_scale *= dynamic_margin_tracker_factor
                    dynamic_margin = float(switch_margin) * float(dynamic_scale)

                    force_switch_streak_ok = (
                        challenger_min_streak is None
                        or (
                            strong_challenger
                            and challenger_run_length >= challenger_min_streak
                        )
                    )
                    challenger_streak_ready = force_switch_streak_ok
                    force_switch = (
                        force_prev_max is not None
                        and force_proposed_min is not None
                        and previous_deep <= float(force_prev_max)
                        and proposed_deep >= float(force_proposed_min)
                        and proposed_score >= previous_score + force_min_advantage
                        and force_switch_streak_ok
                    )
                    streak_switch = (
                        challenger_min_streak is not None
                        and challenger_min_streak > 0
                        and strong_challenger
                        and challenger_run_length >= challenger_min_streak
                        and proposed_score >= previous_score + force_min_advantage
                    )
                    tracker_release_switch = (
                        tracker_release_min_streak is not None
                        and tracker_release_min_streak > 0
                        and previous_selected_init == "tracker_init"
                        and strong_challenger
                        and challenger_run_length >= tracker_release_min_streak
                        and proposed_score >= previous_score + force_min_advantage
                        and (
                            tracker_release_prev_deep_prob_max is None
                            or previous_deep <= float(tracker_release_prev_deep_prob_max)
                        )
                        and (
                            tracker_release_prev_support_max is None
                            or previous_support <= float(tracker_release_prev_support_max)
                        )
                    )
                    if (
                        not force_switch
                        and not streak_switch
                        and not tracker_release_switch
                        and (
                            not challenger_streak_ready
                            or proposed_score < previous_score + dynamic_margin
                        )
                    ):
                        previous_idx = next(
                            idx for idx, candidate in enumerate(candidate_results)
                            if int(candidate["patch_id"]) == previous_patch_id
                        )
                        updated = self._replace_selected_candidate(
                            frame_result,
                            candidate_results,
                            previous_idx,
                            selection_mode="online_patch_hysteresis",
                            selected_candidate_score=float(previous_patch_candidate["final_score"]),
                        )
                        updated["online_patch_hysteresis_state"] = {
                            "previous_patch_id": int(previous_patch_id),
                            "proposed_patch_id": int(proposed_candidate["patch_id"]),
                            "previous_final_score": previous_score,
                            "proposed_final_score": proposed_score,
                            "previous_deep_match_probability": previous_deep,
                            "proposed_deep_match_probability": proposed_deep,
                            "previous_support": previous_support,
                            "proposed_support": proposed_support,
                            "challenger_run_length": int(challenger_run_length),
                            "dynamic_margin": float(dynamic_margin),
                            "previous_selected_init_source": previous_selected_init,
                        }
                        stabilized_results.append(updated)
                        previous_patch_id = int(updated["best_patch_id"])
                        challenger_patch_id = proposed_patch_id if strong_challenger else None
                        continue
                    if force_switch or streak_switch or tracker_release_switch:
                        if tracker_release_switch:
                            selection_mode = "online_patch_hysteresis_tracker_release"
                        elif streak_switch:
                            selection_mode = "online_patch_hysteresis_challenger_streak"
                        else:
                            selection_mode = "online_patch_hysteresis_deep_override"
                        updated = self._replace_selected_candidate(
                            frame_result,
                            candidate_results,
                            proposed_idx,
                            selection_mode=selection_mode,
                            selected_candidate_score=proposed_score,
                        )
                        updated["online_patch_hysteresis_override"] = {
                            "previous_patch_id": int(previous_patch_id),
                            "proposed_patch_id": int(proposed_candidate["patch_id"]),
                            "previous_deep_match_probability": previous_deep,
                            "proposed_deep_match_probability": proposed_deep,
                            "previous_final_score": previous_score,
                            "proposed_final_score": proposed_score,
                            "previous_support": previous_support,
                            "proposed_support": proposed_support,
                            "challenger_run_length": int(challenger_run_length),
                            "dynamic_margin": float(dynamic_margin),
                            "previous_selected_init_source": previous_selected_init,
                            "force_switch": bool(force_switch),
                            "streak_switch": bool(streak_switch),
                            "tracker_release_switch": bool(tracker_release_switch),
                        }
                        stabilized_results.append(updated)
                        previous_patch_id = int(updated["best_patch_id"])
                        challenger_patch_id = None
                        challenger_run_length = 0
                        continue
            stabilized_results.append(frame_result)
            previous_patch_id = int(frame_result["best_patch_id"])
            challenger_patch_id = None
            challenger_run_length = 0
        return stabilized_results

    def _apply_persistent_patch_override(
        self,
        frame_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not frame_results or not bool(self.config.use_persistent_patch_override):
            return frame_results
        directed_pairs = [
            (int(pair[0]), int(pair[1]))
            for pair in getattr(self.config, "persistent_patch_override_pairs", ())
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        ]
        if not directed_pairs:
            return frame_results
        topn = max(1, int(self.config.persistent_patch_override_topn))
        min_run_length = max(2, int(self.config.persistent_patch_override_min_run_length))
        min_geometry_score = float(self.config.persistent_patch_override_min_geometry_score)
        updated_results = list(frame_results)
        frame_idx = 0
        while frame_idx < len(updated_results):
            current_patch_id = int(updated_results[frame_idx].get("best_patch_id", -1))
            applied = False
            for source_patch_id, target_patch_id in directed_pairs:
                if current_patch_id != source_patch_id:
                    continue
                run_end = frame_idx
                target_candidate_indices: list[int] = []
                target_geometry_scores: list[float] = []
                while run_end < len(updated_results):
                    current_frame_result = updated_results[run_end]
                    if int(current_frame_result.get("best_patch_id", -1)) != source_patch_id:
                        break
                    candidate_results = list(current_frame_result.get("candidate_results", []))
                    target_candidate_idx = next(
                        (
                            candidate_idx
                            for candidate_idx, candidate in enumerate(candidate_results[:topn])
                            if int(candidate["patch_id"]) == target_patch_id
                        ),
                        None,
                    )
                    if target_candidate_idx is None:
                        break
                    target_candidate = candidate_results[target_candidate_idx]
                    target_geometry_score = self._top2_confusion_geometry_score(target_candidate)
                    if target_geometry_score < min_geometry_score:
                        break
                    target_candidate_indices.append(int(target_candidate_idx))
                    target_geometry_scores.append(float(target_geometry_score))
                    run_end += 1
                run_length = run_end - frame_idx
                if run_length < min_run_length:
                    continue
                for local_offset, selected_candidate_idx in enumerate(target_candidate_indices):
                    target_frame_result = updated_results[frame_idx + local_offset]
                    candidate_results = list(target_frame_result.get("candidate_results", []))
                    replaced = self._replace_selected_candidate(
                        target_frame_result,
                        candidate_results,
                        selected_candidate_idx,
                        selection_mode="persistent_patch_override",
                        selected_candidate_score=float(candidate_results[selected_candidate_idx].get("final_score", 0.0)),
                    )
                    replaced["persistent_patch_override"] = {
                        "source_patch_id": int(source_patch_id),
                        "target_patch_id": int(target_patch_id),
                        "run_length": int(run_length),
                        "target_geometry_score": float(target_geometry_scores[local_offset]),
                    }
                    updated_results[frame_idx + local_offset] = replaced
                frame_idx = run_end
                applied = True
                break
            if not applied:
                frame_idx += 1
        return updated_results


def localize_sequence(
    config,
    frame_start: int = 0,
    num_frames: int | None = None,
    frame_stride: int = 1,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    localizer = FineLocalizer(config)
    last_frame = len(localizer.sequence_dataset) if num_frames is None else min(
        len(localizer.sequence_dataset),
        int(frame_start) + int(num_frames),
    )
    frame_indices = list(range(int(frame_start), last_frame, max(1, int(frame_stride))))
    frame_results: list[dict[str, Any]] = []
    accepted_poses: list[np.ndarray] = []
    for frame_idx in frame_indices:
        predicted_pose_4x4 = localizer._predict_pose_4x4(
            accepted_poses,
            accepted_frame_results=frame_results,
        )
        frame_result = localizer.localize_frame(frame_idx, predicted_pose_4x4=predicted_pose_4x4)
        frame_results.append(frame_result)
        accepted_poses.append(np.asarray(frame_result["pred_pose_4x4"], dtype=np.float64))
    frame_results = localizer._apply_online_patch_hysteresis(frame_results)
    frame_results = localizer._apply_persistent_patch_override(frame_results)
    frame_results = localizer._apply_sequence_smoothing(frame_results)
    position_errors = np.asarray([item["position_error_m"] for item in frame_results], dtype=np.float64)
    yaw_errors = np.asarray([item["yaw_error_deg"] for item in frame_results], dtype=np.float64)
    report = {
        "num_eval_frames": len(frame_results),
        "frame_start": int(frame_start),
        "frame_stride": int(frame_stride),
        "selection_mode": frame_results[0]["selection_mode"] if frame_results else "online_best",
        "mean_position_error_m": float(position_errors.mean()) if position_errors.size else None,
        "median_position_error_m": float(np.median(position_errors)) if position_errors.size else None,
        "mean_yaw_error_deg": float(yaw_errors.mean()) if yaw_errors.size else None,
        "median_yaw_error_deg": float(np.median(yaw_errors)) if yaw_errors.size else None,
        "frames_with_position_error_below_1m": float(np.mean(position_errors < 1.0)) if position_errors.size else None,
        "frames_with_position_error_below_0p5m": float(np.mean(position_errors < 0.5)) if position_errors.size else None,
        "frame_results": frame_results,
    }
    if output_json is None:
        output_path = load_fine_localization_config(config).output_json
    else:
        output_path = str(output_json)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fine localization from coarse retrieval candidates.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    localize_sequence(
        config=args.config,
        frame_start=args.frame_start,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
