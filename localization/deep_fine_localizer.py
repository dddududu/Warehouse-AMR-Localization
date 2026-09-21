from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from runtime_compat import ensure_runtime_compatibility

ensure_runtime_compatibility()

import torch

from dataset_io.depth_loader import load_depth_png
from dataset_io.fine_localization_dataset import build_stereo_geometry_features
from dataset_io.image_loader import load_rgb_image
from geometry.yaw_utils import wrap_to_pi, yaw_from_pose_matrix
from localization.bev_matcher import match_query_points_to_submap_bev
from localization.fine_localizer import FineLocalizer
from localization.icp_refiner import pose_from_xy_yaw_z, refine_pose_with_icp
from localization.multi_hypothesis_tracker import MultiHypothesisState, select_diverse_hypotheses
from localization.submap_builder import build_local_submap
from models.candidate_reranker import CandidateReranker
from models.confusion_pair_resolver import ConfusionPairResolver
from models.fine_pose_matcher import FinePoseMatcher
from models.occlusion_predictor import StereoOcclusionPredictor
from models.reliability_gate import (
    RELIABILITY_GATE_SOURCES,
    ReliabilityGate,
    build_reliability_gate_features,
)
from preprocess.bev_builder import points_to_bev
from preprocess.dynamic_point_filter import semantic_label_ratio


class DeepFineLocalizer(FineLocalizer):
    def __init__(self, config) -> None:
        super().__init__(config)
        if not self.config.deep_matcher_checkpoint_path:
            raise ValueError("deep_matcher_checkpoint_path must be set.")
        checkpoint = torch.load(self.config.deep_matcher_checkpoint_path, map_location=self.device)
        checkpoint_config = checkpoint.get("config", {})
        self.zero_query_bev = bool(checkpoint_config.get("zero_query_bev", False))
        self.deep_model = FinePoseMatcher(
            descriptor_dim=int(checkpoint_config.get("descriptor_dim", 128)),
            hidden_dim=int(checkpoint_config.get("hidden_dim", 128)),
            local_submap_size_m=float(checkpoint_config.get("local_submap_size_m", self.config.local_submap_size_m)),
            num_xy_bins=int(checkpoint_config.get("num_xy_bins", 31)),
            num_yaw_bins=int(checkpoint_config.get("num_yaw_bins", 72)),
            use_query_image=bool(checkpoint_config.get("use_query_image", True)),
            use_stereo_query_image=bool(checkpoint_config.get("use_stereo_query_image", True)),
            use_query_depth=bool(checkpoint_config.get("use_query_depth", True)),
            use_stereo_geometry=bool(checkpoint_config.get("use_stereo_geometry", True)),
        ).to(self.device)
        self.deep_model.load_state_dict(checkpoint["model"], strict=True)
        self.deep_model.eval()
        self.reliability_gate = None
        self.reliability_gate_feature_mean = None
        self.reliability_gate_feature_std = None
        self.reliability_gate_objective = "classification"
        self._reliability_gate_semantic_feature_cache: dict[int, np.ndarray] = {}
        if self.config.reliability_gate_checkpoint_path:
            gate_checkpoint = torch.load(
                self.config.reliability_gate_checkpoint_path,
                map_location=self.device,
            )
            gate_config = gate_checkpoint.get("config", {})
            self.reliability_gate_objective = str(gate_config.get("objective", "classification"))
            feature_dim = int(gate_config.get("feature_dim", 16))
            self.reliability_gate = ReliabilityGate(
                feature_dim=feature_dim,
                hidden_dim=int(gate_config.get("hidden_dim", 32)),
            ).to(self.device)
            self.reliability_gate.load_state_dict(gate_checkpoint["model"], strict=True)
            self.reliability_gate.eval()
            self.reliability_gate_feature_mean = np.asarray(
                gate_checkpoint["feature_mean"],
                dtype=np.float32,
            )
            self.reliability_gate_feature_std = np.asarray(
                gate_checkpoint["feature_std"],
                dtype=np.float32,
            )
        self.candidate_reranker = None
        if self.config.candidate_reranker_checkpoint_path:
            reranker_checkpoint = torch.load(self.config.candidate_reranker_checkpoint_path, map_location=self.device)
            reranker_cfg = reranker_checkpoint.get("config", {})
            reranker_state = reranker_checkpoint.get("model", {})
            patch_embedding_weight = reranker_state.get("patch_embedding.weight")
            num_patch_ids = int(patch_embedding_weight.shape[0]) if patch_embedding_weight is not None else 0
            patch_embedding_dim = int(patch_embedding_weight.shape[1]) if patch_embedding_weight is not None else 16
            self.candidate_reranker = CandidateReranker(
                pair_embedding_dim=int(checkpoint_config.get("hidden_dim", self.config.topk_candidates)),
                scalar_feature_dim=5,
                hidden_dim=int(reranker_cfg.get("hidden_dim", 128)),
                num_patch_ids=num_patch_ids,
                patch_embedding_dim=patch_embedding_dim,
            ).to(self.device)
            self.candidate_reranker.load_state_dict(reranker_checkpoint["model"], strict=True)
            self.candidate_reranker.eval()
        self.confusion_pair_resolver = None
        self.confusion_pair_to_id: dict[tuple[int, int], int] = {}
        self.confusion_pair_weight_overrides: dict[tuple[int, int], float] = {
            tuple(sorted((int(a), int(b)))): float(weight)
            for a, b, weight in self.config.confusion_pair_resolver_pair_weight_overrides
        }
        if self.config.confusion_pair_resolver_checkpoint_path:
            pair_checkpoint = torch.load(self.config.confusion_pair_resolver_checkpoint_path, map_location=self.device)
            pair_cfg = pair_checkpoint.get("config", {})
            target_pairs = [tuple(sorted((int(pair[0]), int(pair[1])))) for pair in pair_cfg.get("target_pairs", [])]
            self.confusion_pair_to_id = {pair: idx for idx, pair in enumerate(target_pairs)}
            self.confusion_pair_resolver = ConfusionPairResolver(
                num_pairs=max(1, len(target_pairs)),
                pair_embedding_dim=int(checkpoint_config.get("hidden_dim", 128)),
                scalar_feature_dim=5,
                hidden_dim=int(pair_cfg.get("hidden_dim", 128)),
            ).to(self.device)
            self.confusion_pair_resolver.load_state_dict(pair_checkpoint["model"], strict=True)
            self.confusion_pair_resolver.eval()
        resize_hw = checkpoint_config.get("image_resize_hw", (128, 192))
        self.image_resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
        self.depth_scale = float(checkpoint_config.get("depth_scale", 0.001))
        self.depth_min_m = float(checkpoint_config.get("depth_min_m", 0.1))
        self.depth_max_m = float(checkpoint_config.get("depth_max_m", 20.0))
        width_scale = float(self.image_resize_hw[1]) / float(self.sequence_dataset.camera_left.width)
        self.query_fx_px = float(self.sequence_dataset.camera_left.fx) * width_scale
        self.stereo_baseline_m = float(
            np.linalg.norm(
                self.sequence_dataset.calibration.T_cam1_os.translation
                - self.sequence_dataset.calibration.T_cam2_os.translation
            )
        )
        self.occlusion_predictor = None
        self.occlusion_predictor_resize_hw = tuple(
            int(value) for value in self.config.semantic_occlusion_predictor_resize_hw
        )
        if self.config.semantic_occlusion_predictor_checkpoint_path:
            occlusion_checkpoint = torch.load(
                self.config.semantic_occlusion_predictor_checkpoint_path,
                map_location=self.device,
            )
            occlusion_config = occlusion_checkpoint.get("config", {})
            self.occlusion_predictor_resize_hw = tuple(
                int(value)
                for value in occlusion_config.get(
                    "image_resize_hw",
                    self.occlusion_predictor_resize_hw,
                )
            )
            self.occlusion_predictor = StereoOcclusionPredictor(
                hidden_dim=int(occlusion_config.get("hidden_dim", 32)),
            ).to(self.device)
            self.occlusion_predictor.load_state_dict(occlusion_checkpoint["model"], strict=True)
            self.occlusion_predictor.eval()

    def _is_valid_icp(self, icp_result) -> bool:
        return bool(
            icp_result.num_inliers >= int(self.config.icp_min_correspondences)
            and np.isfinite(icp_result.rmse)
        )

    @staticmethod
    def _merge_semantic_geometry_hypotheses(
        filtered_candidates: list[dict[str, object]],
        raw_candidates: list[dict[str, object]],
    ) -> tuple[list[dict[str, object]], int]:
        raw_by_patch = {int(candidate["patch_id"]): candidate for candidate in raw_candidates}
        merged: list[dict[str, object]] = []
        accepted_count = 0
        for filtered_candidate in filtered_candidates:
            raw_candidate = raw_by_patch.get(int(filtered_candidate["patch_id"]))
            if raw_candidate is None:
                filtered_candidate["semantic_filter_geometry_accepted"] = True
                merged.append(filtered_candidate)
                accepted_count += 1
                continue
            filtered_valid = bool(filtered_candidate["icp_valid"])
            raw_valid = bool(raw_candidate["icp_valid"])
            use_filtered = filtered_valid and (
                not raw_valid
                or (
                    float(filtered_candidate["icp_inlier_ratio"])
                    >= float(raw_candidate["icp_inlier_ratio"])
                    and float(filtered_candidate["icp_rmse"])
                    <= float(raw_candidate["icp_rmse"])
                )
            )
            if use_filtered:
                filtered_candidate["semantic_filter_geometry_accepted"] = True
                merged.append(filtered_candidate)
                accepted_count += 1
            else:
                raw_candidate["semantic_filter_geometry_accepted"] = False
                merged.append(raw_candidate)
        return merged, accepted_count

    def _predict_semantic_occlusion_ratio_from_images(self, frame_idx: int) -> float | None:
        if self.occlusion_predictor is None:
            return None
        record = self.sequence_dataset.frame_index[int(frame_idx)]
        image_tensors = []
        for image_path, camera_model in (
            (record.image_left_path, self.sequence_dataset.camera_left),
            (record.image_right_path, self.sequence_dataset.camera_right),
        ):
            image, _ = load_rgb_image(
                image_path,
                camera_model=camera_model,
                use_undistort=False,
                resize_hw=self.occlusion_predictor_resize_hw,
            )
            image = np.asarray(image, dtype=np.float32) / 255.0
            image_tensors.append(np.transpose(image, (2, 0, 1)).astype(np.float32, copy=False))
        stereo_image = torch.from_numpy(np.concatenate(image_tensors, axis=0)[None]).to(self.device).float()
        with torch.no_grad():
            output = self.occlusion_predictor(stereo_image)
        return float(output["ratio"][0].detach().cpu().item())

    def _semantic_sidecar_refine_raw_winner(
        self,
        frame_idx: int,
        raw_frame_result: dict[str, object],
    ) -> object | None:
        selected_idx = int(raw_frame_result["selected_candidate_index"])
        raw_candidate = raw_frame_result["candidate_results"][selected_idx]
        if not bool(raw_candidate.get("icp_valid", False)):
            return None
        patch_id = int(raw_candidate["patch_id"])
        submap = build_local_submap(
            self.map_xyz,
            center_xy=self.patch_metadata[patch_id].center_xy,
            size_m=self.config.local_submap_size_m,
            resolution=self.config.local_submap_resolution,
        )
        filtered_points = self._build_query_points(frame_idx)
        refinement = refine_pose_with_icp(
            query_points_xyz_sensor=filtered_points,
            map_points_xyz_world=submap.points_xyz_world,
            initial_pose_4x4=np.asarray(raw_candidate["final_pose_4x4"], dtype=np.float64),
            voxel_size_m=self.config.voxel_size_m,
            max_iterations=self.config.icp_max_iterations,
            max_correspondence_distance_m=self.config.icp_max_correspondence_distance_m,
            min_correspondences=self.config.icp_min_correspondences,
            nearest_neighbor_backend="ckdtree",
        )
        if not self._is_valid_icp(refinement):
            return None
        if (
            float(refinement.inlier_ratio) < float(raw_candidate["icp_inlier_ratio"])
            or float(refinement.rmse) > float(raw_candidate["icp_rmse"])
        ):
            return None
        return refinement

    def _reliability_gate_semantic_features(self, frame_idx: int) -> np.ndarray:
        cached = self._reliability_gate_semantic_feature_cache.get(int(frame_idx))
        if cached is not None:
            return cached
        record = self.sequence_dataset.frame_index[int(frame_idx)]
        features = np.asarray(
            [
                semantic_label_ratio(
                    record.segmentation_greyscale_left_path,
                    record.segmentation_greyscale_right_path,
                    (13,),
                ),
                semantic_label_ratio(
                    record.segmentation_greyscale_left_path,
                    record.segmentation_greyscale_right_path,
                    (12, 13, 14, 15),
                ),
                semantic_label_ratio(
                    record.segmentation_greyscale_left_path,
                    record.segmentation_greyscale_right_path,
                    (5, 7, 9, 10, 11),
                ),
            ],
            dtype=np.float32,
        )
        self._reliability_gate_semantic_feature_cache[int(frame_idx)] = features
        return features

    def _tracker_pose_hypothesis_is_consistent(
        self,
        tracker_icp_result,
        reference_results: list[object],
    ) -> bool:
        max_xy_m = self.config.tracker_pose_init_consistency_max_xy_m
        max_yaw_deg = self.config.tracker_pose_init_consistency_max_yaw_deg
        if max_xy_m is None or max_yaw_deg is None:
            return True
        tracker_pose = np.asarray(tracker_icp_result.pose_4x4, dtype=np.float64)
        tracker_xy = tracker_pose[:2, 3]
        tracker_yaw = yaw_from_pose_matrix(tracker_pose)
        for reference_result in reference_results:
            if reference_result is None or not self._is_valid_icp(reference_result):
                continue
            reference_pose = np.asarray(reference_result.pose_4x4, dtype=np.float64)
            reference_xy = reference_pose[:2, 3]
            reference_yaw = yaw_from_pose_matrix(reference_pose)
            xy_gap_m = float(np.linalg.norm(tracker_xy - reference_xy))
            yaw_gap_deg = abs(math.degrees(float(wrap_to_pi(tracker_yaw - reference_yaw))))
            if xy_gap_m <= float(max_xy_m) and yaw_gap_deg <= float(max_yaw_deg):
                return True
        return False

    def _tracker_pose_hypothesis_passes_quality_gate(self, tracker_icp_result) -> bool:
        min_inlier_ratio = self.config.tracker_pose_init_min_inlier_ratio
        max_rmse = self.config.tracker_pose_init_max_rmse
        if min_inlier_ratio is not None and float(tracker_icp_result.inlier_ratio) < float(min_inlier_ratio):
            return False
        if max_rmse is not None:
            rmse = float(tracker_icp_result.rmse)
            if (not np.isfinite(rmse)) or rmse > float(max_rmse):
                return False
        return True

    def _select_best_pose_hypothesis(
        self,
        patch_id: int,
        bev_match_score: float,
        bev_icp_result,
        deep_bev_match_score: float | None,
        deep_pose_4x4: np.ndarray | None,
        deep_icp_result,
        tracker_icp_result,
        candidate_deep_match_probability: float,
        predicted_world_xy: np.ndarray,
        predicted_yaw_deg: float,
        predicted_pose_4x4: np.ndarray | None,
        tracker_selected_streak: int = 0,
        previous_selected_patch_streak: int = 0,
        previous_selected_patch_id: int | None = None,
        previous_selected_patch_deep_prob: float | None = None,
        previous_selected_init_source: str | None = None,
    ) -> tuple[str, object, float]:
        hypotheses: list[tuple[str, object, float]] = [("bev_init", bev_icp_result, float(bev_match_score))]
        if deep_pose_4x4 is not None and deep_icp_result is not None:
            hypotheses.append(("deep_init", deep_icp_result, float(deep_bev_match_score if deep_bev_match_score is not None else 0.0)))
        if (
            predicted_pose_4x4 is not None
            and tracker_icp_result is not None
            and self._tracker_pose_hypothesis_passes_quality_gate(tracker_icp_result)
            and self._tracker_pose_hypothesis_is_consistent(
                tracker_icp_result,
                [bev_icp_result, deep_icp_result],
            )
        ):
            hypotheses.append(("tracker_init", tracker_icp_result, float(bev_match_score)))
        best_source = "bev_init"
        best_result = bev_icp_result
        best_temporal = self._temporal_score(bev_icp_result.pose_4x4, predicted_pose_4x4)
        predicted_yaw_rad = math.radians(float(predicted_yaw_deg))
        scored_hypotheses: list[tuple[str, object, float, float]] = []
        for source, result, hypothesis_bev_score in hypotheses:
            pose_xy = np.asarray(result.pose_4x4[:2, 3], dtype=np.float64)
            pose_yaw_rad = yaw_from_pose_matrix(result.pose_4x4)
            xy_error_m = float(np.linalg.norm(pose_xy - predicted_world_xy))
            yaw_error_deg = abs(
                math.degrees(float(wrap_to_pi(float(pose_yaw_rad) - float(predicted_yaw_rad))))
            )
            temporal_score = self._temporal_score(result.pose_4x4, predicted_pose_4x4)
            valid_bonus = (
                float(self.config.deep_pose_hypothesis_valid_bonus)
                if self._is_valid_icp(result)
                else -float(self.config.invalid_icp_penalty)
            )
            hypothesis_score = (
                valid_bonus
                + float(self.config.bev_score_weight) * float(hypothesis_bev_score)
                + float(self.config.icp_inlier_weight) * float(result.inlier_ratio)
                - float(self.config.icp_rmse_weight) * float(result.rmse if np.isfinite(result.rmse) else 10.0)
                + float(self.config.temporal_weight) * float(temporal_score)
                - float(self.config.deep_pose_init_xy_consistency_weight) * float(xy_error_m)
                - float(self.config.deep_pose_init_yaw_consistency_weight) * float(yaw_error_deg)
            )
            scored_hypotheses.append((source, result, float(temporal_score), float(hypothesis_score)))

        tracker_deep_prob_min = self.config.tracker_pose_init_candidate_deep_prob_min
        tracker_required_margin = float(self.config.tracker_pose_init_required_margin_when_deep_low)
        if tracker_deep_prob_min is not None and float(candidate_deep_match_probability) < float(tracker_deep_prob_min):
            best_non_tracker_score = max(
                (
                    score
                    for source, _, _, score in scored_hypotheses
                    if source != "tracker_init"
                ),
                default=-1.0e18,
            )
            filtered_hypotheses: list[tuple[str, object, float, float]] = []
            for source, result, temporal_score, hypothesis_score in scored_hypotheses:
                if (
                    source == "tracker_init"
                    and hypothesis_score < best_non_tracker_score + tracker_required_margin
                ):
                    continue
                filtered_hypotheses.append((source, result, temporal_score, hypothesis_score))
            if filtered_hypotheses:
                scored_hypotheses = filtered_hypotheses

        tracker_deep_prob_high = self.config.tracker_pose_init_candidate_deep_prob_high
        tracker_required_margin_high = float(self.config.tracker_pose_init_required_margin_when_deep_high)
        if tracker_deep_prob_high is not None and float(candidate_deep_match_probability) >= float(tracker_deep_prob_high):
            best_non_tracker_score = max(
                (
                    score
                    for source, _, _, score in scored_hypotheses
                    if source != "tracker_init"
                ),
                default=-1.0e18,
            )
            filtered_hypotheses: list[tuple[str, object, float, float]] = []
            for source, result, temporal_score, hypothesis_score in scored_hypotheses:
                if (
                    source == "tracker_init"
                    and hypothesis_score < best_non_tracker_score + tracker_required_margin_high
                ):
                    continue
                filtered_hypotheses.append((source, result, temporal_score, hypothesis_score))
                if filtered_hypotheses:
                    scored_hypotheses = filtered_hypotheses

        tracker_release_streak_min = self.config.tracker_pose_init_release_streak_min
        tracker_release_deep_prob_min = self.config.tracker_pose_init_release_candidate_deep_prob_min
        tracker_release_required_margin = float(self.config.tracker_pose_init_release_required_margin)
        if (
            tracker_release_streak_min is not None
            and tracker_selected_streak >= int(tracker_release_streak_min)
            and (
                tracker_release_deep_prob_min is None
                or float(candidate_deep_match_probability) >= float(tracker_release_deep_prob_min)
            )
        ):
            best_non_tracker_score = max(
                (
                    score
                    for source, _, _, score in scored_hypotheses
                    if source != "tracker_init"
                ),
                default=-1.0e18,
            )
            filtered_hypotheses: list[tuple[str, object, float, float]] = []
            for source, result, temporal_score, hypothesis_score in scored_hypotheses:
                if (
                    source == "tracker_init"
                    and hypothesis_score < best_non_tracker_score + tracker_release_required_margin
                ):
                    continue
                filtered_hypotheses.append((source, result, temporal_score, hypothesis_score))
            if filtered_hypotheses:
                scored_hypotheses = filtered_hypotheses

        sticky_patch_streak_min = self.config.tracker_pose_init_sticky_patch_streak_min
        sticky_prev_deep_prob_max = self.config.tracker_pose_init_sticky_prev_deep_prob_max
        sticky_candidate_deep_prob_min = self.config.tracker_pose_init_sticky_candidate_deep_prob_min
        sticky_required_margin = float(self.config.tracker_pose_init_sticky_required_margin)
        if (
            sticky_patch_streak_min is not None
            and previous_selected_init_source == "tracker_init"
            and previous_selected_patch_streak >= int(sticky_patch_streak_min)
            and previous_selected_patch_id is not None
            and int(patch_id) == int(previous_selected_patch_id)
            and (
                sticky_prev_deep_prob_max is None
                or (
                    previous_selected_patch_deep_prob is not None
                    and float(previous_selected_patch_deep_prob) <= float(sticky_prev_deep_prob_max)
                )
            )
            and (
                sticky_candidate_deep_prob_min is None
                or float(candidate_deep_match_probability) >= float(sticky_candidate_deep_prob_min)
            )
        ):
            best_non_tracker_score = max(
                (
                    score
                    for source, _, _, score in scored_hypotheses
                    if source != "tracker_init"
                ),
                default=-1.0e18,
            )
            filtered_hypotheses: list[tuple[str, object, float, float]] = []
            for source, result, temporal_score, hypothesis_score in scored_hypotheses:
                if (
                    source == "tracker_init"
                    and hypothesis_score < best_non_tracker_score + sticky_required_margin
                ):
                    continue
                filtered_hypotheses.append((source, result, temporal_score, hypothesis_score))
            if filtered_hypotheses:
                scored_hypotheses = filtered_hypotheses

        best_score = -1.0e18
        for source, result, temporal_score, hypothesis_score in scored_hypotheses:
            if hypothesis_score > best_score:
                best_score = hypothesis_score
                best_source = source
                best_result = result
                best_temporal = temporal_score
        occlusion_threshold = self.config.deep_pose_init_semantic_occlusion_ratio_threshold
        if (
            best_source == "deep_init"
            and occlusion_threshold is not None
            and float(getattr(self, "_active_semantic_occlusion_ratio", 0.0)) >= float(occlusion_threshold)
        ):
            tracker_hypothesis = next(
                (
                    (result, temporal_score)
                    for source, result, temporal_score, _ in scored_hypotheses
                    if source == "tracker_init"
                ),
                None,
            )
            if tracker_hypothesis is not None:
                best_source = "deep_score_tracker_pose"
                best_result, best_temporal = tracker_hypothesis
        return best_source, best_result, float(best_temporal)

    def _predict_candidate_pose(
        self,
        query_points: np.ndarray,
        candidate_bevs: np.ndarray,
        candidate_centers_xy: np.ndarray,
    ) -> dict[str, np.ndarray]:
        query_bev = points_to_bev(query_points, self.query_bev_config)
        if self.zero_query_bev:
            query_bev = np.zeros_like(query_bev, dtype=np.float32)
        query_image_tensor = None
        query_image_right_tensor = None
        query_depth_tensor = None
        if self.deep_model.use_query_image:
            image_left, _ = load_rgb_image(
                self.sequence_dataset.frame_index[self._active_frame_idx].image_left_path,
                camera_model=self.sequence_dataset.camera_left,
                use_undistort=False,
                resize_hw=self.image_resize_hw,
            )
            image_left = np.asarray(image_left, dtype=np.float32) / 255.0
            image_left = np.transpose(image_left, (2, 0, 1)).astype(np.float32, copy=False)
            query_image_tensor = torch.from_numpy(image_left[None]).to(self.device).float()
            if self.deep_model.use_stereo_query_image:
                image_right, _ = load_rgb_image(
                    self.sequence_dataset.frame_index[self._active_frame_idx].image_right_path,
                    camera_model=self.sequence_dataset.camera_right,
                    use_undistort=False,
                    resize_hw=self.image_resize_hw,
                )
                image_right = np.asarray(image_right, dtype=np.float32) / 255.0
                image_right = np.transpose(image_right, (2, 0, 1)).astype(np.float32, copy=False)
                query_image_right_tensor = torch.from_numpy(image_right[None]).to(self.device).float()
        if self.deep_model.use_query_depth:
            depth_left = load_depth_png(
                self.sequence_dataset.frame_index[self._active_frame_idx].depth_left_path,
                depth_scale=None,
                resize_hw=self.image_resize_hw,
            )
            depth_features = build_stereo_geometry_features(
                depth_raw=depth_left,
                fx_px=self.query_fx_px,
                baseline_m=self.stereo_baseline_m,
                depth_scale=self.depth_scale,
                depth_min_m=self.depth_min_m,
                depth_max_m=self.depth_max_m,
                disparity_normalizer_px=float(self.image_resize_hw[1]),
            )
            query_depth_tensor = torch.from_numpy(depth_features[None]).to(self.device).float()
        with torch.no_grad():
            outputs = self.deep_model(
                torch.from_numpy(query_bev[None]).to(self.device).float(),
                torch.from_numpy(candidate_bevs[None]).to(self.device).float(),
                query_image=query_image_tensor,
                query_image_right=query_image_right_tensor,
                query_depth_features=query_depth_tensor,
            )
        pose = outputs["pose"][0].detach().cpu().numpy().astype(np.float64)
        pair_embedding = outputs["pair_embedding"][0].detach().cpu().numpy().astype(np.float32)
        match_probability = torch.softmax(outputs["match_logit"], dim=1)[0].detach().cpu().numpy().astype(np.float64)
        pose_confidence = outputs["pose_confidence"][0].detach().cpu().numpy().astype(np.float64)
        world_xy = candidate_centers_xy.astype(np.float64) + pose[:, :2]
        yaw_rad = pose[:, 2]
        return {
            "match_probability": match_probability,
            "pose_confidence": pose_confidence,
            "pair_embedding": pair_embedding,
            "predicted_world_xy": world_xy,
            "predicted_yaw_deg": np.degrees(yaw_rad).astype(np.float64),
            "predicted_pose_delta": pose,
        }

    def _build_frame_candidate_results(
        self,
        frame_idx: int,
        predicted_pose_4x4: np.ndarray | None = None,
        tracker_pose_init_4x4: np.ndarray | None = None,
        tracker_selected_streak: int = 0,
        previous_selected_patch_streak: int = 0,
        previous_selected_patch_id: int | None = None,
        previous_selected_patch_deep_prob: float | None = None,
        previous_selected_init_source: str | None = None,
        apply_semantic_filter: bool = True,
    ) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
        self._active_frame_idx = int(frame_idx)
        self._active_semantic_occlusion_ratio_source = "disabled"
        self._active_semantic_occlusion_ratio = 0.0
        if self.config.deep_pose_init_semantic_occlusion_ratio_threshold is not None:
            predicted_occlusion_ratio = self._predict_semantic_occlusion_ratio_from_images(frame_idx)
            if predicted_occlusion_ratio is not None:
                self._active_semantic_occlusion_ratio = predicted_occlusion_ratio
                self._active_semantic_occlusion_ratio_source = "image_predictor"
            else:
                self._active_semantic_occlusion_ratio = self._semantic_occlusion_ratio(frame_idx)
                self._active_semantic_occlusion_ratio_source = "semantic_label"
        semantic_gate_features = (
            self._reliability_gate_semantic_features(frame_idx)
            if self.reliability_gate is not None
            else np.zeros(3, dtype=np.float32)
        )
        query_points = self._build_query_points(
            frame_idx,
            apply_semantic_filter=apply_semantic_filter,
        )
        query_points_for_deep_model = self._build_query_points(
            frame_idx,
            apply_semantic_filter=False,
        )
        candidates = self._retrieve_topk_candidates(frame_idx)
        candidate_bevs = []
        candidate_centers_xy = []
        submaps = []
        for candidate in candidates:
            patch_meta = candidate["metadata"]
            submap = build_local_submap(
                self.map_xyz,
                center_xy=patch_meta.center_xy,
                size_m=self.config.local_submap_size_m,
                resolution=self.config.local_submap_resolution,
            )
            submaps.append(submap)
            candidate_bevs.append(np.asarray(submap.bev, dtype=np.float32))
            candidate_centers_xy.append(np.asarray(patch_meta.center_xy, dtype=np.float32))
        deep_prediction = self._predict_candidate_pose(
            query_points_for_deep_model,
            np.stack(candidate_bevs, axis=0).astype(np.float32, copy=False),
            np.stack(candidate_centers_xy, axis=0).astype(np.float32, copy=False),
        )
        candidate_results: list[dict[str, object]] = []
        reranker_logits = None
        if self.candidate_reranker is not None:
            with torch.no_grad():
                reranker_outputs = self.candidate_reranker(
                    torch.from_numpy(deep_prediction["pair_embedding"][None]).to(self.device).float(),
                    torch.from_numpy(
                        np.concatenate(
                            (
                                deep_prediction["match_probability"][:, None].astype(np.float32),
                                deep_prediction["pose_confidence"][:, None].astype(np.float32),
                                deep_prediction["predicted_pose_delta"].astype(np.float32),
                            ),
                            axis=1,
                        )[None]
                    ).to(self.device).float(),
                    torch.tensor(
                        [[int(candidate["patch_id"]) for candidate in candidates]],
                        device=self.device,
                        dtype=torch.long,
                    ),
                )
            reranker_logits = reranker_outputs["logits"][0].detach().cpu().numpy().astype(np.float64)
            reranker_probabilities = torch.softmax(reranker_outputs["logits"], dim=1)[0].detach().cpu().numpy().astype(np.float64)
        else:
            reranker_probabilities = None
        for candidate_idx, candidate in enumerate(candidates):
            submap = submaps[candidate_idx]
            predicted_world_xy = deep_prediction["predicted_world_xy"][candidate_idx]
            predicted_yaw_deg = float(deep_prediction["predicted_yaw_deg"][candidate_idx])
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
            initial_pose_4x4 = pose_from_xy_yaw_z(
                float(bev_match.world_xy[0]),
                float(bev_match.world_xy[1]),
                float(bev_match.yaw_rad),
                z=0.0,
            )
            bev_icp_result = refine_pose_with_icp(
                query_points_xyz_sensor=query_points,
                map_points_xyz_world=submap.points_xyz_world,
                initial_pose_4x4=initial_pose_4x4,
                voxel_size_m=self.config.voxel_size_m,
                max_iterations=self.config.icp_max_iterations,
                max_correspondence_distance_m=self.config.icp_max_correspondence_distance_m,
                min_correspondences=self.config.icp_min_correspondences,
            )
            deep_initial_pose_4x4 = None
            deep_guided_bev_match = None
            deep_icp_result = None
            tracker_icp_result = None
            if bool(self.config.use_deep_pose_init_hypothesis):
                deep_guided_bev_match = match_query_points_to_submap_bev(
                    query_points_xyz=query_points,
                    query_bev_config=self.query_bev_config,
                    submap_bev=submap.bev,
                    submap_bev_config=submap.bev_config,
                    submap_center_xy=submap.center_xy,
                    coarse_yaw_rad=math.radians(predicted_yaw_deg),
                    yaw_half_range_deg=self.config.coarse_yaw_half_range_deg,
                    yaw_step_deg=self.config.coarse_yaw_step_deg,
                    match_channel=self.config.match_channel,
                    coarse_match_method=self.config.coarse_match_method,
                )
                deep_initial_pose_4x4 = pose_from_xy_yaw_z(
                    float(deep_guided_bev_match.world_xy[0]),
                    float(deep_guided_bev_match.world_xy[1]),
                    float(deep_guided_bev_match.yaw_rad),
                    z=0.0,
                )
                deep_icp_result = refine_pose_with_icp(
                    query_points_xyz_sensor=query_points,
                    map_points_xyz_world=submap.points_xyz_world,
                    initial_pose_4x4=deep_initial_pose_4x4,
                    voxel_size_m=self.config.voxel_size_m,
                    max_iterations=self.config.icp_max_iterations,
                    max_correspondence_distance_m=self.config.icp_max_correspondence_distance_m,
                    min_correspondences=self.config.icp_min_correspondences,
                )
            if bool(self.config.use_tracker_pose_init_hypothesis) and tracker_pose_init_4x4 is not None:
                tracker_icp_result = refine_pose_with_icp(
                    query_points_xyz_sensor=query_points,
                    map_points_xyz_world=submap.points_xyz_world,
                    initial_pose_4x4=np.asarray(tracker_pose_init_4x4, dtype=np.float64),
                    voxel_size_m=self.config.voxel_size_m,
                    max_iterations=self.config.icp_max_iterations,
                    max_correspondence_distance_m=self.config.icp_max_correspondence_distance_m,
                    min_correspondences=self.config.icp_min_correspondences,
                )
            selected_init_source, icp_result, temporal_score = self._select_best_pose_hypothesis(
                patch_id=int(candidate["patch_id"]),
                bev_match_score=float(bev_match.score),
                bev_icp_result=bev_icp_result,
                deep_bev_match_score=float(deep_guided_bev_match.score) if deep_guided_bev_match is not None else None,
                deep_pose_4x4=deep_initial_pose_4x4,
                deep_icp_result=deep_icp_result,
                tracker_icp_result=tracker_icp_result,
                candidate_deep_match_probability=float(deep_prediction["match_probability"][candidate_idx]),
                predicted_world_xy=np.asarray(predicted_world_xy, dtype=np.float64),
                predicted_yaw_deg=predicted_yaw_deg,
                predicted_pose_4x4=predicted_pose_4x4,
                tracker_selected_streak=int(tracker_selected_streak),
                previous_selected_patch_streak=int(previous_selected_patch_streak),
                previous_selected_patch_id=(
                    None if previous_selected_patch_id is None else int(previous_selected_patch_id)
                ),
                previous_selected_patch_deep_prob=(
                    None
                    if previous_selected_patch_deep_prob is None
                    else float(previous_selected_patch_deep_prob)
                ),
                previous_selected_init_source=previous_selected_init_source,
            )
            tracker_hypothesis_available = bool(
                predicted_pose_4x4 is not None
                and tracker_icp_result is not None
                and self._tracker_pose_hypothesis_passes_quality_gate(tracker_icp_result)
                and self._tracker_pose_hypothesis_is_consistent(
                    tracker_icp_result,
                    [bev_icp_result, deep_icp_result],
                )
            )
            hypothesis_results = {
                "bev_init": bev_icp_result,
                "deep_init": deep_icp_result,
                "tracker_init": tracker_icp_result if tracker_hypothesis_available else None,
            }
            hypothesis_temporal_scores = {
                source: self._temporal_score(result.pose_4x4, predicted_pose_4x4)
                if result is not None
                else None
                for source, result in hypothesis_results.items()
            }
            gate_features = build_reliability_gate_features(
                deep_match_probability=float(deep_prediction["match_probability"][candidate_idx]),
                deep_pose_confidence=float(deep_prediction["pose_confidence"][candidate_idx]),
                bev_match_score=float(bev_match.score),
                deep_bev_match_score=(
                    float(deep_guided_bev_match.score)
                    if deep_guided_bev_match is not None
                    else None
                ),
                bev_inlier_ratio=float(bev_icp_result.inlier_ratio),
                deep_inlier_ratio=(
                    float(deep_icp_result.inlier_ratio) if deep_icp_result is not None else None
                ),
                tracker_inlier_ratio=(
                    float(tracker_icp_result.inlier_ratio)
                    if tracker_icp_result is not None
                    else None
                ),
                bev_rmse=float(bev_icp_result.rmse),
                deep_rmse=float(deep_icp_result.rmse) if deep_icp_result is not None else None,
                tracker_rmse=(
                    float(tracker_icp_result.rmse)
                    if tracker_icp_result is not None
                    else None
                ),
                bev_temporal_score=float(hypothesis_temporal_scores["bev_init"]),
                deep_temporal_score=hypothesis_temporal_scores["deep_init"],
                tracker_temporal_score=hypothesis_temporal_scores["tracker_init"],
                deep_available=deep_icp_result is not None,
                tracker_available=tracker_hypothesis_available,
                person_ratio=float(semantic_gate_features[0]),
                dynamic_ratio=float(semantic_gate_features[1]),
                movable_ratio=float(semantic_gate_features[2]),
            )
            reliability_gate_probabilities = None
            reliability_gate_source_risks_m = None
            reliability_gate_selected_source = None
            reliability_gate_predicted_improvement_m = None
            reliability_gate_applied = False
            if self.reliability_gate is not None:
                standardized_gate_features = (
                    gate_features - self.reliability_gate_feature_mean
                ) / np.maximum(self.reliability_gate_feature_std, 1.0e-6)
                with torch.no_grad():
                    gate_logits = self.reliability_gate(
                        torch.from_numpy(standardized_gate_features[None]).to(self.device).float()
                    )
                available_source_indices = [
                    source_idx
                    for source_idx, source in enumerate(RELIABILITY_GATE_SOURCES)
                    if hypothesis_results[source] is not None
                ]
                if self.reliability_gate_objective == "risk_regression":
                    reliability_gate_source_risks_m = np.maximum(
                        np.expm1(gate_logits[0].detach().cpu().numpy().astype(np.float64)),
                        0.0,
                    )
                    selected_gate_idx = min(
                        available_source_indices,
                        key=lambda source_idx: float(reliability_gate_source_risks_m[source_idx]),
                    )
                    gate_source = RELIABILITY_GATE_SOURCES[selected_gate_idx]
                    baseline_source = (
                        "tracker_init"
                        if selected_init_source == "deep_score_tracker_pose"
                        else selected_init_source
                    )
                    baseline_idx = RELIABILITY_GATE_SOURCES.index(baseline_source)
                    reliability_gate_predicted_improvement_m = float(
                        reliability_gate_source_risks_m[baseline_idx]
                        - reliability_gate_source_risks_m[selected_gate_idx]
                    )
                    reliability_gate_selected_source = gate_source
                    if (
                        gate_source != baseline_source
                        and reliability_gate_predicted_improvement_m
                        >= float(self.config.reliability_gate_min_predicted_improvement_m)
                    ):
                        selected_init_source = gate_source
                        icp_result = hypothesis_results[gate_source]
                        temporal_score = float(hypothesis_temporal_scores[gate_source])
                        reliability_gate_applied = True
                else:
                    reliability_gate_probabilities = (
                        torch.softmax(gate_logits, dim=1)[0].detach().cpu().numpy().astype(np.float64)
                    )
                    selected_gate_idx = max(
                        available_source_indices,
                        key=lambda source_idx: float(reliability_gate_probabilities[source_idx]),
                    )
                    gate_source = RELIABILITY_GATE_SOURCES[selected_gate_idx]
                    reliability_gate_selected_source = gate_source
                    if float(reliability_gate_probabilities[selected_gate_idx]) >= float(
                        self.config.reliability_gate_confidence_min
                    ):
                        selected_init_source = gate_source
                        icp_result = hypothesis_results[gate_source]
                        temporal_score = float(hypothesis_temporal_scores[gate_source])
                        reliability_gate_applied = True
            candidate_pose = np.asarray(icp_result.pose_4x4, dtype=np.float64)
            temporal_position_jump_m, temporal_yaw_jump_deg = self._temporal_jump_metrics(
                candidate_pose,
                predicted_pose_4x4,
            )
            temporal_gate_valid = self._is_temporal_gate_valid(
                temporal_position_jump_m,
                temporal_yaw_jump_deg,
            )
            selected_bev_score = (
                float(deep_guided_bev_match.score)
                if selected_init_source in {"deep_init", "deep_score_tracker_pose"}
                and deep_guided_bev_match is not None
                else float(bev_match.score)
            )
            candidate_result = {
                "patch_id": int(candidate["patch_id"]),
                "coarse_score": float(candidate["score"]),
                "coarse_query_rotation_deg": float(candidate["query_rotation_deg"]),
                "bev_score": float(bev_match.score),
                "selected_bev_score": float(selected_bev_score),
                "initial_world_xy": [
                    float(bev_match.world_xy[0]),
                    float(bev_match.world_xy[1]),
                ],
                "initial_yaw_deg": float(math.degrees(bev_match.yaw_rad)),
                "deep_pred_world_xy": [
                    float(predicted_world_xy[0]),
                    float(predicted_world_xy[1]),
                ],
                "deep_pred_yaw_deg": predicted_yaw_deg,
                "deep_match_probability": float(deep_prediction["match_probability"][candidate_idx]),
                "deep_pose_confidence": float(deep_prediction["pose_confidence"][candidate_idx]),
                "candidate_reranker_probability": float(reranker_probabilities[candidate_idx]) if reranker_probabilities is not None else None,
                "selected_init_source": selected_init_source,
                "deep_guided_bev_score": float(deep_guided_bev_match.score) if deep_guided_bev_match is not None else None,
                "bev_init_icp_inlier_ratio": float(bev_icp_result.inlier_ratio),
                "bev_init_icp_rmse": float(bev_icp_result.rmse),
                "bev_init_pose_4x4": bev_icp_result.pose_4x4.tolist(),
                "deep_init_icp_inlier_ratio": float(deep_icp_result.inlier_ratio) if deep_icp_result is not None else None,
                "deep_init_icp_rmse": float(deep_icp_result.rmse) if deep_icp_result is not None else None,
                "deep_init_pose_4x4": deep_icp_result.pose_4x4.tolist() if deep_icp_result is not None else None,
                "tracker_init_icp_inlier_ratio": float(tracker_icp_result.inlier_ratio) if tracker_icp_result is not None else None,
                "tracker_init_icp_rmse": float(tracker_icp_result.rmse) if tracker_icp_result is not None else None,
                "tracker_init_pose_4x4": tracker_icp_result.pose_4x4.tolist() if tracker_icp_result is not None else None,
                "icp_inlier_ratio": float(icp_result.inlier_ratio),
                "icp_rmse": float(icp_result.rmse),
                "icp_valid": self._is_valid_icp(icp_result),
                "temporal_position_jump_m": temporal_position_jump_m,
                "temporal_yaw_jump_deg": temporal_yaw_jump_deg,
                "temporal_gate_valid": temporal_gate_valid,
                "temporal_score": float(temporal_score),
                "reliability_gate_features": gate_features.tolist(),
                "reliability_gate_available_sources": [
                    source for source in RELIABILITY_GATE_SOURCES if hypothesis_results[source] is not None
                ],
                "reliability_gate_probabilities": (
                    reliability_gate_probabilities.tolist()
                    if reliability_gate_probabilities is not None
                    else None
                ),
                "reliability_gate_source_risks_m": (
                    reliability_gate_source_risks_m.tolist()
                    if reliability_gate_source_risks_m is not None
                    else None
                ),
                "reliability_gate_selected_source": reliability_gate_selected_source,
                "reliability_gate_predicted_improvement_m": reliability_gate_predicted_improvement_m,
                "reliability_gate_applied": reliability_gate_applied,
                "final_pose_4x4": icp_result.pose_4x4.tolist(),
            }
            candidate_result["final_score"] = float(
                self._candidate_unary_score(candidate_result)
                + float(self.config.temporal_weight) * float(temporal_score)
                + (
                    float(self.config.candidate_reranker_score_weight) * float(reranker_probabilities[candidate_idx])
                    if reranker_probabilities is not None
                    else 0.0
                )
            )
            candidate_results.append(candidate_result)
        if self.confusion_pair_resolver is not None and self.confusion_pair_to_id:
            patch_to_candidate_idx = {int(item["patch_id"]): idx for idx, item in enumerate(candidate_results)}
            top2_allowed_pairs: set[tuple[int, int]] | None = None
            top2_score_gap = None
            if bool(self.config.confusion_pair_resolver_top2_only) and len(candidate_results) >= 2:
                sorted_indices = sorted(
                    range(len(candidate_results)),
                    key=lambda idx: float(candidate_results[idx]["final_score"]),
                    reverse=True,
                )
                first_idx = sorted_indices[0]
                second_idx = sorted_indices[1]
                first_patch = int(candidate_results[first_idx]["patch_id"])
                second_patch = int(candidate_results[second_idx]["patch_id"])
                top2_allowed_pairs = {tuple(sorted((first_patch, second_patch)))}
                top2_score_gap = float(candidate_results[first_idx]["final_score"]) - float(candidate_results[second_idx]["final_score"])
                top1_candidate = candidate_results[first_idx]
                skip_top1_bev = self.config.confusion_pair_resolver_skip_if_top1_bev_score_ge
                skip_top1_inlier = self.config.confusion_pair_resolver_skip_if_top1_inlier_ge
                if skip_top1_bev is not None and skip_top1_inlier is not None:
                    if (
                        float(top1_candidate["selected_bev_score"]) >= float(skip_top1_bev)
                        and float(top1_candidate["icp_inlier_ratio"]) >= float(skip_top1_inlier)
                    ):
                        top2_allowed_pairs = set()
            with torch.no_grad():
                for pair, pair_id in self.confusion_pair_to_id.items():
                    if top2_allowed_pairs is not None:
                        if pair not in top2_allowed_pairs:
                            continue
                        if top2_score_gap is not None and top2_score_gap > float(self.config.confusion_pair_resolver_final_gap_max):
                            continue
                    patch_a, patch_b = pair
                    if patch_a not in patch_to_candidate_idx or patch_b not in patch_to_candidate_idx:
                        continue
                    idx_a = patch_to_candidate_idx[patch_a]
                    idx_b = patch_to_candidate_idx[patch_b]
                    scalar_a = np.concatenate(
                        (
                            np.asarray([deep_prediction["match_probability"][idx_a]], dtype=np.float32),
                            np.asarray([deep_prediction["pose_confidence"][idx_a]], dtype=np.float32),
                            deep_prediction["predicted_pose_delta"][idx_a].astype(np.float32),
                        ),
                        axis=0,
                    )
                    scalar_b = np.concatenate(
                        (
                            np.asarray([deep_prediction["match_probability"][idx_b]], dtype=np.float32),
                            np.asarray([deep_prediction["pose_confidence"][idx_b]], dtype=np.float32),
                            deep_prediction["predicted_pose_delta"][idx_b].astype(np.float32),
                        ),
                        axis=0,
                    )
                    pair_outputs = self.confusion_pair_resolver(
                        torch.tensor([pair_id], device=self.device, dtype=torch.long),
                        torch.from_numpy(deep_prediction["pair_embedding"][idx_a][None]).to(self.device).float(),
                        torch.from_numpy(scalar_a[None]).to(self.device).float(),
                        torch.from_numpy(deep_prediction["pair_embedding"][idx_b][None]).to(self.device).float(),
                        torch.from_numpy(scalar_b[None]).to(self.device).float(),
                    )
                    pair_prob = torch.softmax(pair_outputs["logits"], dim=1)[0].detach().cpu().numpy().astype(np.float64)
                    winner_idx = idx_a if float(pair_prob[0]) >= float(pair_prob[1]) else idx_b
                    loser_idx = idx_b if winner_idx == idx_a else idx_a
                    winner_deep_match = float(deep_prediction["match_probability"][winner_idx])
                    loser_deep_match = float(deep_prediction["match_probability"][loser_idx])
                    deep_match_gap_limit = self.config.confusion_pair_resolver_skip_if_winner_deep_match_lower_by
                    if (
                        deep_match_gap_limit is not None
                        and winner_deep_match + float(deep_match_gap_limit) < loser_deep_match
                    ):
                        continue
                    pair_weight = float(
                        self.confusion_pair_weight_overrides.get(
                            pair,
                            self.config.confusion_pair_resolver_score_weight,
                        )
                    )
                    candidate_results[idx_a]["confusion_pair_probability"] = float(pair_prob[0])
                    candidate_results[idx_b]["confusion_pair_probability"] = float(pair_prob[1])
                    candidate_results[idx_a]["final_score"] = float(
                        candidate_results[idx_a]["final_score"]
                        + pair_weight * (float(pair_prob[0]) - 0.5)
                    )
                    candidate_results[idx_b]["final_score"] = float(
                        candidate_results[idx_b]["final_score"]
                        + pair_weight * (float(pair_prob[1]) - 0.5)
                    )
        return candidate_results, deep_prediction

    def localize_frame(
        self,
        frame_idx: int,
        predicted_pose_4x4: np.ndarray | None = None,
        tracker_pose_init_4x4: np.ndarray | None = None,
        tracker_selected_streak: int = 0,
        previous_selected_patch_streak: int = 0,
        previous_selected_patch_id: int | None = None,
        previous_selected_patch_deep_prob: float | None = None,
        previous_selected_init_source: str | None = None,
        force_raw_geometry: bool = False,
    ) -> dict[str, object]:
        candidate_results, _ = self._build_frame_candidate_results(
            frame_idx=frame_idx,
            predicted_pose_4x4=predicted_pose_4x4,
            tracker_pose_init_4x4=tracker_pose_init_4x4,
            tracker_selected_streak=int(tracker_selected_streak),
            previous_selected_patch_streak=int(previous_selected_patch_streak),
            previous_selected_patch_id=(
                None if previous_selected_patch_id is None else int(previous_selected_patch_id)
            ),
            previous_selected_patch_deep_prob=previous_selected_patch_deep_prob,
            previous_selected_init_source=previous_selected_init_source,
            apply_semantic_filter=not bool(force_raw_geometry),
        )
        self._active_semantic_filter_accepted_candidate_count = 0
        decision = self._semantic_filter_decision_cache.get(int(frame_idx))
        if (
            not bool(force_raw_geometry)
            and bool(self.config.use_semantic_dual_geometry_gate)
            and decision is not None
            and decision.use_conservative_filter
        ):
            raw_candidate_results, _ = self._build_frame_candidate_results(
                frame_idx=frame_idx,
                predicted_pose_4x4=predicted_pose_4x4,
                tracker_pose_init_4x4=tracker_pose_init_4x4,
                tracker_selected_streak=int(tracker_selected_streak),
                previous_selected_patch_streak=int(previous_selected_patch_streak),
                previous_selected_patch_id=(
                    None if previous_selected_patch_id is None else int(previous_selected_patch_id)
                ),
                previous_selected_patch_deep_prob=previous_selected_patch_deep_prob,
                previous_selected_init_source=previous_selected_init_source,
                apply_semantic_filter=False,
            )
            candidate_results, self._active_semantic_filter_accepted_candidate_count = (
                self._merge_semantic_geometry_hypotheses(
                    candidate_results,
                    raw_candidate_results,
                )
            )
        candidate_results.sort(key=lambda item: float(item["final_score"]), reverse=True)
        best_candidate = candidate_results[0]
        gt_pose = self.sequence_dataset.ground_truth.poses_4x4[frame_idx]
        pred_pose = np.asarray(best_candidate["final_pose_4x4"], dtype=np.float64)
        position_error_m = float(np.linalg.norm(pred_pose[:2, 3] - gt_pose[:2, 3]))
        yaw_error_deg = abs(
            math.degrees(float(wrap_to_pi(yaw_from_pose_matrix(pred_pose) - yaw_from_pose_matrix(gt_pose))))
        )
        result = {
            "frame_idx": int(frame_idx),
            "timestamp": float(self.sequence_dataset.frame_index[frame_idx].timestamp),
            "semantic_occlusion_ratio": float(self._active_semantic_occlusion_ratio),
            "semantic_occlusion_ratio_source": str(self._active_semantic_occlusion_ratio_source),
            **self._semantic_filter_metadata(frame_idx),
            "semantic_filter_accepted_candidate_count": int(
                getattr(self, "_active_semantic_filter_accepted_candidate_count", 0)
            ),
            "best_patch_id": int(best_candidate["patch_id"]),
            "pred_pose_4x4": best_candidate["final_pose_4x4"],
            "predicted_pose_4x4_from_tracker": tracker_pose_init_4x4.tolist() if tracker_pose_init_4x4 is not None else None,
            "predicted_pose_4x4_temporal_prior": predicted_pose_4x4.tolist() if predicted_pose_4x4 is not None else None,
            "position_error_m": position_error_m,
            "yaw_error_deg": float(yaw_error_deg),
            "used_tracker_fallback": False,
            "selection_mode": "online_best",
            "selected_candidate_index": 0,
            "candidate_results": candidate_results,
        }
        if bool(self.config.use_candidate_reranker_top2_override) and len(candidate_results) >= 2:
            reranked = sorted(
                enumerate(candidate_results),
                key=lambda item: float(item[1].get("candidate_reranker_probability") or 0.0),
                reverse=True,
            )
            reranker_best_idx, reranker_best = reranked[0]
            reranker_second_prob = float(reranked[1][1].get("candidate_reranker_probability") or 0.0)
            reranker_best_prob = float(reranker_best.get("candidate_reranker_probability") or 0.0)
            reranker_margin = reranker_best_prob - reranker_second_prob
            final_gap = float(candidate_results[0]["final_score"]) - float(candidate_results[1]["final_score"])
            top2_patch_pair = (
                int(candidate_results[0]["patch_id"]),
                int(candidate_results[1]["patch_id"]),
            )
            top2_patch_ids = set(top2_patch_pair)
            allowed_override_pairs = {
                frozenset((int(pair[0]), int(pair[1])))
                for pair in getattr(self.config, "candidate_reranker_top2_pairs", ())
            }
            pair_allowed = (
                not allowed_override_pairs
                or frozenset(top2_patch_pair) in allowed_override_pairs
            )
            if (
                reranker_best_idx in (0, 1)
                and int(reranker_best["patch_id"]) in top2_patch_ids
                and reranker_best_prob >= float(self.config.candidate_reranker_top2_probability_min)
                and reranker_margin >= float(self.config.candidate_reranker_top2_margin_min)
                and final_gap <= float(self.config.candidate_reranker_top2_final_gap_max)
                and pair_allowed
                and reranker_best_idx != 0
            ):
                rerank_pose = np.asarray(reranker_best["final_pose_4x4"], dtype=np.float64)
                rerank_position_error_m = float(np.linalg.norm(rerank_pose[:2, 3] - gt_pose[:2, 3]))
                rerank_yaw_error_deg = abs(
                    math.degrees(float(wrap_to_pi(yaw_from_pose_matrix(rerank_pose) - yaw_from_pose_matrix(gt_pose))))
                )
                result["online_best_patch_id"] = int(result["best_patch_id"])
                result["online_pred_pose_4x4"] = result["pred_pose_4x4"]
                result["best_patch_id"] = int(reranker_best["patch_id"])
                result["pred_pose_4x4"] = reranker_best["final_pose_4x4"]
                result["position_error_m"] = rerank_position_error_m
                result["yaw_error_deg"] = float(rerank_yaw_error_deg)
                result["selection_mode"] = "candidate_reranker_top2_override"
                result["selected_candidate_index"] = int(reranker_best_idx)
                result["candidate_reranker_top2_override"] = {
                    "reranker_best_probability": reranker_best_prob,
                    "reranker_margin": reranker_margin,
                    "final_score_gap": final_gap,
                    "top2_patch_pair": list(top2_patch_pair),
                }
        result = self._apply_top2_confusion_geometry_override(result)
        return self._apply_online_pose_stabilizer(result, predicted_pose_4x4)


def localize_sequence(
    config,
    frame_start: int = 0,
    num_frames: int | None = None,
    frame_stride: int = 1,
    output_json: str | Path | None = None,
    save_every: int = 25,
    resume: bool = True,
) -> dict[str, object]:
    localizer = DeepFineLocalizer(config)
    if bool(localizer.config.use_multi_hypothesis_tracker):
        return _localize_sequence_multi_hypothesis(
            localizer,
            frame_start=frame_start,
            num_frames=num_frames,
            frame_stride=frame_stride,
            output_json=output_json,
            save_every=save_every,
        )
    last_frame = len(localizer.sequence_dataset) if num_frames is None else min(
        len(localizer.sequence_dataset),
        int(frame_start) + int(num_frames),
    )
    frame_indices = list(range(int(frame_start), last_frame, max(1, int(frame_stride))))
    output_path = str(output_json) if output_json is not None else localizer.config.output_json
    resume_state_path = (
        Path(f"{output_path}.resume_state.npz") if output_path is not None else None
    )
    frame_results: list[dict[str, object]] = []
    accepted_poses: list[np.ndarray] = []
    raw_tracker_results: list[dict[str, object]] = []
    raw_tracker_poses: list[np.ndarray] = []
    use_semantic_shadow_tracker = bool(localizer.config.use_semantic_shadow_tracker)

    def write_resume_state() -> None:
        if resume_state_path is None:
            return
        resume_state_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            resume_state_path,
            frame_indices=np.asarray([item["frame_idx"] for item in frame_results], dtype=np.int64),
            accepted_poses=np.asarray(accepted_poses, dtype=np.float64),
        )

    def _current_tracker_selected_streak(accepted_frame_results: list[dict[str, object]]) -> int:
        streak = 0
        for item in reversed(accepted_frame_results):
            candidate_results = list(item.get("candidate_results", []))
            if not candidate_results:
                break
            selected_idx = item.get("selected_candidate_index")
            if selected_idx is None or int(selected_idx) < 0 or int(selected_idx) >= len(candidate_results):
                break
            selected_candidate = candidate_results[int(selected_idx)]
            if selected_candidate.get("selected_init_source") != "tracker_init":
                break
            streak += 1
        return int(streak)

    def _current_selected_patch_context(
        accepted_frame_results: list[dict[str, object]],
    ) -> tuple[int, int | None, float | None, str | None]:
        if not accepted_frame_results:
            return 0, None, None, None
        last_item = accepted_frame_results[-1]
        candidate_results = list(last_item.get("candidate_results", []))
        selected_idx = last_item.get("selected_candidate_index")
        if selected_idx is None or int(selected_idx) < 0 or int(selected_idx) >= len(candidate_results):
            return 0, None, None, None
        selected_candidate = candidate_results[int(selected_idx)]
        selected_patch_id = int(selected_candidate.get("patch_id", -1))
        streak = 0
        for item in reversed(accepted_frame_results):
            current_candidates = list(item.get("candidate_results", []))
            current_idx = item.get("selected_candidate_index")
            if current_idx is None or int(current_idx) < 0 or int(current_idx) >= len(current_candidates):
                break
            current_candidate = current_candidates[int(current_idx)]
            if int(current_candidate.get("patch_id", -1)) != selected_patch_id:
                break
            streak += 1
        return (
            int(streak),
            int(selected_patch_id),
            None
            if selected_candidate.get("deep_match_probability") is None
            else float(selected_candidate["deep_match_probability"]),
            selected_candidate.get("selected_init_source"),
        )

    start_offset = 0
    if output_path is not None and bool(resume):
        existing_path = Path(output_path)
        if existing_path.is_file():
            existing_report = json.loads(existing_path.read_text(encoding="utf-8"))
            existing_results = existing_report.get("frame_results", [])
            if isinstance(existing_results, list):
                valid_results = [
                    item
                    for item in existing_results
                    if int(item.get("frame_idx", -1)) in frame_indices
                ]
                valid_results.sort(key=lambda item: int(item["frame_idx"]))
                frame_results = valid_results
                accepted_poses = []
                if resume_state_path is not None and resume_state_path.is_file():
                    with np.load(resume_state_path) as state:
                        state_indices = np.asarray(state["frame_indices"], dtype=np.int64)
                        state_poses = np.asarray(state["accepted_poses"], dtype=np.float64)
                    expected_indices = np.asarray(
                        [item["frame_idx"] for item in valid_results],
                        dtype=np.int64,
                    )
                    if (
                        state_indices.shape == expected_indices.shape
                        and np.array_equal(state_indices, expected_indices)
                        and state_poses.shape == (len(valid_results), 4, 4)
                    ):
                        accepted_poses = [pose.copy() for pose in state_poses]
                if not accepted_poses:
                    accepted_poses = [
                        np.asarray(item["pred_pose_4x4"], dtype=np.float64)
                        for item in valid_results
                    ]
                start_offset = len(valid_results)
                if use_semantic_shadow_tracker:
                    raw_tracker_results = valid_results
                    raw_tracker_poses = [pose.copy() for pose in accepted_poses]
    for list_idx, frame_idx in enumerate(frame_indices[start_offset:], start=start_offset):
        tracker_poses = raw_tracker_poses if use_semantic_shadow_tracker else accepted_poses
        tracker_results = raw_tracker_results if use_semantic_shadow_tracker else frame_results
        temporal_prior_pose_4x4 = localizer._predict_motion_prior_4x4(tracker_poses)
        predicted_pose_4x4 = localizer._predict_pose_4x4(
            tracker_poses,
            accepted_frame_results=tracker_results,
        )
        tracker_selected_streak = _current_tracker_selected_streak(tracker_results)
        previous_selected_patch_streak, previous_selected_patch_id, previous_selected_patch_deep_prob, previous_selected_init_source = (
            _current_selected_patch_context(tracker_results)
        )
        raw_frame_result = localizer.localize_frame(
            frame_idx,
            predicted_pose_4x4=temporal_prior_pose_4x4,
            tracker_pose_init_4x4=predicted_pose_4x4,
            tracker_selected_streak=tracker_selected_streak,
            previous_selected_patch_streak=previous_selected_patch_streak,
            previous_selected_patch_id=previous_selected_patch_id,
            previous_selected_patch_deep_prob=previous_selected_patch_deep_prob,
            previous_selected_init_source=previous_selected_init_source,
            force_raw_geometry=use_semantic_shadow_tracker,
        )
        if bool(localizer.config.apply_online_patch_hysteresis_during_tracking):
            online_results = localizer._apply_online_patch_hysteresis(tracker_results + [raw_frame_result])
            raw_frame_result = online_results[-1]
        frame_result = raw_frame_result
        if use_semantic_shadow_tracker:
            localizer._build_query_points(frame_idx)
            decision = localizer._semantic_filter_decision_cache.get(int(frame_idx))
            if decision is not None and decision.use_conservative_filter:
                refinement = localizer._semantic_sidecar_refine_raw_winner(
                    frame_idx,
                    raw_frame_result,
                )
                if refinement is not None:
                    frame_result = dict(raw_frame_result)
                    semantic_pose = np.asarray(refinement.pose_4x4, dtype=np.float64)
                    raw_pose = np.asarray(raw_frame_result["pred_pose_4x4"], dtype=np.float64)
                    selected_candidate = raw_frame_result["candidate_results"][
                        int(raw_frame_result["selected_candidate_index"])
                    ]
                    ground_truth_pose = localizer.sequence_dataset.ground_truth.poses_4x4[frame_idx]
                    raw_position_error_m = float(
                        np.linalg.norm(raw_pose[:2, 3] - ground_truth_pose[:2, 3])
                    )
                    raw_yaw_error_deg = abs(
                        math.degrees(float(wrap_to_pi(
                            yaw_from_pose_matrix(raw_pose) - yaw_from_pose_matrix(ground_truth_pose)
                        )))
                    )
                    frame_result["pred_pose_4x4"] = semantic_pose.tolist()
                    frame_result["position_error_m"] = float(
                        np.linalg.norm(semantic_pose[:2, 3] - ground_truth_pose[:2, 3])
                    )
                    frame_result["yaw_error_deg"] = float(
                        abs(math.degrees(float(wrap_to_pi(
                            yaw_from_pose_matrix(semantic_pose)
                            - yaw_from_pose_matrix(ground_truth_pose)
                        ))))
                    )
                    frame_result["semantic_sidecar_applied"] = True
                    frame_result["semantic_sidecar_pose_4x4"] = semantic_pose.tolist()
                    frame_result["raw_tracker_position_error_m"] = raw_position_error_m
                    frame_result["raw_tracker_yaw_error_deg"] = float(raw_yaw_error_deg)
                    frame_result["semantic_sidecar_features"] = {
                        "semi_dynamic_point_ratio": float(decision.semi_dynamic_point_ratio),
                        "raw_icp_inlier_ratio": float(selected_candidate["icp_inlier_ratio"]),
                        "raw_icp_rmse": float(selected_candidate["icp_rmse"]),
                        "sidecar_icp_inlier_ratio": float(refinement.inlier_ratio),
                        "sidecar_icp_rmse": float(refinement.rmse),
                        "icp_inlier_ratio_gain": float(
                            refinement.inlier_ratio - float(selected_candidate["icp_inlier_ratio"])
                        ),
                        "icp_rmse_gain": float(
                            float(selected_candidate["icp_rmse"]) - refinement.rmse
                        ),
                        "pose_translation_delta_m": float(
                            np.linalg.norm(semantic_pose[:2, 3] - raw_pose[:2, 3])
                        ),
                        "pose_yaw_delta_deg": float(
                            abs(math.degrees(float(wrap_to_pi(
                                yaw_from_pose_matrix(semantic_pose) - yaw_from_pose_matrix(raw_pose)
                            ))))
                        ),
                        "raw_deep_match_probability": float(
                            selected_candidate["deep_match_probability"]
                        ),
                        "raw_temporal_score": float(selected_candidate["temporal_score"]),
                    }
                    frame_result["semantic_filter_accepted_candidate_count"] = 1
            frame_result.update(localizer._semantic_filter_metadata(frame_idx))
            raw_tracker_results.append(raw_frame_result)
            raw_tracker_poses.append(np.asarray(raw_frame_result["pred_pose_4x4"], dtype=np.float64))
        frame_results.append(frame_result)
        accepted_poses.append(
            np.asarray(
                raw_frame_result["pred_pose_4x4"] if use_semantic_shadow_tracker else frame_result["pred_pose_4x4"],
                dtype=np.float64,
            )
        )
        if output_path is not None and int(save_every) > 0 and ((list_idx + 1) % int(save_every) == 0):
            partial_report = _build_localization_report(
                frame_results,
                frame_start=frame_start,
                frame_stride=frame_stride,
            )
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(partial_report, indent=2), encoding="utf-8")
            write_resume_state()
    if not bool(localizer.config.apply_online_patch_hysteresis_during_tracking):
        frame_results = localizer._apply_online_patch_hysteresis(frame_results)
    frame_results = localizer._apply_persistent_patch_override(frame_results)
    frame_results = localizer._apply_sequence_smoothing(frame_results)
    report = _build_localization_report(
        frame_results,
        frame_start=frame_start,
        frame_stride=frame_stride,
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_resume_state()
    print(json.dumps(report, indent=2))
    return report


def _localize_sequence_multi_hypothesis(
    localizer: DeepFineLocalizer,
    *,
    frame_start: int,
    num_frames: int | None,
    frame_stride: int,
    output_json: str | Path | None,
    save_every: int,
) -> dict[str, object]:
    """Causal beam tracker over independently refined candidate trajectories."""
    if bool(localizer.config.use_semantic_shadow_tracker):
        raise ValueError("Multi-hypothesis tracking is not compatible with semantic shadow tracking.")

    last_frame = len(localizer.sequence_dataset) if num_frames is None else min(
        len(localizer.sequence_dataset),
        int(frame_start) + int(num_frames),
    )
    frame_indices = list(range(int(frame_start), last_frame, max(1, int(frame_stride))))
    beam_size = max(1, int(localizer.config.multi_hypothesis_beam_size))
    branch_factor = max(1, int(localizer.config.multi_hypothesis_branch_factor))
    merge_distance_m = max(0.0, float(localizer.config.multi_hypothesis_pose_merge_distance_m))
    states = [MultiHypothesisState(cumulative_score=0.0)]
    online_results: list[dict[str, object]] = []

    def selected_context(state: MultiHypothesisState) -> tuple[int, int | None, float | None, str | None, int]:
        if not state.frame_results:
            return 0, None, None, None, 0
        last_result = state.frame_results[-1]
        selected_idx = int(last_result["selected_candidate_index"])
        candidates = list(last_result["candidate_results"])
        if selected_idx < 0 or selected_idx >= len(candidates):
            return 0, None, None, None, 0
        selected = candidates[selected_idx]
        patch_id = int(selected["patch_id"])
        patch_streak = 0
        tracker_streak = 0
        for result in reversed(state.frame_results):
            result_candidates = list(result["candidate_results"])
            candidate = result_candidates[int(result["selected_candidate_index"])]
            if int(candidate["patch_id"]) != patch_id:
                break
            patch_streak += 1
            if candidate.get("selected_init_source") == "tracker_init":
                tracker_streak += 1
            else:
                break
        return (
            patch_streak,
            patch_id,
            None if selected.get("deep_match_probability") is None else float(selected["deep_match_probability"]),
            str(selected.get("selected_init_source") or ""),
            tracker_streak,
        )

    def make_frame_result(
        frame_idx: int,
        candidates: list[dict[str, object]],
        selected_index: int,
        parent_rank: int,
        cumulative_score: float,
        pose_override: np.ndarray | None = None,
    ) -> dict[str, object]:
        selected = candidates[selected_index] if selected_index >= 0 else None
        pose = (
            np.asarray(pose_override, dtype=np.float64)
            if pose_override is not None
            else np.asarray(selected["final_pose_4x4"], dtype=np.float64)
        )
        ground_truth = localizer.sequence_dataset.ground_truth.poses_4x4[frame_idx]
        return {
            "frame_idx": int(frame_idx),
            "timestamp": float(localizer.sequence_dataset.frame_index[frame_idx].timestamp),
            "best_patch_id": int(selected["patch_id"]) if selected is not None else -1,
            "pred_pose_4x4": pose.tolist(),
            "position_error_m": float(np.linalg.norm(pose[:2, 3] - ground_truth[:2, 3])),
            "yaw_error_deg": float(abs(math.degrees(float(wrap_to_pi(
                yaw_from_pose_matrix(pose) - yaw_from_pose_matrix(ground_truth)
            ))))),
            "used_tracker_fallback": selected is None,
            "selection_mode": "online_multi_hypothesis" if selected is not None else "online_multi_hypothesis_tracker_fallback",
            "selected_candidate_index": int(selected_index),
            "candidate_results": candidates,
            "multi_hypothesis_parent_rank": int(parent_rank),
            "multi_hypothesis_cumulative_score": float(cumulative_score),
        }

    for sequence_index, frame_idx in enumerate(frame_indices):
        expanded_states: list[MultiHypothesisState] = []
        for parent_rank, state in enumerate(states):
            prior_pose = localizer._predict_motion_prior_4x4(state.poses)
            tracker_pose = localizer._predict_pose_4x4(
                state.poses,
                accepted_frame_results=state.frame_results,
            )
            patch_streak, patch_id, deep_probability, init_source, tracker_streak = selected_context(state)
            candidates, _ = localizer._build_frame_candidate_results(
                frame_idx=frame_idx,
                predicted_pose_4x4=prior_pose,
                tracker_pose_init_4x4=tracker_pose,
                tracker_selected_streak=tracker_streak,
                previous_selected_patch_streak=patch_streak,
                previous_selected_patch_id=patch_id,
                previous_selected_patch_deep_prob=deep_probability,
                previous_selected_init_source=init_source or None,
            )
            candidates.sort(key=lambda item: float(item["final_score"]), reverse=True)
            selected_indices = list(range(min(branch_factor, len(candidates))))
            branch_scores = {
                candidate_idx: float(candidates[candidate_idx]["final_score"])
                for candidate_idx in selected_indices
            }
            if prior_pose is not None and bool(localizer.config.use_online_pose_stabilizer):
                admissible = [
                    candidate_idx
                    for candidate_idx, candidate in enumerate(candidates)
                    if (
                        (jump := localizer._candidate_jump_to_prediction(candidate, prior_pose))[0] is not None
                        and jump[1] is not None
                        and jump[0] <= float(localizer.config.online_stabilizer_max_position_jump_m)
                        and jump[1] <= float(localizer.config.online_stabilizer_max_yaw_jump_deg)
                    )
                ]
                if admissible:
                    admissible.sort(
                        key=lambda candidate_idx: localizer._online_stabilizer_candidate_score(
                            candidates[candidate_idx], prior_pose
                        ),
                        reverse=True,
                    )
                    selected_indices = admissible[:branch_factor]
                    branch_scores = {
                        candidate_idx: localizer._online_stabilizer_candidate_score(
                            candidates[candidate_idx], prior_pose
                        )
                        for candidate_idx in selected_indices
                    }
                elif (
                    bool(localizer.config.online_stabilizer_fallback_to_tracker)
                    and bool(localizer.config.multi_hypothesis_keep_reacquisition_branch)
                ):
                    fallback_score = float(
                        max(float(candidate["final_score"]) for candidate in candidates)
                    )
                    fallback_result = make_frame_result(
                        frame_idx,
                        candidates,
                        -1,
                        parent_rank,
                        state.cumulative_score + fallback_score,
                        pose_override=prior_pose,
                    )
                    expanded_states.append(
                        MultiHypothesisState(
                            cumulative_score=float(state.cumulative_score + fallback_score),
                            poses=state.poses + [np.asarray(prior_pose, dtype=np.float64)],
                            frame_results=state.frame_results + [fallback_result],
                        )
                    )
                elif bool(localizer.config.online_stabilizer_fallback_to_tracker):
                    fallback_result = make_frame_result(
                        frame_idx,
                        candidates,
                        -1,
                        parent_rank,
                        state.cumulative_score,
                        pose_override=prior_pose,
                    )
                    expanded_states.append(
                        MultiHypothesisState(
                            cumulative_score=float(state.cumulative_score),
                            poses=state.poses + [np.asarray(prior_pose, dtype=np.float64)],
                            frame_results=state.frame_results + [fallback_result],
                        )
                    )
                    continue
            for selected_index in selected_indices:
                selected = candidates[selected_index]
                cumulative_score = float(state.cumulative_score + branch_scores[selected_index])
                frame_result = make_frame_result(
                    frame_idx,
                    candidates,
                    selected_index,
                    parent_rank,
                    cumulative_score,
                )
                expanded_states.append(
                    MultiHypothesisState(
                        cumulative_score=cumulative_score,
                        poses=state.poses + [np.asarray(selected["final_pose_4x4"], dtype=np.float64)],
                        frame_results=state.frame_results + [frame_result],
                    )
                )
        states = select_diverse_hypotheses(
            expanded_states,
            beam_size=beam_size,
            pose_merge_distance_m=merge_distance_m,
        )
        if not states:
            raise RuntimeError("Multi-hypothesis tracker pruned every state.")
        online_results.append(dict(states[0].frame_results[-1]))
        if output_json is not None and int(save_every) > 0 and ((sequence_index + 1) % int(save_every) == 0):
            partial = _build_localization_report(
                online_results,
                frame_start=frame_start,
                frame_stride=frame_stride,
            )
            partial["multi_hypothesis"] = {
                "beam_size": beam_size,
                "branch_factor": branch_factor,
                "pose_merge_distance_m": merge_distance_m,
            }
            path = Path(output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    if bool(localizer.config.use_online_patch_hysteresis):
        online_results = localizer._apply_online_patch_hysteresis(online_results)
    report = _build_localization_report(
        online_results,
        frame_start=frame_start,
        frame_stride=frame_stride,
    )
    report["multi_hypothesis"] = {
        "beam_size": beam_size,
        "branch_factor": branch_factor,
        "pose_merge_distance_m": merge_distance_m,
        "causal": True,
    }
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def _build_localization_report(
    frame_results: list[dict[str, object]],
    *,
    frame_start: int,
    frame_stride: int,
) -> dict[str, object]:
    position_errors = np.asarray([item["position_error_m"] for item in frame_results], dtype=np.float64)
    yaw_errors = np.asarray([item["yaw_error_deg"] for item in frame_results], dtype=np.float64)
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep fine localization from coarse retrieval candidates.")
    parser.add_argument("--config", default="configs/fine_localization_oct12_aisle_ccw_deep_v6_guidedbev_online.yaml")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    localize_sequence(
        config=args.config,
        frame_start=args.frame_start,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        output_json=args.output_json,
        save_every=args.save_every,
        resume=not bool(args.no_resume),
    )


if __name__ == "__main__":
    main()
