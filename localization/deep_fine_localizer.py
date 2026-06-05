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
from localization.submap_builder import build_local_submap
from models.candidate_reranker import CandidateReranker
from models.confusion_pair_resolver import ConfusionPairResolver
from models.fine_pose_matcher import FinePoseMatcher
from preprocess.bev_builder import points_to_bev


class DeepFineLocalizer(FineLocalizer):
    def __init__(self, config) -> None:
        super().__init__(config)
        if not self.config.deep_matcher_checkpoint_path:
            raise ValueError("deep_matcher_checkpoint_path must be set.")
        checkpoint = torch.load(self.config.deep_matcher_checkpoint_path, map_location=self.device)
        checkpoint_config = checkpoint.get("config", {})
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

    def _is_valid_icp(self, icp_result) -> bool:
        return bool(
            icp_result.num_inliers >= int(self.config.icp_min_correspondences)
            and np.isfinite(icp_result.rmse)
        )

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
    ) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
        self._active_frame_idx = int(frame_idx)
        self._active_semantic_occlusion_ratio = (
            self._semantic_occlusion_ratio(frame_idx)
            if self.config.deep_pose_init_semantic_occlusion_ratio_threshold is not None
            else 0.0
        )
        query_points = self._build_query_points(frame_idx)
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
            query_points,
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
                "deep_init_icp_inlier_ratio": float(deep_icp_result.inlier_ratio) if deep_icp_result is not None else None,
                "deep_init_icp_rmse": float(deep_icp_result.rmse) if deep_icp_result is not None else None,
                "tracker_init_icp_inlier_ratio": float(tracker_icp_result.inlier_ratio) if tracker_icp_result is not None else None,
                "tracker_init_icp_rmse": float(tracker_icp_result.rmse) if tracker_icp_result is not None else None,
                "icp_inlier_ratio": float(icp_result.inlier_ratio),
                "icp_rmse": float(icp_result.rmse),
                "icp_valid": self._is_valid_icp(icp_result),
                "temporal_position_jump_m": temporal_position_jump_m,
                "temporal_yaw_jump_deg": temporal_yaw_jump_deg,
                "temporal_gate_valid": temporal_gate_valid,
                "temporal_score": float(temporal_score),
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
    last_frame = len(localizer.sequence_dataset) if num_frames is None else min(
        len(localizer.sequence_dataset),
        int(frame_start) + int(num_frames),
    )
    frame_indices = list(range(int(frame_start), last_frame, max(1, int(frame_stride))))
    output_path = str(output_json) if output_json is not None else localizer.config.output_json
    frame_results: list[dict[str, object]] = []
    accepted_poses: list[np.ndarray] = []

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
                accepted_poses = [
                    np.asarray(item["pred_pose_4x4"], dtype=np.float64)
                    for item in valid_results
                ]
                start_offset = len(valid_results)
    for list_idx, frame_idx in enumerate(frame_indices[start_offset:], start=start_offset):
        temporal_prior_pose_4x4 = localizer._predict_motion_prior_4x4(accepted_poses)
        predicted_pose_4x4 = localizer._predict_pose_4x4(
            accepted_poses,
            accepted_frame_results=frame_results,
        )
        tracker_selected_streak = _current_tracker_selected_streak(frame_results)
        previous_selected_patch_streak, previous_selected_patch_id, previous_selected_patch_deep_prob, previous_selected_init_source = (
            _current_selected_patch_context(frame_results)
        )
        frame_result = localizer.localize_frame(
            frame_idx,
            predicted_pose_4x4=temporal_prior_pose_4x4,
            tracker_pose_init_4x4=predicted_pose_4x4,
            tracker_selected_streak=tracker_selected_streak,
            previous_selected_patch_streak=previous_selected_patch_streak,
            previous_selected_patch_id=previous_selected_patch_id,
            previous_selected_patch_deep_prob=previous_selected_patch_deep_prob,
            previous_selected_init_source=previous_selected_init_source,
        )
        if bool(localizer.config.apply_online_patch_hysteresis_during_tracking):
            online_results = localizer._apply_online_patch_hysteresis(frame_results + [frame_result])
            frame_result = online_results[-1]
        frame_results.append(frame_result)
        accepted_poses.append(np.asarray(frame_result["pred_pose_4x4"], dtype=np.float64))
        if output_path is not None and int(save_every) > 0 and ((list_idx + 1) % int(save_every) == 0):
            partial_report = _build_localization_report(
                frame_results,
                frame_start=frame_start,
                frame_stride=frame_stride,
            )
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(partial_report, indent=2), encoding="utf-8")
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
