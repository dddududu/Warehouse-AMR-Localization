from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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

    def _build_query_points(self, frame_idx: int) -> np.ndarray:
        lidar_path = self.sequence_dataset.frame_index[frame_idx].lidar_path
        points = load_pcd_xyz(lidar_path)
        return crop_local_lidar_points(points, self.crop_config)

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

    def localize_frame(self, frame_idx: int) -> dict[str, Any]:
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
            final_score = (
                float(self.config.retrieval_score_weight) * float(candidate["score"])
                + float(self.config.bev_score_weight) * float(bev_match.score)
                + float(self.config.icp_inlier_weight) * float(icp_result.inlier_ratio)
                - float(self.config.icp_rmse_weight) * float(icp_result.rmse if np.isfinite(icp_result.rmse) else 10.0)
            )
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
                    final_score=float(final_score),
                    final_pose_4x4=icp_result.pose_4x4.copy(),
                )
            )

        best_candidate = max(candidate_results, key=lambda item: item.final_score)
        gt_pose = self.sequence_dataset.ground_truth.poses_4x4[frame_idx]
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
            "position_error_m": position_error_m,
            "yaw_error_deg": float(math.degrees(abs(yaw_error_rad))),
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
        return result


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
    frame_results = [localizer.localize_frame(frame_idx) for frame_idx in frame_indices]
    position_errors = np.asarray([item["position_error_m"] for item in frame_results], dtype=np.float64)
    yaw_errors = np.asarray([item["yaw_error_deg"] for item in frame_results], dtype=np.float64)
    report = {
        "num_eval_frames": len(frame_results),
        "frame_start": int(frame_start),
        "frame_stride": int(frame_stride),
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
