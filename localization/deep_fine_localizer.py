from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from runtime_compat import ensure_runtime_compatibility

ensure_runtime_compatibility()

import torch

from geometry.yaw_utils import wrap_to_pi, yaw_from_pose_matrix
from localization.fine_localizer import FineLocalizer
from localization.icp_refiner import pose_from_xy_yaw_z, refine_pose_with_icp
from localization.submap_builder import build_local_submap
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
        ).to(self.device)
        self.deep_model.load_state_dict(checkpoint["model"], strict=True)
        self.deep_model.eval()

    def _predict_candidate_pose(
        self,
        query_points: np.ndarray,
        candidate_bev: np.ndarray,
        candidate_center_xy: tuple[float, float],
    ) -> dict[str, float | tuple[float, float] | np.ndarray]:
        query_bev = points_to_bev(query_points, self.query_bev_config)
        with torch.no_grad():
            outputs = self.deep_model(
                torch.from_numpy(query_bev[None]).to(self.device).float(),
                torch.from_numpy(candidate_bev[None]).to(self.device).float(),
            )
        pose = outputs["pose"][0].detach().cpu().numpy().astype(np.float64)
        match_probability = float(torch.sigmoid(outputs["match_logit"])[0].item())
        pose_confidence = float(outputs["pose_confidence"][0].item())
        world_x = float(candidate_center_xy[0] + pose[0])
        world_y = float(candidate_center_xy[1] + pose[1])
        yaw_rad = float(pose[2])
        return {
            "initial_pose_4x4": pose_from_xy_yaw_z(world_x, world_y, yaw_rad, z=0.0),
            "match_probability": match_probability,
            "pose_confidence": pose_confidence,
            "predicted_world_xy": (world_x, world_y),
            "predicted_yaw_deg": float(math.degrees(yaw_rad)),
        }

    def localize_frame(self, frame_idx: int, predicted_pose_4x4: np.ndarray | None = None) -> dict[str, object]:
        query_points = self._build_query_points(frame_idx)
        candidates = self._retrieve_topk_candidates(frame_idx)
        candidate_results: list[dict[str, object]] = []
        for candidate in candidates:
            patch_meta = candidate["metadata"]
            submap = build_local_submap(
                self.map_xyz,
                center_xy=patch_meta.center_xy,
                size_m=self.config.local_submap_size_m,
                resolution=self.config.local_submap_resolution,
            )
            deep_prediction = self._predict_candidate_pose(query_points, submap.bev.astype(np.float32), patch_meta.center_xy)
            icp_result = refine_pose_with_icp(
                query_points_xyz_sensor=query_points,
                map_points_xyz_world=submap.points_xyz_world,
                initial_pose_4x4=deep_prediction["initial_pose_4x4"],
                voxel_size_m=self.config.voxel_size_m,
                max_iterations=self.config.icp_max_iterations,
                max_correspondence_distance_m=self.config.icp_max_correspondence_distance_m,
                min_correspondences=self.config.icp_min_correspondences,
            )
            temporal_score = self._temporal_score(icp_result.pose_4x4, predicted_pose_4x4)
            final_score = (
                float(self.config.retrieval_score_weight) * float(candidate["score"])
                + float(self.config.deep_matcher_score_weight) * float(deep_prediction["match_probability"])
                + float(self.config.deep_pose_confidence_weight) * float(deep_prediction["pose_confidence"])
                + float(self.config.icp_inlier_weight) * float(icp_result.inlier_ratio)
                - float(self.config.icp_rmse_weight) * float(icp_result.rmse if np.isfinite(icp_result.rmse) else 10.0)
                + float(self.config.temporal_weight) * float(temporal_score)
            )
            candidate_results.append(
                {
                    "patch_id": int(candidate["patch_id"]),
                    "coarse_score": float(candidate["score"]),
                    "coarse_query_rotation_deg": float(candidate["query_rotation_deg"]),
                    "bev_score": 0.0,
                    "initial_world_xy": [
                        float(deep_prediction["predicted_world_xy"][0]),
                        float(deep_prediction["predicted_world_xy"][1]),
                    ],
                    "initial_yaw_deg": float(deep_prediction["predicted_yaw_deg"]),
                    "deep_match_probability": float(deep_prediction["match_probability"]),
                    "deep_pose_confidence": float(deep_prediction["pose_confidence"]),
                    "icp_inlier_ratio": float(icp_result.inlier_ratio),
                    "icp_rmse": float(icp_result.rmse),
                    "icp_valid": bool(
                        icp_result.num_inliers >= int(self.config.icp_min_correspondences)
                        and np.isfinite(icp_result.rmse)
                    ),
                    "temporal_score": float(temporal_score),
                    "final_score": float(final_score),
                    "final_pose_4x4": icp_result.pose_4x4.tolist(),
                }
            )
        candidate_results.sort(key=lambda item: float(item["final_score"]), reverse=True)
        best_candidate = candidate_results[0]
        gt_pose = self.sequence_dataset.ground_truth.poses_4x4[frame_idx]
        pred_pose = np.asarray(best_candidate["final_pose_4x4"], dtype=np.float64)
        position_error_m = float(np.linalg.norm(pred_pose[:2, 3] - gt_pose[:2, 3]))
        yaw_error_deg = abs(
            math.degrees(float(wrap_to_pi(yaw_from_pose_matrix(pred_pose) - yaw_from_pose_matrix(gt_pose))))
        )
        return {
            "frame_idx": int(frame_idx),
            "timestamp": float(self.sequence_dataset.frame_index[frame_idx].timestamp),
            "best_patch_id": int(best_candidate["patch_id"]),
            "pred_pose_4x4": best_candidate["final_pose_4x4"],
            "predicted_pose_4x4_from_tracker": predicted_pose_4x4.tolist() if predicted_pose_4x4 is not None else None,
            "position_error_m": position_error_m,
            "yaw_error_deg": float(yaw_error_deg),
            "used_tracker_fallback": False,
            "selection_mode": "online_best",
            "selected_candidate_index": 0,
            "candidate_results": candidate_results,
        }


def localize_sequence(
    config,
    frame_start: int = 0,
    num_frames: int | None = None,
    frame_stride: int = 1,
    output_json: str | Path | None = None,
) -> dict[str, object]:
    localizer = DeepFineLocalizer(config)
    last_frame = len(localizer.sequence_dataset) if num_frames is None else min(
        len(localizer.sequence_dataset),
        int(frame_start) + int(num_frames),
    )
    frame_indices = list(range(int(frame_start), last_frame, max(1, int(frame_stride))))
    frame_results: list[dict[str, object]] = []
    accepted_poses: list[np.ndarray] = []
    for frame_idx in frame_indices:
        predicted_pose_4x4 = localizer._predict_pose_4x4(accepted_poses)
        frame_result = localizer.localize_frame(frame_idx, predicted_pose_4x4=predicted_pose_4x4)
        frame_results.append(frame_result)
        accepted_poses.append(np.asarray(frame_result["pred_pose_4x4"], dtype=np.float64))
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
    output_path = str(output_json) if output_json is not None else localizer.config.output_json
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep fine localization from coarse retrieval candidates.")
    parser.add_argument("--config", default="configs/fine_localization_oct12_aisle_ccw_deep.yaml")
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
