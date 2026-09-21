from __future__ import annotations

from pathlib import Path

import yaml


DATA_ROOT = Path("D:/TorWIC/TorWIC SLAM Dataset")
CALIBRATION_PATH = (
    DATA_ROOT
    / "Jun. 15, 2022"
    / "Aisle_CCW_Run_1"
    / "Aisle_CCW_Run_1"
    / "calibrations.txt"
)
MAP_PATH = (
    DATA_ROOT
    / "Jun. 15, 2022"
    / "Aisle_CCW_Run_1"
    / "Aisle_CCW_Run_1"
    / "groundtruth_map.ply"
)

SEQUENCES = (
    ("train", "jun15_ccw_run2", "Jun. 15, 2022", "Aisle_CCW_Run_2"),
    ("train", "jun15_cw_run2", "Jun. 15, 2022", "Aisle_CW_Run_2"),
    ("val", "jun23_ccw_run2", "Jun. 23, 2022", "Aisle_CCW_Run_2"),
    ("val", "jun23_cw_run2", "Jun. 23, 2022", "Aisle_CW_Run_2"),
)


def coarse_config(sequence_name: str, sequence_root: Path) -> dict[str, object]:
    cache_dir = f"./cache/semantic_stability_gate/{sequence_name}"
    return {
        "sequence_root": str(sequence_root).replace("\\", "/"),
        "calibration_path": str(CALIBRATION_PATH).replace("\\", "/"),
        "map_path": str(MAP_PATH).replace("\\", "/"),
        "crop_x_min": -10.0,
        "crop_x_max": 10.0,
        "crop_y_min": -10.0,
        "crop_y_max": 10.0,
        "crop_z_min": 0.13,
        "crop_z_max": 4.73,
        "bev_resolution": 0.1,
        "bev_size_xy_m": 20.0,
        "patch_size_m": 20.0,
        "patch_stride_m": 10.0,
        "descriptor_dim": 256,
        "topk": 3,
        "cache_dir": cache_dir,
        "align_query_to_gt_yaw_train": True,
        "use_query_rotation_search": True,
        "query_rotation_search_step_deg": 30.0,
        "num_hard_negative_patches": 8,
        "num_random_negative_patches": 8,
        "hard_negative_min_distance_m": 5.0,
        "hard_negative_max_distance_m": 30.0,
        "train_batch_size": 6,
        "train_num_workers": 4,
        "train_epochs": 2,
        "learning_rate": 0.0005,
        "weight_decay": 0.0001,
        "lr_decay_gamma": 0.98,
        "temperature": 0.07,
        "model_seed": 0,
        "device": "cuda",
        "use_amp": True,
        "eval_every_epochs": 1,
        "save_best_only": True,
        "share_query_patch_encoder": False,
        "use_patch_classification_loss": True,
        "classification_loss_weight": 1.0,
        "classifier_score_weight": 0.75,
    }


def fine_config(sequence_name: str, coarse_path: Path) -> dict[str, object]:
    return {
        "coarse_config_path": str(coarse_path).replace("\\", "/"),
        "coarse_checkpoint_path": "checkpoints/coarse_retrieval_classifier_head_e2_stride10_top1_0p4426_r3_0p7672.pt",
        "descriptor_bank_path": f"./cache/semantic_stability_gate/{sequence_name}/descriptor_bank_{sequence_name}.npz",
        "deep_matcher_checkpoint_path": "outputs/fine_pose_matcher_jun15_fullroutes_generic.pt",
        "output_json": f"outputs/semantic_stability_20260731/experiment3/gate_collection_{sequence_name}.json",
        "topk_candidates": 5,
        "local_submap_size_m": 30.0,
        "local_submap_resolution": 0.1,
        "voxel_size_m": 0.3,
        "icp_max_iterations": 20,
        "icp_max_correspondence_distance_m": 1.0,
        "icp_min_correspondences": 24,
        "retrieval_score_weight": 0.2,
        "deep_matcher_score_weight": 0.35,
        "deep_pose_confidence_weight": 0.10,
        "icp_inlier_weight": 0.35,
        "icp_rmse_weight": 0.10,
        "temporal_weight": 0.20,
        "temporal_position_sigma_m": 1.5,
        "temporal_yaw_sigma_deg": 20.0,
        "use_constant_velocity_prediction": True,
        "use_sequence_smoothing": False,
        "deep_pose_init_xy_consistency_weight": 0.0,
        "deep_pose_init_yaw_consistency_weight": 0.0,
        "use_deep_pose_init_hypothesis": True,
        "use_tracker_pose_init_hypothesis": True,
        "use_online_pose_stabilizer": True,
        "online_stabilizer_max_position_jump_m": 2.5,
        "online_stabilizer_max_yaw_jump_deg": 20.0,
        "online_stabilizer_min_icp_inlier_ratio": 0.55,
        "online_stabilizer_min_bev_score": 0.08,
        "online_stabilizer_score_margin": 0.20,
        "online_stabilizer_temporal_bonus": 0.10,
        "online_stabilizer_fallback_to_tracker": True,
        "use_online_patch_hysteresis": True,
        "online_patch_switch_margin": 0.30,
        "use_persistent_patch_override": False,
        "use_top2_confusion_geometry_override": False,
        "apply_online_patch_hysteresis_during_tracking": True,
        "tracker_pose_init_sticky_patch_streak_min": 5,
        "tracker_pose_init_sticky_prev_deep_prob_max": 0.05,
        "tracker_pose_init_sticky_required_margin": 0.0,
    }


def main() -> None:
    root = Path(__file__).resolve().parent / "experiment3_configs"
    root.mkdir(parents=True, exist_ok=True)
    for split, sequence_name, date_name, route_name in SEQUENCES:
        sequence_root = DATA_ROOT / date_name / route_name / route_name
        coarse_path = root / f"{split}_{sequence_name}_coarse.yaml"
        fine_path = root / f"{split}_{sequence_name}_fine.yaml"
        coarse_path.write_text(
            yaml.safe_dump(coarse_config(sequence_name, sequence_root), sort_keys=False),
            encoding="utf-8",
        )
        fine_path.write_text(
            yaml.safe_dump(fine_config(sequence_name, coarse_path), sort_keys=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
