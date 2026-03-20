from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from runtime_compat import ensure_runtime_compatibility

ensure_runtime_compatibility()

import torch
from torch import nn

from dataset_io.lidar_loader import load_pcd_xyz
from dataset_io.sequence_dataset import WarehouseSequenceDataset
from models.coarse_retrieval_model import CoarseRetrievalModel
from models.coarse_query_encoder import CoarseQueryEncoder
from preprocess.bev_builder import BEVConfig, points_to_bev
from preprocess.bev_transforms import rotate_bev_tensor
from preprocess.local_lidar_cropper import LocalCropConfig, crop_local_lidar_points
from preprocess.map_patch_builder import PatchMetadata, choose_gt_patch_id
from retrieval.config import load_coarse_retrieval_config


def load_descriptor_bank(descriptor_bank_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    path = Path(descriptor_bank_path)
    bank = np.load(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["patch_metadata"]
    return bank["descriptor_bank"], bank["patch_ids"], metadata


def _standardize_scores(scores: np.ndarray) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float32)
    mean = float(array.mean())
    std = float(array.std())
    if std < 1.0e-6:
        return array - mean
    return (array - mean) / std


def score_query_bev_against_bank(
    query_bev: np.ndarray,
    encoder: CoarseQueryEncoder,
    descriptor_bank: np.ndarray,
    rotation_angles_deg: list[float],
    device: torch.device,
    classifier: nn.Module | None = None,
    classifier_score_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    best_scores = np.full(descriptor_bank.shape[0], -np.inf, dtype=np.float32)
    best_angles = np.zeros(descriptor_bank.shape[0], dtype=np.float32)
    with torch.no_grad():
        for angle_deg in rotation_angles_deg:
            rotated_bev = rotate_bev_tensor(query_bev, angle_deg) if abs(angle_deg) > 1e-6 else query_bev
            descriptor_tensor = encoder(torch.from_numpy(rotated_bev[None, ...]).to(device))
            descriptor = descriptor_tensor.cpu().numpy()[0]
            scores = descriptor_bank @ descriptor
            if classifier is not None and classifier_score_weight > 0.0:
                classifier_scores = classifier(descriptor_tensor).cpu().numpy()[0]
                if classifier_scores.shape[0] == descriptor_bank.shape[0]:
                    scores = (
                        (1.0 - float(classifier_score_weight)) * _standardize_scores(scores)
                        + float(classifier_score_weight) * _standardize_scores(classifier_scores)
                    )
            improve = scores > best_scores
            best_scores[improve] = scores[improve]
            best_angles[improve] = float(angle_deg)
    return best_scores, best_angles


def retrieve_topk_for_frame(
    frame_idx: int,
    config,
    descriptor_bank_path: str | Path,
    checkpoint_path: str | Path | None = None,
    sequence_name: str | None = None,
) -> dict:
    cfg = load_coarse_retrieval_config(config)
    descriptor_bank, patch_ids, metadata = load_descriptor_bank(descriptor_bank_path)
    entry = cfg.resolve_single_sequence_entry(sequence_name=sequence_name, prefer_validation=True)
    single_cfg = load_coarse_retrieval_config(
        {
            **cfg.__dict__,
            "sequence_root": entry["sequence_root"],
            "calibration_path": entry["calibration_path"],
            "map_path": entry["map_path"],
        }
    )
    dataset = WarehouseSequenceDataset(
        sequence_root=single_cfg.sequence_root,
        calibration_path=single_cfg.calibration_path,
        config={"load_lidar": False},
    )
    record = dataset.frame_index[frame_idx]
    points = load_pcd_xyz(record.lidar_path)
    cropper = LocalCropConfig(
        x_min=cfg.crop_x_min,
        x_max=single_cfg.crop_x_max,
        y_min=single_cfg.crop_y_min,
        y_max=single_cfg.crop_y_max,
        z_min=single_cfg.crop_z_min,
        z_max=single_cfg.crop_z_max,
    )
    bev_config = BEVConfig(
        x_min=single_cfg.crop_x_min,
        x_max=single_cfg.crop_x_max,
        y_min=single_cfg.crop_y_min,
        y_max=single_cfg.crop_y_max,
        resolution=single_cfg.bev_resolution,
    )
    query_bev = points_to_bev(crop_local_lidar_points(points, cropper), bev_config)

    device = torch.device(single_cfg.device)
    state = torch.load(checkpoint_path, map_location=device) if checkpoint_path is not None else None
    num_patch_classes = int(descriptor_bank.shape[0]) if state and state.get("query_classifier") is not None else None
    retrieval_model = CoarseRetrievalModel(
        descriptor_dim=single_cfg.descriptor_dim,
        init_seed=single_cfg.model_seed,
        share_query_patch_encoder=bool((state or {}).get("config", {}).get("share_query_patch_encoder", False)),
        num_patch_classes=num_patch_classes,
    ).to(device)
    if checkpoint_path is not None:
        retrieval_model.query_encoder.load_state_dict(state["query_encoder"])
        if retrieval_model.query_classifier is not None and state.get("query_classifier") is not None:
            retrieval_model.query_classifier.load_state_dict(state["query_classifier"])
    retrieval_model.eval()

    scores, best_angles = score_query_bev_against_bank(
        query_bev=query_bev,
        encoder=retrieval_model.query_encoder,
        descriptor_bank=descriptor_bank,
        rotation_angles_deg=single_cfg.query_rotation_search_angles_deg,
        device=device,
        classifier=retrieval_model.query_classifier,
        classifier_score_weight=single_cfg.classifier_score_weight,
    )
    order = np.argsort(scores)[::-1][: single_cfg.topk]
    topk_patch_ids = patch_ids[order].astype(int).tolist()

    gt_position = dataset.ground_truth.positions[frame_idx]
    gt_patch_id = choose_gt_patch_id(
        [PatchMetadata(**item) for item in metadata],
        float(gt_position[0]),
        float(gt_position[1]),
    )
    result = {
        "frame_idx": int(frame_idx),
        "sequence_name": entry["sequence_name"],
        "gt_patch_id": int(gt_patch_id),
        "gt_in_topk": bool(gt_patch_id in topk_patch_ids),
        "topk_patch_ids": topk_patch_ids,
        "topk_scores": scores[order].astype(float).tolist(),
        "topk_best_query_rotation_deg": best_angles[order].astype(float).tolist(),
        "topk_patch_centers_xy": [metadata[i]["center_xy"] for i in order],
        "topk_patch_bboxes": [metadata[i]["bbox_xy"] for i in order],
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-k map patches for one query frame.")
    parser.add_argument("--frame-idx", type=int, required=True)
    parser.add_argument("--config", default="configs/coarse_retrieval_a.yaml")
    parser.add_argument("--descriptor-bank", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sequence-name", default=None)
    args = parser.parse_args()
    retrieve_topk_for_frame(
        frame_idx=args.frame_idx,
        config=args.config,
        descriptor_bank_path=args.descriptor_bank,
        checkpoint_path=args.checkpoint,
        sequence_name=args.sequence_name,
    )


if __name__ == "__main__":
    main()
