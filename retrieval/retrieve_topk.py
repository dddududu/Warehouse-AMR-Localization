from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def load_descriptor_bank(
    descriptor_bank_path: str | Path,
    include_patch_tensors: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict]] | tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    path = Path(descriptor_bank_path)
    bank = np.load(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["patch_metadata"]
    if include_patch_tensors:
        return bank["descriptor_bank"], bank["patch_ids"], metadata, bank["patch_tensors"]
    return bank["descriptor_bank"], bank["patch_ids"], metadata


def _standardize_scores(scores: np.ndarray) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float32)
    mean = float(array.mean())
    std = float(array.std())
    if std < 1.0e-6:
        return array - mean
    return (array - mean) / std


def _resolve_ensemble_config(
    cfg,
    checkpoint_path: str | Path | None,
) -> tuple[list[str], list[float], list[float]]:
    checkpoint_paths = [str(path) for path in (cfg.ensemble_checkpoint_paths or ([] if checkpoint_path is None else [checkpoint_path]))]
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint path is required for retrieval.")
    model_weights = list(cfg.ensemble_model_weights or [1.0] * len(checkpoint_paths))
    classifier_score_weights = list(
        cfg.ensemble_classifier_score_weights or [float(cfg.classifier_score_weight)] * len(checkpoint_paths)
    )
    if len(model_weights) != len(checkpoint_paths):
        raise ValueError("ensemble_model_weights length must match ensemble_checkpoint_paths length.")
    if len(classifier_score_weights) != len(checkpoint_paths):
        raise ValueError("ensemble_classifier_score_weights length must match ensemble_checkpoint_paths length.")
    weight_sum = float(sum(model_weights))
    if weight_sum <= 0.0:
        raise ValueError("ensemble_model_weights must sum to a positive value.")
    normalized_weights = [float(weight) / weight_sum for weight in model_weights]
    return checkpoint_paths, normalized_weights, classifier_score_weights


def _build_retrieval_model(
    cfg,
    checkpoint_path: str | Path | None,
    num_patch_classes: int | None,
    device: torch.device,
) -> tuple[CoarseRetrievalModel, dict[str, Any]]:
    state: dict[str, Any] = {} if checkpoint_path is None else torch.load(checkpoint_path, map_location=device)
    state_cfg = state.get("config", {}) if isinstance(state, dict) else {}
    retrieval_model = CoarseRetrievalModel(
        descriptor_dim=cfg.descriptor_dim,
        init_seed=cfg.model_seed,
        backbone_variant=str(state_cfg.get("backbone_variant", getattr(cfg, "backbone_variant", "legacy"))),
        share_query_patch_encoder=bool((state or {}).get("config", {}).get("share_query_patch_encoder", False)),
        num_patch_classes=num_patch_classes if state.get("query_classifier") is not None else None,
    ).to(device)
    if state.get("query_encoder") is not None:
        retrieval_model.query_encoder.load_state_dict(state["query_encoder"])
    if state.get("patch_encoder") is not None:
        retrieval_model.patch_encoder.load_state_dict(state["patch_encoder"])
    if retrieval_model.query_classifier is not None and state.get("query_classifier") is not None:
        retrieval_model.query_classifier.load_state_dict(state["query_classifier"])
    retrieval_model.eval()
    return retrieval_model, state


def build_ensemble_specs(
    cfg,
    checkpoint_path: str | Path | None,
    patch_tensors: np.ndarray,
    device: torch.device,
) -> list[dict[str, Any]]:
    checkpoint_paths, model_weights, classifier_score_weights = _resolve_ensemble_config(cfg, checkpoint_path)
    specs: list[dict[str, Any]] = []
    for path, model_weight, classifier_score_weight in zip(checkpoint_paths, model_weights, classifier_score_weights):
        retrieval_model, _ = _build_retrieval_model(
            cfg,
            checkpoint_path=path,
            num_patch_classes=int(patch_tensors.shape[0]),
            device=device,
        )
        descriptors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(patch_tensors), 16):
                batch = torch.from_numpy(patch_tensors[start : start + 16]).to(device).float()
                descriptors.append(retrieval_model.encode_patch(batch).cpu().numpy())
        specs.append(
            {
                "model": retrieval_model,
                "descriptor_bank": np.concatenate(descriptors, axis=0),
                "model_weight": float(model_weight),
                "classifier_score_weight": float(classifier_score_weight),
            }
        )
    return specs


def combine_model_scores(
    score_entries: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray]:
    combined_scores = np.zeros_like(score_entries[0][0], dtype=np.float32)
    best_angles = np.array(score_entries[0][1], copy=True)
    best_contribution = np.full_like(combined_scores, -np.inf, dtype=np.float32)
    for scores, angles, model_weight in score_entries:
        contribution = float(model_weight) * _standardize_scores(scores)
        combined_scores += contribution
        improve = contribution > best_contribution
        best_contribution[improve] = contribution[improve]
        best_angles[improve] = angles[improve]
    return combined_scores, best_angles


def score_query_bev_with_ensemble(
    query_bev: np.ndarray,
    specs: list[dict[str, Any]],
    rotation_angles_deg: list[float],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    score_entries: list[tuple[np.ndarray, np.ndarray, float]] = []
    for spec in specs:
        scores, best_angles = score_query_bev_against_bank(
            query_bev=query_bev,
            encoder=spec["model"].query_encoder,
            descriptor_bank=spec["descriptor_bank"],
            rotation_angles_deg=rotation_angles_deg,
            device=device,
            classifier=spec["model"].query_classifier,
            classifier_score_weight=spec["classifier_score_weight"],
        )
        score_entries.append((scores, best_angles, spec["model_weight"]))
    return combine_model_scores(score_entries)


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
    descriptor_bank, patch_ids, metadata, patch_tensors = load_descriptor_bank(
        descriptor_bank_path,
        include_patch_tensors=True,
    )
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
    if cfg.ensemble_checkpoint_paths:
        ensemble_specs = build_ensemble_specs(single_cfg, checkpoint_path=None, patch_tensors=patch_tensors, device=device)
        scores, best_angles = score_query_bev_with_ensemble(
            query_bev=query_bev,
            specs=ensemble_specs,
            rotation_angles_deg=single_cfg.query_rotation_search_angles_deg,
            device=device,
        )
    else:
        retrieval_model, state = _build_retrieval_model(
            single_cfg,
            checkpoint_path=checkpoint_path,
            num_patch_classes=int(descriptor_bank.shape[0]),
            device=device,
        )
        if state.get("query_classifier") is None:
            retrieval_model.query_classifier = None
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
