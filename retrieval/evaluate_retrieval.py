from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from runtime_compat import ensure_runtime_compatibility

ensure_runtime_compatibility()

import torch

from dataset_io.retrieval_dataset import CoarseRetrievalDataset
from dataset_io.lidar_loader import load_pcd_xyz
from dataset_io.sequence_dataset import WarehouseSequenceDataset
from models.coarse_retrieval_model import CoarseRetrievalModel
from models.coarse_query_encoder import CoarseQueryEncoder
from preprocess.bev_builder import BEVConfig, points_to_bev
from preprocess.local_lidar_cropper import LocalCropConfig, crop_local_lidar_points
from preprocess.map_patch_builder import PatchMetadata, choose_gt_patch_id
from retrieval.config import load_coarse_retrieval_config
from retrieval.retrieve_topk import (
    build_ensemble_specs,
    combine_model_scores,
    load_descriptor_bank,
    score_query_bev_against_bank,
)


def evaluate_retrieval(
    config,
    descriptor_bank_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    frame_start: int = 0,
    num_frames: int | None = None,
    frame_stride: int = 1,
    output_json: str | Path | None = None,
) -> dict:
    cfg = load_coarse_retrieval_config(config)
    device = torch.device(cfg.device)

    train_entries, val_entries = cfg.split_sequence_entries()
    sequence_entries = val_entries if val_entries else train_entries

    if len(sequence_entries) == 1 and descriptor_bank_path is not None:
        descriptor_bank, patch_ids, metadata_raw, patch_tensors = load_descriptor_bank(
            descriptor_bank_path,
            include_patch_tensors=True,
        )
        metadata = [PatchMetadata(**item) for item in metadata_raw]
        dataset = WarehouseSequenceDataset(
            sequence_root=sequence_entries[0]["sequence_root"],
            calibration_path=sequence_entries[0]["calibration_path"],
            config={"load_lidar": False},
        )
        cropper = LocalCropConfig(
            x_min=cfg.crop_x_min,
            x_max=cfg.crop_x_max,
            y_min=cfg.crop_y_min,
            y_max=cfg.crop_y_max,
            z_min=cfg.crop_z_min,
            z_max=cfg.crop_z_max,
        )
        bev_config = BEVConfig(
            x_min=cfg.crop_x_min,
            x_max=cfg.crop_x_max,
            y_min=cfg.crop_y_min,
            y_max=cfg.crop_y_max,
            resolution=cfg.bev_resolution,
        )
        ensemble_specs = (
            build_ensemble_specs(cfg, checkpoint_path=None, patch_tensors=patch_tensors, device=device)
            if cfg.ensemble_checkpoint_paths
            else None
        )
        model = None
        if ensemble_specs is None:
            state = torch.load(checkpoint_path, map_location=device) if checkpoint_path is not None else None
            num_patch_classes = int(descriptor_bank.shape[0]) if state and state.get("query_classifier") is not None else None
            model = CoarseRetrievalModel(
                descriptor_dim=cfg.descriptor_dim,
                init_seed=cfg.model_seed,
                share_query_patch_encoder=bool((state or {}).get("config", {}).get("share_query_patch_encoder", False)),
                num_patch_classes=num_patch_classes,
            ).to(device)
            if checkpoint_path is not None:
                model.query_encoder.load_state_dict(state["query_encoder"])
                if model.query_classifier is not None and state.get("query_classifier") is not None:
                    model.query_classifier.load_state_dict(state["query_classifier"])
            model.eval()

        last_frame = len(dataset) if num_frames is None else min(len(dataset), frame_start + num_frames)
        frame_indices = list(range(frame_start, last_frame, max(1, int(frame_stride))))

        hits_at_k = 0
        hits_at_1 = 0
        reciprocal_ranks: list[float] = []
        details: list[dict] = []
        for frame_idx in frame_indices:
            record = dataset.frame_index[frame_idx]
            points = load_pcd_xyz(record.lidar_path)
            query_bev = points_to_bev(crop_local_lidar_points(points, cropper), bev_config)
            if ensemble_specs is not None:
                score_entries = []
                for spec in ensemble_specs:
                    scores_i, angles_i = score_query_bev_against_bank(
                        query_bev=query_bev,
                        encoder=spec["model"].query_encoder,
                        descriptor_bank=spec["descriptor_bank"],
                        rotation_angles_deg=cfg.query_rotation_search_angles_deg,
                        device=device,
                        classifier=spec["model"].query_classifier,
                        classifier_score_weight=spec["classifier_score_weight"],
                    )
                    score_entries.append((scores_i, angles_i, spec["model_weight"]))
                scores, best_angles = combine_model_scores(score_entries)
            else:
                scores, best_angles = score_query_bev_against_bank(
                    query_bev=query_bev,
                    encoder=model.query_encoder,
                    descriptor_bank=descriptor_bank,
                    rotation_angles_deg=cfg.query_rotation_search_angles_deg,
                    device=device,
                    classifier=model.query_classifier,
                    classifier_score_weight=cfg.classifier_score_weight,
                )
            ranking = np.argsort(scores)[::-1]
            topk_order = ranking[: cfg.topk]
            topk_patch_ids = patch_ids[topk_order].astype(int).tolist()

            gt_position = dataset.ground_truth.positions[frame_idx]
            gt_patch_id = choose_gt_patch_id(metadata, float(gt_position[0]), float(gt_position[1]))
            ranked_patch_ids = patch_ids[ranking].astype(int)
            gt_rank = int(np.where(ranked_patch_ids == gt_patch_id)[0][0]) + 1

            if gt_patch_id == topk_patch_ids[0]:
                hits_at_1 += 1
            if gt_patch_id in topk_patch_ids:
                hits_at_k += 1
            reciprocal_ranks.append(1.0 / gt_rank)
            details.append(
                {
                    "sequence_name": sequence_entries[0]["sequence_name"],
                    "frame_idx": int(frame_idx),
                    "timestamp": float(record.timestamp),
                    "gt_patch_id": int(gt_patch_id),
                    "gt_rank": int(gt_rank),
                    "topk_patch_ids": topk_patch_ids,
                    "topk_scores": scores[topk_order].astype(float).tolist(),
                    "topk_best_query_rotation_deg": best_angles[topk_order].astype(float).tolist(),
                    "gt_in_topk": bool(gt_patch_id in topk_patch_ids),
                }
            )
        num_eval_frames = len(frame_indices)
        report = {
            "num_eval_frames": num_eval_frames,
            "evaluated_sequences": [sequence_entries[0]["sequence_name"]],
            "frame_start": int(frame_start),
            "frame_stride": int(frame_stride),
            "topk": int(cfg.topk),
            "rotation_search_angles_deg": cfg.query_rotation_search_angles_deg,
            "top1_accuracy": hits_at_1 / num_eval_frames if num_eval_frames else None,
            f"recall@{cfg.topk}": hits_at_k / num_eval_frames if num_eval_frames else None,
            "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None,
            "details": details,
        }
    else:
        dataset = CoarseRetrievalDataset(cfg, sequence_entries=sequence_entries, align_query_to_gt_yaw=False)
        patch_counts = {len(resources.patch_tensors) for resources in dataset.sequence_resources}
        num_patch_classes = int(next(iter(patch_counts))) if len(patch_counts) == 1 else None
        model = None
        ensemble_specs = None
        if cfg.ensemble_checkpoint_paths:
            first_resources = dataset.sequence_resources[0]
            ensemble_specs = build_ensemble_specs(
                cfg,
                checkpoint_path=None,
                patch_tensors=first_resources.patch_tensors,
                device=device,
            )
        else:
            model = CoarseRetrievalModel(
                descriptor_dim=cfg.descriptor_dim,
                init_seed=cfg.model_seed,
                share_query_patch_encoder=cfg.share_query_patch_encoder,
                num_patch_classes=num_patch_classes if cfg.classifier_score_weight > 0.0 else None,
            ).to(device)
            if checkpoint_path is not None:
                state = torch.load(checkpoint_path, map_location=device)
                model.query_encoder.load_state_dict(state["query_encoder"])
                model.patch_encoder.load_state_dict(state["patch_encoder"])
                if model.query_classifier is not None and state.get("query_classifier") is not None:
                    model.query_classifier.load_state_dict(state["query_classifier"])
            model.eval()
        per_sequence_descriptor_bank: dict[str, np.ndarray] = {}
        per_sequence_ensemble_banks: dict[str, list[np.ndarray]] = {}
        per_sequence_patch_ids: dict[str, np.ndarray] = {}
        for resources in dataset.sequence_resources:
            if ensemble_specs is not None:
                model_banks: list[np.ndarray] = []
                for spec in ensemble_specs:
                    descriptors: list[np.ndarray] = []
                    with torch.no_grad():
                        for start in range(0, len(resources.patch_tensors), 16):
                            batch = torch.from_numpy(resources.patch_tensors[start : start + 16]).to(device).float()
                            descriptors.append(spec["model"].encode_patch(batch).cpu().numpy())
                    model_banks.append(np.concatenate(descriptors, axis=0))
                per_sequence_ensemble_banks[resources.sequence_name] = model_banks
            else:
                descriptors = []
                with torch.no_grad():
                    for start in range(0, len(resources.patch_tensors), 16):
                        batch = torch.from_numpy(resources.patch_tensors[start : start + 16]).to(device).float()
                        descriptors.append(model.encode_patch(batch).cpu().numpy())
                per_sequence_descriptor_bank[resources.sequence_name] = np.concatenate(descriptors, axis=0)
            per_sequence_patch_ids[resources.sequence_name] = np.arange(len(resources.patch_tensors), dtype=np.int32)

        filtered_sample_indices = [
            sample_idx
            for sample_idx, (_, frame_idx) in enumerate(dataset.sample_index)
            if frame_idx >= frame_start and ((frame_idx - frame_start) % max(1, int(frame_stride)) == 0)
        ]
        if num_frames is not None:
            filtered_sample_indices = filtered_sample_indices[:num_frames]

        hits_at_k = 0
        hits_at_1 = 0
        reciprocal_ranks: list[float] = []
        details: list[dict] = []
        for sample_index in filtered_sample_indices:
            sample = dataset[sample_index]
            sequence_name = str(sample["sequence_name"])
            gt_patch_id = int(sample["gt_patch_id"])
            if ensemble_specs is not None:
                score_entries = []
                for spec, descriptor_bank_i in zip(ensemble_specs, per_sequence_ensemble_banks[sequence_name]):
                    scores_i, angles_i = score_query_bev_against_bank(
                        query_bev=sample["query_bev"].numpy(),
                        encoder=spec["model"].query_encoder,
                        descriptor_bank=descriptor_bank_i,
                        rotation_angles_deg=cfg.query_rotation_search_angles_deg,
                        device=device,
                        classifier=spec["model"].query_classifier,
                        classifier_score_weight=spec["classifier_score_weight"],
                    )
                    score_entries.append((scores_i, angles_i, spec["model_weight"]))
                scores, best_angles = combine_model_scores(score_entries)
            else:
                scores, best_angles = score_query_bev_against_bank(
                    query_bev=sample["query_bev"].numpy(),
                    encoder=model.query_encoder,
                    descriptor_bank=per_sequence_descriptor_bank[sequence_name],
                    rotation_angles_deg=cfg.query_rotation_search_angles_deg,
                    device=device,
                    classifier=model.query_classifier,
                    classifier_score_weight=cfg.classifier_score_weight,
                )
            ranking = np.argsort(scores)[::-1]
            patch_ids = per_sequence_patch_ids[sequence_name]
            topk_order = ranking[: cfg.topk]
            topk_patch_ids = patch_ids[topk_order].astype(int).tolist()
            ranked_patch_ids = patch_ids[ranking].astype(int)
            gt_rank = int(np.where(ranked_patch_ids == gt_patch_id)[0][0]) + 1
            if gt_patch_id == topk_patch_ids[0]:
                hits_at_1 += 1
            if gt_patch_id in topk_patch_ids:
                hits_at_k += 1
            reciprocal_ranks.append(1.0 / gt_rank)
            details.append(
                {
                    "sequence_name": sequence_name,
                    "frame_idx": int(sample["frame_idx"]),
                    "gt_patch_id": int(gt_patch_id),
                    "gt_rank": int(gt_rank),
                    "topk_patch_ids": topk_patch_ids,
                    "topk_scores": scores[topk_order].astype(float).tolist(),
                    "topk_best_query_rotation_deg": best_angles[topk_order].astype(float).tolist(),
                    "gt_in_topk": bool(gt_patch_id in topk_patch_ids),
                }
            )
        num_eval_frames = len(filtered_sample_indices)
        report = {
            "num_eval_frames": num_eval_frames,
            "evaluated_sequences": [entry["sequence_name"] for entry in sequence_entries],
            "frame_start": int(frame_start),
            "frame_stride": int(frame_stride),
            "topk": int(cfg.topk),
            "rotation_search_angles_deg": cfg.query_rotation_search_angles_deg,
            "top1_accuracy": hits_at_1 / num_eval_frames if num_eval_frames else None,
            f"recall@{cfg.topk}": hits_at_k / num_eval_frames if num_eval_frames else None,
            "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None,
            "details": details,
        }
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate coarse retrieval accuracy over a frame range.")
    parser.add_argument("--config", default="configs/coarse_retrieval_a.yaml")
    parser.add_argument("--descriptor-bank", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    evaluate_retrieval(
        config=args.config,
        descriptor_bank_path=args.descriptor_bank,
        checkpoint_path=args.checkpoint,
        frame_start=args.frame_start,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
