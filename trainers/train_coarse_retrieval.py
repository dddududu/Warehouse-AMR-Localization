from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime_compat import ensure_runtime_compatibility

ensure_runtime_compatibility()

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_io.retrieval_dataset import CoarseRetrievalDataset
from losses.retrieval_loss import RetrievalInfoNCELoss
from models.coarse_retrieval_model import CoarseRetrievalModel
from retrieval.config import load_coarse_retrieval_config
from retrieval.retrieve_topk import score_query_bev_against_bank


def _build_patch_descriptor_bank(model: CoarseRetrievalModel, patch_tensors: np.ndarray, device: torch.device) -> np.ndarray:
    descriptors: list[np.ndarray] = []
    model.patch_encoder.eval()
    with torch.no_grad():
        for start in range(0, len(patch_tensors), 16):
            batch = torch.from_numpy(patch_tensors[start : start + 16]).to(device).float()
            descriptors.append(model.encode_patch(batch).cpu().numpy())
    return np.concatenate(descriptors, axis=0)


def _evaluate_recall(model: CoarseRetrievalModel, dataset: CoarseRetrievalDataset, device: torch.device, topk: int) -> dict:
    if len(dataset) == 0:
        return {"num_val_frames": 0, "recall@1": None, "recall@5": None, f"recall@{topk}": None}

    per_sequence_descriptor_bank = {
        resources.sequence_name: _build_patch_descriptor_bank(model, resources.patch_tensors, device=device)
        for resources in dataset.sequence_resources
    }
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_k = 0
    model.query_encoder.eval()
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        query_bev = sample["query_bev"].numpy()
        gt_patch_id = int(sample["gt_patch_id"])
        sequence_name = str(sample["sequence_name"])
        scores, _ = score_query_bev_against_bank(
            query_bev=query_bev,
            encoder=model.query_encoder,
            descriptor_bank=per_sequence_descriptor_bank[sequence_name],
            rotation_angles_deg=dataset.config.query_rotation_search_angles_deg,
            device=device,
        )
        ranking = np.argsort(scores)[::-1]
        if gt_patch_id == int(ranking[0]):
            hits_at_1 += 1
        if gt_patch_id in ranking[: min(5, ranking.shape[0])]:
            hits_at_5 += 1
        if gt_patch_id in ranking[: min(topk, ranking.shape[0])]:
            hits_at_k += 1

    num_frames = len(dataset)
    return {
        "num_val_frames": num_frames,
        "recall@1": hits_at_1 / num_frames,
        "recall@5": hits_at_5 / num_frames,
        f"recall@{topk}": hits_at_k / num_frames,
    }


def train_coarse_retrieval(config, output_checkpoint: str | Path | None = None) -> dict:
    cfg = load_coarse_retrieval_config(config)
    train_entries, val_entries = cfg.split_sequence_entries()
    train_dataset = CoarseRetrievalDataset(cfg, sequence_entries=train_entries, align_query_to_gt_yaw=True)
    val_dataset = CoarseRetrievalDataset(cfg, sequence_entries=val_entries, align_query_to_gt_yaw=False)
    dataloader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True)
    device = torch.device(cfg.device)

    model = CoarseRetrievalModel(descriptor_dim=cfg.descriptor_dim, init_seed=cfg.model_seed).to(device)
    criterion = RetrievalInfoNCELoss(temperature=cfg.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history: list[float] = []
    validation_history: list[dict] = []
    best_metric = -1.0
    best_state: dict | None = None
    for epoch_idx in range(cfg.train_epochs):
        model.train()
        for batch in dataloader:
            query_bev = batch["query_bev"].to(device).float()
            positive_patch_bev = batch["positive_patch_bev"].to(device).float()
            negative_patch_bevs = batch["negative_patch_bevs"].to(device).float()

            outputs = model(query_bev, positive_patch_bev, negative_patch_bevs)
            loss = criterion(
                outputs["query_descriptor"],
                outputs["positive_descriptor"],
                outputs["negative_descriptor"],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(float(loss.item()))

        if (epoch_idx + 1) % cfg.eval_every_epochs == 0:
            metrics = _evaluate_recall(model, val_dataset, device=device, topk=cfg.topk)
            metrics["epoch"] = epoch_idx + 1
            validation_history.append(metrics)
            recall_at_1 = metrics["recall@1"] or 0.0
            if recall_at_1 >= best_metric:
                best_metric = recall_at_1
                best_state = {
                    "query_encoder": model.query_encoder.state_dict(),
                    "patch_encoder": model.patch_encoder.state_dict(),
                    "history": history.copy(),
                    "validation_history": validation_history.copy(),
                    "config": cfg.__dict__,
                }

    final_state = {
        "query_encoder": model.query_encoder.state_dict(),
        "patch_encoder": model.patch_encoder.state_dict(),
        "history": history,
        "validation_history": validation_history,
        "config": cfg.__dict__,
    }
    checkpoint_payload = best_state if (cfg.save_best_only and best_state is not None) else final_state

    if output_checkpoint is None:
        checkpoint_path = Path(cfg.cache_dir) / "coarse_retrieval_model.pt"
    else:
        checkpoint_path = Path(output_checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, checkpoint_path)
    result = {
        "checkpoint_path": str(checkpoint_path),
        "num_steps": len(history),
        "final_loss": history[-1] if history else None,
        "validation_history": validation_history,
        "num_train_frames": int(len(train_dataset)),
        "num_val_frames": int(len(val_dataset)),
        "train_sequences": [entry["sequence_name"] for entry in train_entries],
        "val_sequences": [entry["sequence_name"] for entry in val_entries],
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the coarse retrieval model.")
    parser.add_argument("--config", default="configs/coarse_retrieval_a.yaml")
    parser.add_argument("--output-checkpoint", default=None)
    args = parser.parse_args()
    train_coarse_retrieval(args.config, output_checkpoint=args.output_checkpoint)


if __name__ == "__main__":
    main()
