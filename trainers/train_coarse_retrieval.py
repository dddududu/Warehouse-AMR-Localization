from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime_compat import ensure_runtime_compatibility

ensure_runtime_compatibility()

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset_io.retrieval_dataset import CoarseRetrievalDataset
from losses.retrieval_loss import RetrievalInfoNCELoss
from models.coarse_retrieval_model import CoarseRetrievalModel
from retrieval.config import load_coarse_retrieval_config
from retrieval.retrieve_topk import build_patch_search_bank, score_query_bev_against_bank


def _resolve_num_patch_classes(dataset: CoarseRetrievalDataset) -> int | None:
    patch_counts = {len(resources.patch_tensors) for resources in dataset.sequence_resources}
    if not patch_counts:
        return None
    if len(patch_counts) != 1:
        raise ValueError("Patch classification requires all sequences to share the same patch count.")
    return int(next(iter(patch_counts)))


def _load_init_checkpoint(model: CoarseRetrievalModel, checkpoint_path: str | Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if state.get("query_encoder") is not None:
        model.query_encoder.load_state_dict(state["query_encoder"], strict=False)
    if state.get("patch_encoder") is not None:
        model.patch_encoder.load_state_dict(state["patch_encoder"], strict=False)
    if model.query_classifier is not None and state.get("query_classifier") is not None:
        classifier_state = state["query_classifier"]
        if classifier_state is not None:
            try:
                model.query_classifier.load_state_dict(classifier_state, strict=False)
            except RuntimeError:
                pass
    if model.local_matcher is not None and state.get("local_matcher") is not None:
        local_matcher_state = state["local_matcher"]
        if local_matcher_state is not None:
            try:
                model.local_matcher.load_state_dict(local_matcher_state, strict=False)
            except RuntimeError:
                pass


def _set_module_trainable(module: torch.nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = bool(trainable)


def _evaluate_recall(model: CoarseRetrievalModel, dataset: CoarseRetrievalDataset, device: torch.device, topk: int) -> dict:
    if len(dataset) == 0:
        return {"num_val_frames": 0, "recall@1": None, "recall@5": None, f"recall@{topk}": None}

    per_sequence_descriptor_bank: dict[str, np.ndarray] = {}
    per_sequence_local_feature_bank: dict[str, np.ndarray | None] = {}
    for resources in dataset.sequence_resources:
        descriptor_bank, local_feature_bank = build_patch_search_bank(model, resources.patch_tensors, device=device)
        per_sequence_descriptor_bank[resources.sequence_name] = descriptor_bank
        per_sequence_local_feature_bank[resources.sequence_name] = local_feature_bank
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
            classifier=model.query_classifier,
            classifier_score_weight=dataset.config.classifier_score_weight,
            local_matcher=model.local_matcher,
            local_feature_bank=per_sequence_local_feature_bank[sequence_name],
            local_feature_level=model.local_matcher_feature_level,
            local_matcher_score_weight=dataset.config.local_matcher_score_weight,
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
    device = torch.device(cfg.device)
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=max(0, int(cfg.train_num_workers)),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(cfg.train_num_workers) > 0),
    )

    use_classifier = bool(cfg.use_patch_classification_loss or cfg.classifier_score_weight > 0.0)
    num_patch_classes = _resolve_num_patch_classes(train_dataset) if use_classifier else None

    model = CoarseRetrievalModel(
        descriptor_dim=cfg.descriptor_dim,
        init_seed=cfg.model_seed,
        backbone_variant=cfg.backbone_variant,
        share_query_patch_encoder=cfg.share_query_patch_encoder,
        num_patch_classes=num_patch_classes,
        use_local_matcher=cfg.use_local_matcher,
        local_matcher_feature_level=cfg.local_matcher_feature_level,
        local_matcher_hidden_channels=cfg.local_matcher_hidden_channels,
        local_matcher_max_shift_cells=cfg.local_matcher_max_shift_cells,
    ).to(device)
    if cfg.init_checkpoint_path:
        _load_init_checkpoint(model, cfg.init_checkpoint_path, device=device)
    _set_module_trainable(model.query_encoder, not cfg.freeze_query_encoder)
    if model.patch_encoder is model.query_encoder:
        if cfg.freeze_query_encoder or cfg.freeze_patch_encoder:
            _set_module_trainable(model.patch_encoder, False)
    else:
        _set_module_trainable(model.patch_encoder, not cfg.freeze_patch_encoder)
    criterion = RetrievalInfoNCELoss(temperature=cfg.temperature)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain after applying freeze configuration.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
    scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_decay_gamma)
        if cfg.lr_decay_gamma < 0.999999
        else None
    )

    history: list[float] = []
    validation_history: list[dict] = []
    best_metric = -1.0
    best_state: dict | None = None
    for epoch_idx in range(cfg.train_epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in dataloader:
            query_bev = batch["query_bev"].to(device).float()
            positive_patch_bev = batch["positive_patch_bev"].to(device).float()
            negative_patch_bevs = batch["negative_patch_bevs"].to(device).float()
            gt_patch_id = batch["gt_patch_id"].to(device).long()

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(query_bev, positive_patch_bev, negative_patch_bevs)
                loss = criterion(
                    outputs["query_descriptor"],
                    outputs["positive_descriptor"],
                    outputs["negative_descriptor"],
                )
                if model.query_classifier is not None and cfg.classification_loss_weight > 0.0:
                    classification_loss = F.cross_entropy(outputs["query_logits"], gt_patch_id)
                    loss = loss + cfg.classification_loss_weight * classification_loss
                if model.local_matcher is not None and cfg.local_matcher_loss_weight > 0.0:
                    local_positive_scores = model.local_matcher.score_pairs(
                        outputs["query_local_map"],
                        outputs["positive_local_map"],
                    )
                    local_negative_scores = model.local_matcher.score_pairwise(
                        outputs["query_local_map"],
                        outputs["negative_local_maps"],
                    )
                    local_logits = torch.cat(
                        (local_positive_scores[:, None], local_negative_scores),
                        dim=1,
                    ) / float(cfg.local_matcher_temperature)
                    local_labels = torch.zeros(local_logits.shape[0], dtype=torch.long, device=local_logits.device)
                    local_loss = F.cross_entropy(local_logits, local_labels)
                    loss = loss + cfg.local_matcher_loss_weight * local_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            history.append(float(loss.item()))
            epoch_losses.append(float(loss.item()))

        if scheduler is not None:
            scheduler.step()

        epoch_report: dict[str, object] = {
            "epoch": epoch_idx + 1,
            "train_loss_mean": float(np.mean(epoch_losses)) if epoch_losses else None,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }

        if (epoch_idx + 1) % cfg.eval_every_epochs == 0:
            metrics = _evaluate_recall(model, val_dataset, device=device, topk=cfg.topk)
            metrics["epoch"] = epoch_idx + 1
            validation_history.append(metrics)
            epoch_report["validation"] = metrics
            recall_at_1 = metrics["recall@1"] or 0.0
            if recall_at_1 >= best_metric:
                best_metric = recall_at_1
                best_state = {
                    "query_encoder": model.query_encoder.state_dict(),
                    "patch_encoder": model.patch_encoder.state_dict(),
                    "query_classifier": model.query_classifier.state_dict() if model.query_classifier is not None else None,
                    "local_matcher": model.local_matcher.state_dict() if model.local_matcher is not None else None,
                    "history": history.copy(),
                    "validation_history": validation_history.copy(),
                    "config": cfg.__dict__,
                }
        print(json.dumps(epoch_report))

    final_state = {
        "query_encoder": model.query_encoder.state_dict(),
        "patch_encoder": model.patch_encoder.state_dict(),
        "query_classifier": model.query_classifier.state_dict() if model.query_classifier is not None else None,
        "local_matcher": model.local_matcher.state_dict() if model.local_matcher is not None else None,
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
