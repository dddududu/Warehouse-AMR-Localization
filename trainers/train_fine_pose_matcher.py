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

from dataset_io.fine_localization_dataset import DeepFineLocalizationDataset
from localization.deep_config import load_deep_fine_matcher_train_config
from models.fine_pose_matcher import FinePoseMatcher
from retrieval.config import load_coarse_retrieval_config


def _quantize_targets(
    pose_targets: torch.Tensor,
    model: FinePoseMatcher,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_targets = pose_targets[:, 0]
    y_targets = pose_targets[:, 1]
    yaw_targets = pose_targets[:, 2]
    x_distances = torch.abs(x_targets[:, None] - model.xy_bin_centers[None, :].to(pose_targets.device))
    y_distances = torch.abs(y_targets[:, None] - model.xy_bin_centers[None, :].to(pose_targets.device))
    yaw_diff = torch.atan2(
        torch.sin(yaw_targets[:, None] - model.yaw_bin_centers[None, :].to(pose_targets.device)),
        torch.cos(yaw_targets[:, None] - model.yaw_bin_centers[None, :].to(pose_targets.device)),
    ).abs()
    x_bin_idx = torch.argmin(x_distances, dim=1)
    y_bin_idx = torch.argmin(y_distances, dim=1)
    yaw_bin_idx = torch.argmin(yaw_diff, dim=1)
    x_center = model.xy_bin_centers.to(pose_targets.device)[x_bin_idx]
    y_center = model.xy_bin_centers.to(pose_targets.device)[y_bin_idx]
    yaw_center = model.yaw_bin_centers.to(pose_targets.device)[yaw_bin_idx]
    residual_targets = torch.stack(
        (
            (x_targets - x_center) / max(model.xy_bin_width * 0.5, 1.0e-6),
            (y_targets - y_center) / max(model.xy_bin_width * 0.5, 1.0e-6),
            torch.atan2(torch.sin(yaw_targets - yaw_center), torch.cos(yaw_targets - yaw_center))
            / max(model.yaw_bin_width * 0.5, 1.0e-6),
        ),
        dim=1,
    ).clamp(min=-1.0, max=1.0)
    return x_bin_idx, y_bin_idx, yaw_bin_idx, residual_targets


def _evaluate_matcher(
    model: FinePoseMatcher,
    dataset: DeepFineLocalizationDataset,
    device: torch.device,
) -> dict[str, float | int | None]:
    if len(dataset) == 0:
        return {"num_val_frames": 0, "match_accuracy": None, "pose_l1_mean": None}
    model.eval()
    correct = 0
    pose_errors: list[float] = []
    with torch.no_grad():
        for sample_idx in range(len(dataset)):
            sample = dataset[sample_idx]
            query_bev = sample["query_bev"][None].to(device).float()
            candidate_bevs = sample["candidate_bevs"][None].to(device).float()
            gt_candidate_index = int(sample["gt_candidate_index"])
            candidate_pose_targets = sample["candidate_pose_targets"][None].to(device).float()
            outputs = model(query_bev, candidate_bevs)
            if int(torch.argmax(outputs["match_logit"], dim=1).item()) == gt_candidate_index:
                correct += 1
            pose_error = torch.abs(
                outputs["pose"][:, gt_candidate_index, :].cpu()
                - candidate_pose_targets[:, gt_candidate_index, :].cpu()
            ).mean().item()
            pose_errors.append(float(pose_error))
    return {
        "num_val_frames": int(len(dataset)),
        "match_accuracy": correct / max(1, len(dataset)),
        "pose_l1_mean": float(np.mean(pose_errors)) if pose_errors else None,
    }


def train_fine_pose_matcher(config, output_checkpoint: str | Path | None = None) -> dict:
    cfg = load_deep_fine_matcher_train_config(config)
    coarse_cfg = load_coarse_retrieval_config(cfg.coarse_config_path)
    train_entries, val_entries = coarse_cfg.split_sequence_entries()
    train_dataset = DeepFineLocalizationDataset(cfg, sequence_entries=train_entries, max_samples=cfg.max_train_samples)
    val_dataset = DeepFineLocalizationDataset(cfg, sequence_entries=val_entries, max_samples=cfg.max_val_samples)
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
    model = FinePoseMatcher(
        descriptor_dim=cfg.descriptor_dim,
        hidden_dim=cfg.hidden_dim,
        local_submap_size_m=cfg.local_submap_size_m,
        num_xy_bins=cfg.num_xy_bins,
        num_yaw_bins=cfg.num_yaw_bins,
    ).to(device)
    if cfg.init_checkpoint_path:
        state = torch.load(cfg.init_checkpoint_path, map_location=device)
        if state.get("model") is not None:
            model.load_state_dict(state["model"], strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
    scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_decay_gamma)
        if cfg.lr_decay_gamma < 0.999999
        else None
    )
    validation_history: list[dict] = []
    history: list[float] = []
    best_metric = -1.0
    best_state: dict | None = None
    for epoch_idx in range(cfg.train_epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in dataloader:
            query_bev = batch["query_bev"].to(device).float()
            candidate_bevs = batch["candidate_bevs"].to(device).float()
            candidate_pose_targets = batch["candidate_pose_targets"].to(device).float()
            gt_candidate_index = batch["gt_candidate_index"].to(device).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(query_bev, candidate_bevs)
                match_loss = F.cross_entropy(outputs["match_logit"], gt_candidate_index)
                gather_index = gt_candidate_index[:, None, None].expand(-1, 1, 3)
                selected_pose = outputs["pose"].gather(1, gather_index).squeeze(1)
                selected_target = candidate_pose_targets.gather(1, gather_index).squeeze(1)
                x_bin_idx, y_bin_idx, yaw_bin_idx, residual_targets = _quantize_targets(selected_target, model)
                x_logits = outputs["x_bin_logits"].gather(
                    1,
                    gt_candidate_index[:, None, None].expand(-1, 1, outputs["x_bin_logits"].shape[-1]),
                ).squeeze(1)
                y_logits = outputs["y_bin_logits"].gather(
                    1,
                    gt_candidate_index[:, None, None].expand(-1, 1, outputs["y_bin_logits"].shape[-1]),
                ).squeeze(1)
                yaw_logits = outputs["yaw_bin_logits"].gather(
                    1,
                    gt_candidate_index[:, None, None].expand(-1, 1, outputs["yaw_bin_logits"].shape[-1]),
                ).squeeze(1)
                residual_pred = outputs["pose_residual"].gather(1, gather_index).squeeze(1)
                pose_loss = (
                    F.cross_entropy(x_logits, x_bin_idx)
                    + F.cross_entropy(y_logits, y_bin_idx)
                    + F.cross_entropy(yaw_logits, yaw_bin_idx)
                    + F.smooth_l1_loss(torch.tanh(residual_pred), residual_targets)
                )
                selected_confidence = outputs["pose_confidence"].gather(1, gt_candidate_index[:, None]).squeeze(1)
                confidence_target = torch.exp(-torch.abs(selected_pose - selected_target).mean(dim=1))
                confidence_loss = F.mse_loss(selected_confidence, confidence_target)
                loss = (
                    float(cfg.match_loss_weight) * match_loss
                    + float(cfg.pose_loss_weight) * pose_loss
                    + 0.1 * confidence_loss
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.item()))
            history.append(float(loss.item()))
        if scheduler is not None:
            scheduler.step()
        epoch_report: dict[str, object] = {
            "epoch": epoch_idx + 1,
            "train_loss_mean": float(np.mean(epoch_losses)) if epoch_losses else None,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if (epoch_idx + 1) % cfg.eval_every_epochs == 0:
            metrics = _evaluate_matcher(model, val_dataset, device=device)
            metrics["epoch"] = epoch_idx + 1
            validation_history.append(metrics)
            epoch_report["validation"] = metrics
            metric_value = float(metrics["match_accuracy"] or 0.0)
            if metric_value >= best_metric:
                best_metric = metric_value
                best_state = {
                    "model": model.state_dict(),
                    "history": history.copy(),
                    "validation_history": validation_history.copy(),
                    "config": cfg.__dict__,
                }
        print(json.dumps(epoch_report))
    final_state = {
        "model": model.state_dict(),
        "history": history,
        "validation_history": validation_history,
        "config": cfg.__dict__,
    }
    payload = best_state if (cfg.save_best_only and best_state is not None) else final_state
    checkpoint_path = Path(output_checkpoint) if output_checkpoint is not None else Path(cfg.cache_dir) / "fine_pose_matcher.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
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
    parser = argparse.ArgumentParser(description="Train the deep fine pose matcher.")
    parser.add_argument("--config", default="configs/fine_pose_matcher_train.yaml")
    parser.add_argument("--output-checkpoint", default=None)
    args = parser.parse_args()
    train_fine_pose_matcher(args.config, output_checkpoint=args.output_checkpoint)


if __name__ == "__main__":
    main()
