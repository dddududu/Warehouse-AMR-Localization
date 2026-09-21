from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from experiments.semantic_stability_20260731.train_reliability_gate import (
    _available_mask,
    _load_yaml,
    _metrics,
    _sources,
    collect_samples,
)
from models.reliability_gate import RELIABILITY_GATE_SOURCES, ReliabilityGate


def _risk_targets(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    targets = np.zeros((len(samples), len(RELIABILITY_GATE_SOURCES)), dtype=np.float32)
    importance = np.zeros_like(targets)
    for row_index, sample in enumerate(samples):
        for source_index, source in enumerate(RELIABILITY_GATE_SOURCES):
            error_key = f"{source.split('_')[0]}_error_m" if source != "tracker_init" else "tracker_error_m"
            error = sample.get(error_key)
            if error is None:
                continue
            error = float(error)
            targets[row_index, source_index] = np.log1p(error)
            importance[row_index, source_index] = 1.0 + min(error, 2.0)
    return targets, importance


def _choose_actions(
    predicted_risks: np.ndarray,
    samples: list[dict[str, Any]],
    min_predicted_improvement_m: float,
) -> tuple[list[str], np.ndarray]:
    actions: list[str] = []
    predicted_improvements: list[float] = []
    for risks, sample in zip(predicted_risks, samples, strict=True):
        available_indices = [
            source_index
            for source_index, source in enumerate(RELIABILITY_GATE_SOURCES)
            if source in sample["available_sources"]
        ]
        baseline_source = str(sample["baseline_source"])
        baseline_index = RELIABILITY_GATE_SOURCES.index(baseline_source)
        candidate_index = min(available_indices, key=lambda source_index: float(risks[source_index]))
        improvement = float(risks[baseline_index] - risks[candidate_index])
        if (
            candidate_index != baseline_index
            and improvement >= float(min_predicted_improvement_m)
        ):
            actions.append(RELIABILITY_GATE_SOURCES[candidate_index])
        else:
            actions.append(baseline_source)
        predicted_improvements.append(improvement)
    return actions, np.asarray(predicted_improvements, dtype=np.float64)


def _evaluate_margin_scan(
    predicted_risks: np.ndarray,
    samples: list[dict[str, Any]],
    margins: list[float],
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    scan: list[dict[str, Any]] = []
    for margin in margins:
        actions, improvements = _choose_actions(predicted_risks, samples, margin)
        metrics = _metrics(samples, actions)
        metrics["min_predicted_improvement_m"] = float(margin)
        metrics["gate_apply_rate"] = float(
            np.mean(np.asarray(actions) != np.asarray([sample["baseline_source"] for sample in samples]))
        )
        metrics["mean_predicted_improvement_m"] = float(np.mean(improvements))
        scan.append(metrics)
    baseline_mean = float(scan[0]["baseline_mean_position_error_m"])
    non_degrading = [
        item for item in scan
        if float(item["mean_position_error_m"]) <= baseline_mean + 1.0e-9
    ]
    best = min(
        non_degrading or scan,
        key=lambda item: (float(item["mean_position_error_m"]), -float(item["gate_apply_rate"])),
    )
    return float(best["min_predicted_improvement_m"]), best, scan


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a risk-regression gate for fine-localization initializations.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config: dict[str, Any] = _load_yaml(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(config.get("device", "cuda"))
    margin = float(config.get("keep_baseline_margin_m", 0.03))
    include_semantic_features = bool(config.get("include_semantic_features", True))
    train_samples = collect_samples(
        _sources(config["train_sources"]),
        margin,
        include_semantic_features=include_semantic_features,
    )
    val_samples = collect_samples(
        _sources(config["val_sources"]),
        margin,
        include_semantic_features=include_semantic_features,
    )
    train_features = np.stack([sample["features"] for sample in train_samples]).astype(np.float32)
    val_features = np.stack([sample["features"] for sample in val_samples]).astype(np.float32)
    feature_mean = train_features.mean(axis=0)
    feature_std = np.maximum(train_features.std(axis=0), 1.0e-6)
    train_targets, train_importance = _risk_targets(train_samples)
    val_targets, val_importance = _risk_targets(val_samples)
    train_masks = _available_mask(train_samples)
    val_masks = _available_mask(val_samples)
    model = ReliabilityGate(
        feature_dim=train_features.shape[1],
        hidden_dim=int(config.get("hidden_dim", 32)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.001)),
        weight_decay=float(config.get("weight_decay", 1.0e-4)),
    )
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy((train_features - feature_mean) / feature_std),
            torch.from_numpy(train_targets),
            torch.from_numpy(train_masks),
            torch.from_numpy(train_importance),
        ),
        batch_size=int(config.get("batch_size", 64)),
        shuffle=True,
    )
    val_features_tensor = torch.from_numpy((val_features - feature_mean) / feature_std).to(device)
    val_targets_tensor = torch.from_numpy(val_targets).to(device)
    val_masks_tensor = torch.from_numpy(val_masks).to(device)
    val_importance_tensor = torch.from_numpy(val_importance).to(device)
    best_state = None
    best_epoch = 0
    best_val_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(config.get("epochs", 300)) + 1):
        model.train()
        train_losses = []
        for features, targets, availability, importance in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss_matrix = torch.nn.functional.smooth_l1_loss(
                model(features.to(device)),
                targets.to(device),
                reduction="none",
            )
            weighted_mask = availability.to(device) * importance.to(device)
            loss = (loss_matrix * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss_matrix = torch.nn.functional.smooth_l1_loss(
                model(val_features_tensor),
                val_targets_tensor,
                reduction="none",
            )
            weighted_mask = val_masks_tensor * val_importance_tensor
            validation_loss = float(
                ((validation_loss_matrix * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)).cpu()
            )
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(train_losses)), "val_loss": validation_loss})
        if validation_loss < best_val_loss - 1.0e-6:
            best_val_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= int(config.get("early_stop_patience", 40)):
                break
    if best_state is None:
        raise RuntimeError("Risk gate did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_risks = np.maximum(
            np.expm1(model(val_features_tensor).cpu().numpy().astype(np.float64)),
            0.0,
        )
    selected_margin, validation, margin_scan = _evaluate_margin_scan(
        val_risks,
        val_samples,
        [float(value) for value in config.get("margin_scan_m", [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12])],
    )
    train_actions, _ = _choose_actions(
        np.maximum(
            np.expm1(
                model(torch.from_numpy((train_features - feature_mean) / feature_std).to(device))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            ),
            0.0,
        ),
        train_samples,
        selected_margin,
    )
    training = _metrics(train_samples, train_actions)
    training["gate_apply_rate"] = float(
        np.mean(np.asarray(train_actions) != np.asarray([sample["baseline_source"] for sample in train_samples]))
    )
    output_stem = str(config.get("output_stem", "reliability_risk_gate"))
    checkpoint_path = output_dir / f"{output_stem}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "feature_dim": int(train_features.shape[1]),
                "hidden_dim": int(config.get("hidden_dim", 32)),
                "objective": "risk_regression",
            },
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "min_predicted_improvement_m": selected_margin,
        },
        checkpoint_path,
    )
    summary = {
        "config_path": str(Path(args.config)),
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "selected_min_predicted_improvement_m": selected_margin,
        "training": training,
        "validation": validation,
        "margin_scan": margin_scan,
        "history": history,
    }
    (output_dir / f"{output_stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
