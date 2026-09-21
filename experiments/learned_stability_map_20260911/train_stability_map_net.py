from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as functional
from torch.utils.data import DataLoader, TensorDataset

from stability_map_learning import StabilityMapNet, build_feature_patches, heuristic_stability, regression_metrics


def _predict(model: StabilityMapNet, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), 256):
            inputs = torch.from_numpy(features[start : start + 256]).to(device)
            chunks.append(model(inputs).cpu().numpy())
    return np.concatenate(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a learned semantic stability map without Oct12 supervision.")
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = np.load(Path(config["dataset_path"]))
    features = dataset["features"].astype(np.float32)
    targets = dataset["targets"].astype(np.float32)
    train_mask = dataset["train_mask"].astype(bool)
    validation_mask = dataset["validation_mask"].astype(bool)
    device = torch.device("cuda" if torch.cuda.is_available() and bool(config.get("use_cuda", True)) else "cpu")
    model = StabilityMapNet(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features[train_mask]), torch.from_numpy(targets[train_mask])),
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )
    best_state = None
    best_loss = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses = []
        for batch_features, batch_targets in loader:
            predictions = model(batch_features.to(device))
            loss = functional.smooth_l1_loss(predictions, batch_targets.to(device), beta=0.10)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        validation_predictions = _predict(model, features[validation_mask], device)
        validation_loss = float(np.mean(np.abs(validation_predictions - targets[validation_mask])))
        history.append({"epoch": epoch, "train_huber": float(np.mean(losses)), "validation_mae": validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No model checkpoint was produced.")
    model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "input_channels": int(features.shape[1])}, output_dir / "stability_map_net.pt")
    validation_predictions = _predict(model, features[validation_mask], device)
    report = {
        "protocol": "Validation cells are held out by spatial blocks. The model sees Jun15 initial-map patches only; Jun23 supplies training targets; Oct12 remains untouched.",
        "device": str(device),
        "best_validation_mae": best_loss,
        "learned_validation": regression_metrics(validation_predictions, targets[validation_mask]),
        "heuristic_validation": regression_metrics(dataset["heuristic_stability"][validation_mask], targets[validation_mask]),
        "epochs": int(config["epochs"]),
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    source_map_path = Path(config["source_map_path"])
    with np.load(source_map_path) as source:
        initial_cells = source["initial_cells"].astype(np.int32)
        initial_counts = source["initial_counts"].astype(np.int32)
        all_features = build_feature_patches(initial_cells, initial_counts, int(config["patch_radius_cells"]))
        learned_stability = _predict(model, all_features, device).astype(np.float32)
        np.savez_compressed(
            output_dir / "learned_stability_map.npz",
            cell_size_m=source["cell_size_m"],
            cells=initial_cells,
            counts=initial_counts,
            learned_stability=learned_stability,
            heuristic_stability=heuristic_stability(initial_counts),
            label_cells=source["initial_label_cells"].astype(np.int32),
            label_counts=source["initial_label_counts"].astype(np.int32),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
