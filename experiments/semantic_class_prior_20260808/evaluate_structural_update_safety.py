from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from experiments.semantic_stability_20260731.evaluate_semantic_update_safety import (
    _collect_cell_observations,
    _future_confirmed,
    _load_yaml,
    _map_lookup,
    _sequence_configs,
)
from preprocess.local_lidar_cropper import LocalCropConfig


def _accepts(
    policy: str,
    current: Any,
    map_info: dict[str, float | int] | None,
    structural_labels: set[int],
    static_labels: set[int],
    minimum_support: int,
    minimum_stability: float,
    minimum_evidence: int,
) -> bool:
    current_label = int(current[1:].argmax())
    if policy == "naive":
        return True
    if current_label not in static_labels:
        return False
    if policy == "current_static":
        return True
    if map_info is None or int(map_info["dominant_label"]) != current_label:
        return False
    if float(map_info["stability"]) < minimum_stability or int(map_info["evidence"]) < minimum_evidence:
        return False
    if policy == "persistent_static":
        return int(current[0]) >= minimum_support
    return current_label in structural_labels and int(current[0]) >= minimum_support


def _evaluate(
    current_store: dict[tuple[int, int], Any],
    future_store: dict[tuple[int, int], Any],
    map_store: dict[tuple[int, int], dict[str, float | int]],
    class_prior: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    structural_labels = {int(label) for label in class_prior["groups"]["structural_static"]}
    static_labels = structural_labels | {int(label) for label in class_prior["groups"]["support_surface"]}
    minimum_support = int(config["min_current_frame_support"])
    minimum_stability = float(config["min_stability"])
    minimum_evidence = int(config["min_map_evidence"])
    outcomes: dict[str, Any] = {}
    for policy in ("naive", "current_static", "persistent_static", "structural_persistent"):
        accepted = [
            (cell, value)
            for cell, value in current_store.items()
            if _accepts(policy, value, map_store.get(cell), structural_labels, static_labels, minimum_support, minimum_stability, minimum_evidence)
        ]
        confirmations = [(value, _future_confirmed(value, future_store.get(cell))) for cell, value in accepted]
        observed = [(value, confirmed) for value, confirmed in confirmations if confirmed is not None]
        labels = Counter(int(value[1:].argmax()) for value, _ in observed)
        confirmed_labels = Counter(int(value[1:].argmax()) for value, confirmed in observed if confirmed)
        outcomes[policy] = {
            "accepted_cells": len(accepted),
            "coverage_over_current_cells": len(accepted) / max(len(current_store), 1),
            "future_observed_cells": len(observed),
            "future_confirmed_safe_cells": sum(confirmed for _, confirmed in observed),
            "future_confirmation_rate": sum(confirmed for _, confirmed in observed) / max(len(observed), 1),
            "future_contradiction_rate": sum(not confirmed for _, confirmed in observed) / max(len(observed), 1),
            "future_observed_by_label": {str(label): count for label, count in sorted(labels.items())},
            "future_confirmed_by_label": {str(label): confirmed_labels[label] for label in sorted(labels)},
        }
    return outcomes


def run(config_path: str | Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    class_prior = yaml.safe_load(Path(config["class_prior_path"]).read_text(encoding="utf-8")) or {}
    crop = LocalCropConfig(**{key: float(value) for key, value in config["crop"].items()})
    current_store = _collect_cell_observations(_sequence_configs(config["current_sequences"]), crop, float(config["cell_size_m"]), int(config["frame_stride"]))
    future_store = _collect_cell_observations(_sequence_configs(config["future_sequences"]), crop, float(config["cell_size_m"]), int(config["frame_stride"]))
    policies = _evaluate(current_store, future_store, _map_lookup(Path(config["stability_map_path"])), class_prior, config)
    summary = {
        "current_cells": len(current_store),
        "future_cells": len(future_store),
        "class_prior": class_prior,
        "policies": policies,
        "decision_data": "Jun.15/Jun.23 only",
        "future_confirmation_data": "Oct.12 Aisle only",
    }
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "structural_update_safety_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate class-prior structural map-update safety.")
    parser.add_argument("--config", default="experiments/semantic_class_prior_20260808/structural_update_safety.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
