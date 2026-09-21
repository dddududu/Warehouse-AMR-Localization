from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from stability_map_learning import build_learning_examples, spatial_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Jun15-to-Jun23 spatially separated stability-learning dataset.")
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(config["source_map_path"])
    with np.load(source_path) as payload:
        examples = build_learning_examples(
            payload,
            radius=int(config["patch_radius_cells"]),
            minimum_initial_observations=int(config["minimum_initial_observations"]),
            minimum_crossday_observations=int(config["minimum_crossday_observations"]),
        )
    train_mask, validation_mask = spatial_split(
        examples.cells,
        block_size_cells=int(config["spatial_block_size_cells"]),
        validation_group=int(config["validation_group"]),
    )
    dataset_path = output_dir / "jun15_to_jun23_stability_dataset.npz"
    np.savez_compressed(
        dataset_path,
        cells=examples.cells,
        features=examples.features,
        targets=examples.targets,
        heuristic_stability=examples.heuristic_stability,
        crossday_observations=examples.crossday_observations,
        train_mask=train_mask,
        validation_mask=validation_mask,
    )
    summary = {
        "source_map_path": str(source_path),
        "dataset_path": str(dataset_path),
        "protocol": "Features use Jun15 initial-map evidence only. Targets are derived from the incremental Jun23 observations only. Oct12 data is not read by this stage.",
        "samples": int(len(examples.cells)),
        "train_samples": int(train_mask.sum()),
        "validation_samples": int(validation_mask.sum()),
        "feature_shape": list(examples.features.shape[1:]),
        "target_quantiles": [float(value) for value in np.quantile(examples.targets, [0.0, 0.25, 0.5, 0.75, 1.0])],
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
