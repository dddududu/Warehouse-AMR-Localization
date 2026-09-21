from __future__ import annotations

from pathlib import Path

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parent
GATE_CHECKPOINT = "outputs/semantic_stability_20260731/experiment3/reliability_risk_gate.pt"

BASELINES = {
    "oct12_aisle_ccw": EXPERIMENT_ROOT.parent / "aisle_generic_20260521" / "fine_localization_oct12_aisle_ccw_trackerinit_generic_v14a_samepatchrelease.yaml",
    "oct12_aisle_cw": EXPERIMENT_ROOT.parent / "aisle_generic_20260521" / "fine_localization_oct12_aisle_cw_trackerinit_generic_v14a_samepatchrelease.yaml",
}


def main() -> None:
    output_dir = EXPERIMENT_ROOT / "experiment3_configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, baseline_path in BASELINES.items():
        config = yaml.safe_load(baseline_path.read_text(encoding="utf-8")) or {}
        baseline_config = dict(config)
        baseline_config["output_json"] = f"outputs/semantic_stability_20260731/experiment3/{name}_baseline.json"
        (output_dir / f"blind_{name}_baseline.yaml").write_text(
            yaml.safe_dump(baseline_config, sort_keys=False),
            encoding="utf-8",
        )
        config["reliability_gate_checkpoint_path"] = GATE_CHECKPOINT
        config["reliability_gate_min_predicted_improvement_m"] = 0.03
        config["output_json"] = f"outputs/semantic_stability_20260731/experiment3/{name}_reliability_gate.json"
        (output_dir / f"blind_{name}_reliability_gate.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
