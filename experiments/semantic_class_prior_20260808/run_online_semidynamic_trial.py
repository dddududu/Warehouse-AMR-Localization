from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from localization.config import FineLocalizationConfig
from localization.deep_fine_localizer import localize_sequence


TRIAL_MODES = ("original", "dynamic_only", "adaptive_semidynamic")


def _load_mapping(path: str | Path) -> dict[str, Any]:
    mapping = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(mapping, dict):
        raise ValueError("Fine localization configuration must be a mapping.")
    return dict(mapping)


def build_trial_config(
    base_config_path: str | Path,
    mode: str,
    output_json: str | Path,
) -> FineLocalizationConfig:
    if mode not in TRIAL_MODES:
        raise ValueError(f"Unknown trial mode: {mode}.")
    mapping = _load_mapping(base_config_path)
    mapping["output_json"] = str(output_json)
    mapping["use_semantic_dynamic_filter"] = mode != "original"
    mapping["use_semidynamic_filter_trigger"] = mode == "adaptive_semidynamic"
    mapping["semantic_filter_apply_dynamic_filter_without_trigger"] = mode == "dynamic_only"
    mapping["use_semantic_dual_geometry_gate"] = mode == "adaptive_semidynamic"
    mapping["use_semantic_shadow_tracker"] = mode == "adaptive_semidynamic"
    mapping["semantic_dynamic_labels"] = [12, 13, 14, 15]
    mapping["semantic_semi_dynamic_labels"] = [5, 7, 9, 10, 11]
    mapping["semantic_semi_dynamic_ratio_threshold"] = 0.10
    mapping["semantic_dynamic_mask_dilation_px"] = 3
    return FineLocalizationConfig.from_mapping(mapping)


def _summary(report: dict[str, Any], mode: str, base_config_path: str | Path) -> dict[str, Any]:
    errors = np.asarray(
        [float(row["position_error_m"]) for row in report["frame_results"]],
        dtype=np.float64,
    )
    conservative = [
        bool(row.get("semantic_filter_conservative_enabled", False))
        for row in report["frame_results"]
    ]
    exposures = [
        row.get("semantic_filter_semi_dynamic_point_ratio")
        for row in report["frame_results"]
    ]
    exposures = [float(value) for value in exposures if value is not None]
    return {
        "mode": mode,
        "base_config_path": str(base_config_path),
        "num_frames": int(errors.size),
        "mean_position_error_m": float(errors.mean()) if errors.size else None,
        "median_position_error_m": float(np.median(errors)) if errors.size else None,
        "p95_position_error_m": float(np.quantile(errors, 0.95)) if errors.size else None,
        "frames_below_0p5m_ratio": float(np.mean(errors < 0.5)) if errors.size else None,
        "conservative_filter_ratio": float(np.mean(conservative)) if conservative else 0.0,
        "semi_dynamic_point_ratio_percentiles": (
            np.quantile(exposures, [0.1, 0.5, 0.9]).tolist() if exposures else None
        ),
    }


def run_trial(
    base_config_path: str | Path,
    mode: str,
    output_json: str | Path,
    frame_start: int = 0,
    num_frames: int | None = None,
    frame_stride: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    config = build_trial_config(base_config_path, mode, output_json)
    with contextlib.redirect_stdout(io.StringIO()):
        report = localize_sequence(
            config,
            frame_start=frame_start,
            num_frames=num_frames,
            frame_stride=frame_stride,
            output_json=output_json,
            resume=resume,
        )
    summary = _summary(report, mode, base_config_path)
    summary_path = Path(output_json).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a causal online semantic-filter localization trial.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--mode", choices=TRIAL_MODES, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run_trial(
        args.base_config,
        args.mode,
        args.output_json,
        frame_start=args.frame_start,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
