# -*- coding: utf-8 -*-
"""Train and audit a read-only, cross-date localization risk monitor.

The monitor deliberately consumes only diagnostics that exist after the current
frame has been localized.  It never reads pose error at inference time and it
does not modify the tracker or the selected patch.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline


REPORTS = {
    "jun15_ccw": "sidecar_gain_gate_train_jun15_ccw_full.json",
    "jun15_cw": "sidecar_gain_gate_train_jun15_cw_full.json",
    "jun23_ccw": "sidecar_gain_gate_train_jun23_ccw_full.json",
    "jun23_cw": "sidecar_gain_gate_train_jun23_cw_full.json",
    "oct12_ccw": "online_blind_oct12_ccw_sidecar_deterministic_exact.json",
    "oct12_cw": "online_blind_oct12_cw_sidecar_deterministic_exact.json",
}

SELECTED_FEATURES = [
    "coarse_score",
    "selected_bev_score",
    "deep_match_probability",
    "deep_pose_confidence",
    "icp_inlier_ratio",
    "icp_rmse",
    "final_score",
    "temporal_position_jump_m",
    "temporal_yaw_jump_deg",
    "temporal_score",
    "semantic_occlusion_ratio",
    "semantic_filter_semi_dynamic_point_ratio",
    "semantic_filter_applied",
    "semantic_filter_conservative_enabled",
    "final_score_margin",
    "deep_match_margin",
    "icp_rmse_margin",
    "icp_inlier_margin",
]
BASE_CANDIDATE_FEATURES = SELECTED_FEATURES[:10]
MARGIN_FEATURES = {
    "final_score_margin": ("final_score", True),
    "deep_match_margin": ("deep_match_probability", True),
    "icp_rmse_margin": ("icp_rmse", False),
    "icp_inlier_margin": ("icp_inlier_ratio", True),
}
HIGH_ERROR_METERS = 0.2


def as_float(value: Any) -> float:
    try:
        return float(value) if value is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def select_candidate(frame: dict[str, Any]) -> dict[str, Any]:
    candidates = frame.get("candidate_results") or []
    index = frame.get("selected_candidate_index")
    if not isinstance(index, int) or not 0 <= index < len(candidates):
        index = 0
    return candidates[index] if candidates else {}


def extract_deployment_features(frame: dict[str, Any]) -> dict[str, float]:
    """Build features without reading `position_error_m` or other GT fields."""
    selected = select_candidate(frame)
    values = {key: as_float(selected.get(key)) for key in BASE_CANDIDATE_FEATURES}
    values.update(
        {
            "semantic_occlusion_ratio": as_float(frame.get("semantic_occlusion_ratio")),
            "semantic_filter_semi_dynamic_point_ratio": as_float(
                frame.get("semantic_filter_semi_dynamic_point_ratio")
            ),
            "semantic_filter_applied": float(bool(frame.get("semantic_filter_applied"))),
            "semantic_filter_conservative_enabled": float(
                bool(frame.get("semantic_filter_conservative_enabled"))
            ),
        }
    )
    candidates = frame.get("candidate_results") or []
    for output_name, (candidate_name, descending) in MARGIN_FEATURES.items():
        candidate_values = [as_float(item.get(candidate_name)) for item in candidates]
        candidate_values = [item for item in candidate_values if np.isfinite(item)]
        candidate_values.sort(reverse=descending)
        values[output_name] = (
            candidate_values[0] - candidate_values[1] if len(candidate_values) >= 2 else np.nan
        )
    return values


def load_report(report_path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    frames = payload["frame_results"]
    matrix = np.asarray(
        [[extract_deployment_features(frame)[name] for name in SELECTED_FEATURES] for frame in frames],
        dtype=float,
    )
    errors = np.asarray([as_float(frame.get("position_error_m")) for frame in frames], dtype=float)
    return matrix, errors, frames


def metrics_at_alarm_budget(scores: np.ndarray, labels: np.ndarray, fraction: float) -> dict[str, float | int]:
    alarm_count = max(1, round(len(scores) * fraction))
    alarm_indices = np.argsort(-scores)[:alarm_count]
    true_positives = int(labels[alarm_indices].sum())
    positives = int(labels.sum())
    return {
        "alarm_fraction": fraction,
        "alarm_count": alarm_count,
        "true_positives": true_positives,
        "recall": true_positives / positives if positives else 0.0,
        "precision": true_positives / alarm_count,
    }


def evaluate_scores(scores: np.ndarray, errors: np.ndarray) -> dict[str, Any]:
    labels = errors >= HIGH_ERROR_METERS
    return {
        "frames": int(len(labels)),
        "high_error_frames": int(labels.sum()),
        "mean_position_error_m": float(np.mean(errors)),
        "p95_position_error_m": float(np.quantile(errors, 0.95)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "budget_metrics": [metrics_at_alarm_budget(scores, labels, fraction) for fraction in (0.02, 0.05, 0.10)],
    }


def write_alarm_csv(output_path: Path, frames: list[dict[str, Any]], scores: np.ndarray, fraction: float) -> None:
    count = max(1, round(len(scores) * fraction))
    alarm_indices = np.argsort(-scores)[:count]
    fieldnames = ["frame_idx", "timestamp", "best_patch_id", "risk_score", *SELECTED_FEATURES]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in alarm_indices:
            features = extract_deployment_features(frames[int(index)])
            writer.writerow(
                {
                    "frame_idx": frames[int(index)].get("frame_idx"),
                    "timestamp": frames[int(index)].get("timestamp"),
                    "best_patch_id": frames[int(index)].get("best_patch_id"),
                    "risk_score": f"{scores[int(index)]:.6f}",
                    **{name: features[name] for name in SELECTED_FEATURES},
                }
            )


def plot_validation(scores: np.ndarray, errors: np.ndarray, output_path: Path) -> None:
    order = np.argsort(-scores)
    labels = (errors >= HIGH_ERROR_METERS)[order]
    recalls = np.cumsum(labels) / max(1, int(labels.sum()))
    fractions = (np.arange(len(scores)) + 1) / len(scores)
    plt.figure(figsize=(7.2, 4.2), dpi=160)
    plt.plot(fractions * 100.0, recalls * 100.0, color="#1976D2", linewidth=2)
    for budget in (2, 5, 10):
        plt.axvline(budget, color="#9E9E9E", linewidth=0.8, linestyle="--")
    plt.xlim(0, 20)
    plt.ylim(0, 100)
    plt.xlabel("Alarm budget (% of frames)")
    plt.ylabel("Recall of >=0.20 m errors (%)")
    plt.title("Jun.15-trained risk monitor on Jun.23 validation")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run(report_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    report_dir = Path(report_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = {name: load_report(report_dir / filename) for name, filename in REPORTS.items()}
    train_x = np.vstack([loaded["jun15_ccw"][0], loaded["jun15_cw"][0]])
    train_y = np.concatenate(
        [loaded["jun15_ccw"][1] >= HIGH_ERROR_METERS, loaded["jun15_cw"][1] >= HIGH_ERROR_METERS]
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=8,
            max_features=0.7,
            class_weight="balanced_subsample",
            random_state=0,
            n_jobs=-1,
        ),
    )
    model.fit(train_x, train_y)
    result: dict[str, Any] = {
        "protocol": {
            "training": "Jun.15 Aisle CW + CCW",
            "validation": "Jun.23 Aisle CW + CCW",
            "frozen_blind_audit": "Oct.12 Aisle CW + CCW",
            "risk_label_for_offline_evaluation": f"position error >= {HIGH_ERROR_METERS:.2f} m",
            "inference_guard": "The model receives no pose error or ground-truth information and only emits a read-only alarm.",
        },
        "feature_names": SELECTED_FEATURES,
    }
    validation_scores: list[np.ndarray] = []
    validation_errors: list[np.ndarray] = []
    for split in ("jun23_ccw", "jun23_cw"):
        scores = model.predict_proba(loaded[split][0])[:, 1]
        result[split] = evaluate_scores(scores, loaded[split][1])
        validation_scores.append(scores)
        validation_errors.append(loaded[split][1])
    all_validation_scores = np.concatenate(validation_scores)
    all_validation_errors = np.concatenate(validation_errors)
    result["jun23_all"] = evaluate_scores(all_validation_scores, all_validation_errors)
    plot_validation(all_validation_scores, all_validation_errors, output_dir / "jun23_alarm_budget_recall.png")

    for split in ("oct12_ccw", "oct12_cw"):
        scores = model.predict_proba(loaded[split][0])[:, 1]
        result[split] = evaluate_scores(scores, loaded[split][1])
        write_alarm_csv(output_dir / f"{split}_top5pct_alarms.csv", loaded[split][2], scores, 0.05)

    importances = model[-1].feature_importances_
    result["feature_importance"] = [
        {"feature": name, "importance": float(importance)}
        for name, importance in sorted(zip(SELECTED_FEATURES, importances), key=lambda item: -item[1])
    ]
    (output_dir / "risk_monitor_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a read-only localization risk monitor.")
    parser.add_argument("--report-dir", default="outputs/semantic_class_prior_20260808")
    parser.add_argument("--output-dir", default="outputs/target_motion_20260812/localization_risk_monitor")
    args = parser.parse_args()
    result = run(args.report_dir, args.output_dir)
    print(json.dumps({name: result[name] for name in ("jun23_all", "oct12_ccw", "oct12_cw")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
