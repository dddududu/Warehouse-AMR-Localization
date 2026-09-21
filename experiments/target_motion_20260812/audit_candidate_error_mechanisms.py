# -*- coding: utf-8 -*-
"""Audit whether high localization errors come from candidate absence or selection.

This is an offline diagnostic. Ground truth is used only to label the already
frozen candidate poses; it is never used to construct a localization score.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import groupby
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from dataset_io.gt_loader import load_ground_truth


HIGH_ERROR_M = 0.20
USABLE_CANDIDATE_M = 0.20
COARSE_CANDIDATE_M = 0.50
CASES = {
    "Aisle_CCW": {
        "report": "online_blind_oct12_ccw_sidecar_deterministic_exact.json",
        "ground_truth": "D:/TorWIC/TorWIC SLAM Dataset/Oct. 12, 2022/Aisle_CCW/Aisle_CCW/traj_gt.txt",
    },
    "Aisle_CW": {
        "report": "online_blind_oct12_cw_sidecar_deterministic_exact.json",
        "ground_truth": "D:/TorWIC/TorWIC SLAM Dataset/Oct. 12, 2022/Aisle_CW/Aisle_CW/traj_gt.txt",
    },
}


def candidate_errors(frame: dict[str, Any], ground_truth_xy: np.ndarray) -> np.ndarray:
    ground_truth_xy = np.asarray(ground_truth_xy, dtype=float)[:2]
    return np.asarray(
        [
            np.linalg.norm(np.asarray(candidate["final_pose_4x4"], dtype=float)[:2, 3] - ground_truth_xy)
            for candidate in frame.get("candidate_results", [])
        ],
        dtype=float,
    )


def category_for_error(selected_error: float, oracle_error: float) -> str:
    if selected_error < HIGH_ERROR_M:
        return "normal"
    if oracle_error < USABLE_CANDIDATE_M:
        return "topk_selection_failure"
    if oracle_error < COARSE_CANDIDATE_M:
        return "topk_partial_recovery_only"
    return "topk_candidate_absence"


def contiguous_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    high_rows = [row for row in rows if row["selected_position_error_m"] >= HIGH_ERROR_M]
    segments: list[dict[str, Any]] = []
    for _, grouped in groupby(enumerate(high_rows), lambda item: item[1]["frame_idx"] - item[0]):
        members = [item[1] for item in grouped]
        categories = [str(item["failure_category"]) for item in members]
        segments.append(
            {
                "start_frame": int(members[0]["frame_idx"]),
                "end_frame": int(members[-1]["frame_idx"]),
                "frames": len(members),
                "max_selected_error_m": float(max(item["selected_position_error_m"] for item in members)),
                "mean_oracle_error_m": float(np.mean([item["oracle_top5_error_m"] for item in members])),
                "selection_failure_frames": int(sum(item == "topk_selection_failure" for item in categories)),
                "partial_recovery_frames": int(sum(item == "topk_partial_recovery_only" for item in categories)),
                "candidate_absence_frames": int(sum(item == "topk_candidate_absence" for item in categories)),
                "sidecar_applied_frames": int(sum(bool(item["semantic_filter_applied"]) for item in members)),
            }
        )
    return segments


def audit_case(name: str, report_path: Path, ground_truth_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    trajectory = load_ground_truth(ground_truth_path)
    rows: list[dict[str, Any]] = []
    for frame in report["frame_results"]:
        frame_idx = int(frame["frame_idx"])
        errors = candidate_errors(frame, trajectory.positions[frame_idx])
        selected_index = frame.get("selected_candidate_index")
        if not isinstance(selected_index, int) or not 0 <= selected_index < len(errors):
            selected_index = 0
        selected_error = float(frame["position_error_m"])
        oracle_index = int(np.argmin(errors))
        oracle_error = float(errors[oracle_index])
        selected_candidate_error = float(errors[selected_index])
        category = category_for_error(selected_error, oracle_error)
        rows.append(
            {
                "sequence": name,
                "frame_idx": frame_idx,
                "selected_patch_id": int(frame.get("best_patch_id", -1)),
                "selected_candidate_rank": selected_index + 1,
                "selected_position_error_m": selected_error,
                "selected_candidate_position_error_m": selected_candidate_error,
                "oracle_top5_error_m": oracle_error,
                "oracle_top5_rank": oracle_index + 1,
                "oracle_top5_patch_id": int(frame["candidate_results"][oracle_index]["patch_id"]),
                "failure_category": category,
                "semantic_filter_applied": bool(frame.get("semantic_filter_applied", False)),
                "semi_dynamic_point_ratio": frame.get("semantic_filter_semi_dynamic_point_ratio"),
                "semantic_occlusion_ratio": frame.get("semantic_occlusion_ratio"),
                "candidate_final_score_margin": _score_margin(frame, "final_score", descending=True),
                "candidate_icp_rmse_margin": _score_margin(frame, "icp_rmse", descending=False),
            }
        )
    selected_errors = np.asarray([row["selected_position_error_m"] for row in rows])
    oracle_errors = np.asarray([row["oracle_top5_error_m"] for row in rows])
    high_rows = [row for row in rows if row["failure_category"] != "normal"]
    summary = {
        "frames": len(rows),
        "mean_selected_error_m": float(selected_errors.mean()),
        "mean_top5_oracle_error_m": float(oracle_errors.mean()),
        "p95_selected_error_m": float(np.quantile(selected_errors, 0.95)),
        "p95_top5_oracle_error_m": float(np.quantile(oracle_errors, 0.95)),
        "high_error_frames": len(high_rows),
        "high_error_categories": {
            category: int(sum(row["failure_category"] == category for row in high_rows))
            for category in ("topk_selection_failure", "topk_partial_recovery_only", "topk_candidate_absence")
        },
        "high_error_segments": contiguous_segments(rows),
    }
    return rows, summary


def _score_margin(frame: dict[str, Any], key: str, descending: bool) -> float | None:
    values = [candidate.get(key) for candidate in frame.get("candidate_results", [])]
    values = [float(value) for value in values if value is not None]
    if len(values) < 2:
        return None
    values.sort(reverse=descending)
    return float(values[0] - values[1])


def plot_categories(summary: dict[str, Any], output_path: Path) -> None:
    names = list(summary["per_sequence"])
    labels = ["Top-5 selection failure", "Top-5 partial recovery", "Top-5 candidate absence"]
    keys = ["topk_selection_failure", "topk_partial_recovery_only", "topk_candidate_absence"]
    colors = ["#2E7D32", "#F9A825", "#C62828"]
    bottom = np.zeros(len(names))
    plt.figure(figsize=(7.0, 4.4), dpi=160)
    for label, key, color in zip(labels, keys, colors):
        values = np.asarray([summary["per_sequence"][name]["high_error_categories"][key] for name in names])
        plt.bar(names, values, bottom=bottom, label=label, color=color)
        bottom += values
    plt.ylabel("Frames with >=0.20 m error")
    plt.title("Oct.12 high-error mechanism audit")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run(report_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    report_dir = Path(report_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"thresholds_m": {"high_error": HIGH_ERROR_M, "usable": USABLE_CANDIDATE_M, "coarse": COARSE_CANDIDATE_M}, "per_sequence": {}}
    for name, config in CASES.items():
        rows, case_summary = audit_case(name, report_dir / config["report"], Path(config["ground_truth"]))
        all_rows.extend(rows)
        summary["per_sequence"][name] = case_summary
    summary["combined"] = {
        "high_error_frames": int(sum(item["high_error_frames"] for item in summary["per_sequence"].values())),
        "high_error_categories": {
            key: int(sum(item["high_error_categories"][key] for item in summary["per_sequence"].values()))
            for key in ("topk_selection_failure", "topk_partial_recovery_only", "topk_candidate_absence")
        },
    }
    with (output_dir / "oct12_candidate_error_rows.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (output_dir / "oct12_candidate_error_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_categories(summary, output_dir / "oct12_candidate_error_categories.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen candidate error mechanisms on Oct.12 Aisle.")
    parser.add_argument("--report-dir", default="outputs/semantic_class_prior_20260808")
    parser.add_argument("--output-dir", default="outputs/target_motion_20260812/candidate_error_mechanisms")
    args = parser.parse_args()
    print(json.dumps(run(args.report_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
