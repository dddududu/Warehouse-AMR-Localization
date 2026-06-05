from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


PERSON_LABEL = 13
DYNAMIC_LABELS = (12, 13, 14, 15)


def _load_result_frames(result_path: Path) -> list[dict]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    frames = payload.get("frame_results")
    if not isinstance(frames, list):
        raise ValueError(f"frame_results is missing from {result_path}")
    return frames


def _read_label_ratio(path: Path, labels: tuple[int, ...]) -> float:
    label_image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label_image is None:
        raise FileNotFoundError(path)
    return float(np.isin(label_image, labels).mean())


def _read_blur_score(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    resized = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(resized, cv2.CV_64F).var())


def _selected_candidate(frame: dict) -> dict:
    candidates = frame.get("candidate_results") or []
    selected_index = frame.get("selected_candidate_index")
    if isinstance(selected_index, int) and 0 <= selected_index < len(candidates):
        return candidates[selected_index]
    patch_id = frame.get("best_patch_id")
    for candidate in candidates:
        if candidate.get("patch_id") == patch_id:
            return candidate
    return {}


def _safe_correlation(left: list[float], right: list[float]) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.size < 2 or np.std(left_array) < 1.0e-12 or np.std(right_array) < 1.0e-12:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _find_runs(rows: list[dict], error_threshold_m: float) -> list[dict]:
    runs: list[list[dict]] = []
    current: list[dict] = []
    for row in rows:
        if row["position_error_m"] >= error_threshold_m:
            if current and row["frame_idx"] != current[-1]["frame_idx"] + 1:
                runs.append(current)
                current = []
            current.append(row)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    summaries = []
    for run in runs:
        peak = max(run, key=lambda item: item["position_error_m"])
        summaries.append(
            {
                "frame_start": run[0]["frame_idx"],
                "frame_end": run[-1]["frame_idx"],
                "num_frames": len(run),
                "peak_frame_idx": peak["frame_idx"],
                "peak_position_error_m": peak["position_error_m"],
                "max_person_ratio": max(item["person_ratio"] for item in run),
                "max_dynamic_ratio": max(item["dynamic_ratio"] for item in run),
                "min_blur_score": min(item["blur_score"] for item in run),
            }
        )
    return sorted(summaries, key=lambda item: item["peak_position_error_m"], reverse=True)


def analyze_sequence(
    sequence_name: str,
    sequence_root: Path,
    result_path: Path,
    person_ratio_threshold: float,
    error_threshold_m: float,
) -> dict:
    frame_results = _load_result_frames(result_path)
    rows: list[dict] = []
    for frame in frame_results:
        frame_idx = int(frame["frame_idx"])
        stem = f"{frame_idx:06d}.png"
        person_left = _read_label_ratio(
            sequence_root / "segmentation_greyscale_left" / stem,
            (PERSON_LABEL,),
        )
        person_right = _read_label_ratio(
            sequence_root / "segmentation_greyscale_right" / stem,
            (PERSON_LABEL,),
        )
        dynamic_left = _read_label_ratio(
            sequence_root / "segmentation_greyscale_left" / stem,
            DYNAMIC_LABELS,
        )
        dynamic_right = _read_label_ratio(
            sequence_root / "segmentation_greyscale_right" / stem,
            DYNAMIC_LABELS,
        )
        blur_left = _read_blur_score(sequence_root / "image_left" / stem)
        blur_right = _read_blur_score(sequence_root / "image_right" / stem)
        candidate = _selected_candidate(frame)
        rows.append(
            {
                "sequence": sequence_name,
                "frame_idx": frame_idx,
                "position_error_m": float(frame["position_error_m"]),
                "yaw_error_deg": float(frame["yaw_error_deg"]),
                "best_patch_id": int(frame["best_patch_id"]),
                "selection_mode": str(frame.get("selection_mode", "")),
                "selected_init_source": str(candidate.get("selected_init_source", "")),
                "person_ratio": max(person_left, person_right),
                "person_ratio_mean": 0.5 * (person_left + person_right),
                "dynamic_ratio": max(dynamic_left, dynamic_right),
                "blur_score": min(blur_left, blur_right),
                "icp_inlier_ratio": candidate.get("icp_inlier_ratio"),
                "icp_rmse": candidate.get("icp_rmse"),
                "deep_match_probability": candidate.get("deep_match_probability"),
                "selected_bev_score": candidate.get("selected_bev_score"),
            }
        )

    errors = [row["position_error_m"] for row in rows]
    person_ratios = [row["person_ratio"] for row in rows]
    dynamic_ratios = [row["dynamic_ratio"] for row in rows]
    inverse_blur = [1.0 / max(row["blur_score"], 1.0e-6) for row in rows]
    person_frames = [row for row in rows if row["person_ratio"] >= person_ratio_threshold]
    non_person_frames = [row for row in rows if row["person_ratio"] < person_ratio_threshold]
    spike_frames = [row for row in rows if row["position_error_m"] >= error_threshold_m]
    person_spikes = [row for row in spike_frames if row["person_ratio"] >= person_ratio_threshold]

    def mean_error(items: list[dict]) -> float | None:
        if not items:
            return None
        return float(np.mean([item["position_error_m"] for item in items]))

    return {
        "sequence": sequence_name,
        "sequence_root": str(sequence_root),
        "result_path": str(result_path),
        "num_frames": len(rows),
        "person_ratio_threshold": person_ratio_threshold,
        "error_threshold_m": error_threshold_m,
        "num_person_frames": len(person_frames),
        "num_spike_frames": len(spike_frames),
        "num_person_spike_frames": len(person_spikes),
        "mean_error_person_frames_m": mean_error(person_frames),
        "mean_error_non_person_frames_m": mean_error(non_person_frames),
        "correlation_error_person_ratio": _safe_correlation(errors, person_ratios),
        "correlation_error_dynamic_ratio": _safe_correlation(errors, dynamic_ratios),
        "correlation_error_inverse_blur": _safe_correlation(errors, inverse_blur),
        "top_error_frames": sorted(rows, key=lambda item: item["position_error_m"], reverse=True)[:30],
        "top_person_frames": sorted(rows, key=lambda item: item["person_ratio"], reverse=True)[:30],
        "spike_runs": _find_runs(rows, error_threshold_m),
        "rows": rows,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Aisle localization spikes and dynamic-object evidence.")
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--sequence-root", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--person-ratio-threshold", type=float, default=0.001)
    parser.add_argument("--error-threshold-m", type=float, default=0.5)
    args = parser.parse_args()

    report = analyze_sequence(
        sequence_name=args.sequence_name,
        sequence_root=Path(args.sequence_root),
        result_path=Path(args.result_json),
        person_ratio_threshold=float(args.person_ratio_threshold),
        error_threshold_m=float(args.error_threshold_m),
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(Path(args.output_csv), report["rows"])
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
