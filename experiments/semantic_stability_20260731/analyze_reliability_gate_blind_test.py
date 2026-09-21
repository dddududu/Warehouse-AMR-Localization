from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _metrics(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean_position_error_m": float(errors.mean()),
        "median_position_error_m": float(np.median(errors)),
        "p95_position_error_m": float(np.percentile(errors, 95)),
        "max_position_error_m": float(errors.max()),
        "below_0p5m": float(np.mean(errors < 0.5)),
    }


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    frames = report.get("frame_results", [])
    if not frames:
        raise ValueError(f"No frame results in {path}.")
    return {int(frame["frame_idx"]): frame for frame in frames}


def _selected_candidate(frame: dict[str, Any]) -> dict[str, Any] | None:
    candidates = frame.get("candidate_results", [])
    index = int(frame.get("selected_candidate_index", 0))
    if not 0 <= index < len(candidates):
        return None
    return candidates[index]


def _compare_sequence(baseline_path: Path, gate_path: Path) -> dict[str, Any]:
    baseline = _load_report(baseline_path)
    gate = _load_report(gate_path)
    common_indices = sorted(set(baseline).intersection(gate))
    baseline_errors = np.asarray([baseline[index]["position_error_m"] for index in common_indices], dtype=np.float64)
    gate_errors = np.asarray([gate[index]["position_error_m"] for index in common_indices], dtype=np.float64)
    applied = []
    source_counts: dict[str, int] = {}
    rows = []
    for index, baseline_error, gate_error in zip(common_indices, baseline_errors, gate_errors, strict=True):
        candidate = _selected_candidate(gate[index])
        was_applied = bool(candidate and candidate.get("reliability_gate_applied"))
        applied.append(was_applied)
        source = str(candidate.get("selected_init_source", "unknown")) if candidate else "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1
        rows.append(
            {
                "frame_idx": int(index),
                "baseline_error_m": float(baseline_error),
                "gate_error_m": float(gate_error),
                "delta_m": float(baseline_error - gate_error),
                "gate_applied": was_applied,
                "selected_init_source": source,
            }
        )
    return {
        "baseline": _metrics(baseline_errors),
        "reliability_gate": _metrics(gate_errors),
        "mean_improvement_m": float(baseline_errors.mean() - gate_errors.mean()),
        "median_improvement_m": float(np.median(baseline_errors - gate_errors)),
        "improved_frame_ratio": float(np.mean(gate_errors < baseline_errors)),
        "gate_apply_rate": float(np.mean(applied)),
        "selected_source_counts": source_counts,
        "rows": rows,
    }


def _draw_comparison(output_path: Path, result: dict[str, Any]) -> None:
    canvas = Image.new("RGB", (1540, 940), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(30, True)
    body_font = _font(21)
    small_font = _font(17)
    draw.text((48, 30), "实验三：学习式初始化可靠性门控——十月盲测（每 8 帧）", font=title_font, fill="#1d2939")
    colors = {"baseline": "#64748b", "reliability_gate": "#2563eb"}
    panels = [(60, 115, "Aisle_CCW"), (800, 115, "Aisle_CW")]
    for left, top, sequence_name in panels:
        sequence = result["per_sequence"][sequence_name]
        rows = sequence["rows"]
        width, height = 650, 470
        draw.rounded_rectangle((left, top, left + width, top + height), radius=16, outline="#cbd5e1", width=2)
        draw.text((left + 20, top + 18), sequence_name, font=body_font, fill="#1d2939")
        errors = np.asarray(
            [[row["baseline_error_m"], row["gate_error_m"]] for row in rows],
            dtype=np.float64,
        )
        display_max = max(float(np.percentile(errors, 98)), 0.4)
        display_max = min(display_max * 1.15, 2.0)
        chart_left, chart_top = left + 55, top + 75
        chart_width, chart_height = width - 85, height - 140
        draw.line((chart_left, chart_top, chart_left, chart_top + chart_height), fill="#98a2b3", width=2)
        draw.line((chart_left, chart_top + chart_height, chart_left + chart_width, chart_top + chart_height), fill="#98a2b3", width=2)
        for level in (0.0, 0.5, 1.0):
            if level > display_max:
                continue
            y = chart_top + chart_height - chart_height * level / display_max
            draw.line((chart_left, y, chart_left + chart_width, y), fill="#e2e8f0", width=1)
            draw.text((chart_left - 42, y - 9), f"{level:.1f}", font=small_font, fill="#667085")
        for metric, color in colors.items():
            values = errors[:, 0] if metric == "baseline" else errors[:, 1]
            points = []
            for index, value in enumerate(values):
                x = chart_left + chart_width * index / max(1, len(values) - 1)
                y = chart_top + chart_height - chart_height * min(value, display_max) / display_max
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
        draw.text((chart_left + 5, chart_top + 5), "灰：基线    蓝：学习门控", font=small_font, fill="#475467")
        draw.text((chart_left + chart_width - 92, chart_top + chart_height + 13), "采样帧", font=small_font, fill="#667085")
        baseline_mean = sequence["baseline"]["mean_position_error_m"]
        gate_mean = sequence["reliability_gate"]["mean_position_error_m"]
        draw.text(
            (left + 23, top + height - 47),
            f"平均误差：{baseline_mean:.4f} m → {gate_mean:.4f} m  （改善 {baseline_mean - gate_mean:+.4f} m）",
            font=small_font,
            fill="#344054",
        )
    combined = result["combined"]
    draw.rounded_rectangle((60, 650, 1480, 875), radius=16, outline="#cbd5e1", width=2)
    draw.text((85, 675), "合并结果", font=body_font, fill="#1d2939")
    labels = [
        ("基线平均误差", f"{combined['baseline']['mean_position_error_m']:.4f} m"),
        ("门控平均误差", f"{combined['reliability_gate']['mean_position_error_m']:.4f} m"),
        ("平均改善", f"{combined['mean_improvement_m']:+.4f} m"),
        ("<0.5m", f"{combined['baseline']['below_0p5m'] * 100:.2f}% → {combined['reliability_gate']['below_0p5m'] * 100:.2f}%"),
        ("门控实际接管", f"{combined['gate_apply_rate'] * 100:.2f}%"),
    ]
    for index, (label, value) in enumerate(labels):
        x = 88 + index * 274
        draw.text((x, 740), label, font=small_font, fill="#667085")
        draw.text((x, 780), value, font=body_font, fill="#1d2939")
    draw.text((85, 838), "注意：这是与同一采样间隔基线的盲测对照；门控的训练和阈值均未使用十月数据。", font=small_font, fill="#475467")
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired sampled Oct.12 baseline and reliability-gate runs.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    output_dir = Path(config["output_dir"])
    per_sequence = {
        name: _compare_sequence(Path(paths["baseline"]), Path(paths["reliability_gate"]))
        for name, paths in config["sequences"].items()
    }
    all_baseline = np.concatenate(
        [np.asarray([row["baseline_error_m"] for row in result["rows"]]) for result in per_sequence.values()]
    )
    all_gate = np.concatenate(
        [np.asarray([row["gate_error_m"] for row in result["rows"]]) for result in per_sequence.values()]
    )
    all_applied = np.concatenate(
        [np.asarray([row["gate_applied"] for row in result["rows"]]) for result in per_sequence.values()]
    )
    combined = {
        "baseline": _metrics(all_baseline),
        "reliability_gate": _metrics(all_gate),
        "mean_improvement_m": float(all_baseline.mean() - all_gate.mean()),
        "median_improvement_m": float(np.median(all_baseline - all_gate)),
        "improved_frame_ratio": float(np.mean(all_gate < all_baseline)),
        "gate_apply_rate": float(np.mean(all_applied)),
    }
    summary = {"per_sequence": per_sequence, "combined": combined}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reliability_gate_blind_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _draw_comparison(output_dir / "reliability_gate_blind_comparison.png", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
