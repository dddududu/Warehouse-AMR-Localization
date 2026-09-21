# -*- coding: utf-8 -*-
"""Compare automatic semi-dynamic exposure across frozen localization outcomes.

The semantic ratio is an automatically produced scene statistic, not a manual
person or motion label. This analysis therefore measures association only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


NORMAL_ERROR_M = 0.10
HIGH_ERROR_M = 0.20
TRIGGER_RATIO = 0.10
GROUPS = (
    ("normal", "正常帧（误差≤0.10m）"),
    ("topk_selection_failure", "Top-5 选错"),
    ("topk_partial_recovery_only", "Top-5 只能部分恢复"),
    ("topk_candidate_absence", "Top-5 没有可用候选"),
)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def _ratio(row: dict[str, str]) -> float | None:
    value = row.get("semi_dynamic_point_ratio")
    if value in {None, ""}:
        return None
    return float(value)


def _group_rows(rows: list[dict[str, str]], group: str) -> list[dict[str, str]]:
    if group == "normal":
        return [row for row in rows if float(row["selected_position_error_m"]) <= NORMAL_ERROR_M]
    return [row for row in rows if row["failure_category"] == group]


def _statistics(rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    ratios = np.asarray([value for row in rows if (value := _ratio(row)) is not None], dtype=float)
    if not ratios.size:
        return {"frames": len(rows), "mean_ratio": None, "median_ratio": None, "trigger_rate": None}
    return {
        "frames": len(rows),
        "mean_ratio": float(ratios.mean()),
        "median_ratio": float(np.median(ratios)),
        "trigger_rate": float(np.mean(ratios > TRIGGER_RATIO)),
    }


def _permutation_test(high: np.ndarray, normal: np.ndarray, trials: int = 20_000) -> dict[str, float]:
    observed = float(high.mean() - normal.mean())
    values = np.concatenate((high, normal)).copy()
    generator = np.random.default_rng(0)
    simulated = np.empty(trials, dtype=float)
    for index in range(trials):
        generator.shuffle(values)
        simulated[index] = values[: high.size].mean() - values[high.size :].mean()
    p_value = float((np.count_nonzero(np.abs(simulated) >= abs(observed)) + 1) / (trials + 1))
    return {"mean_difference_high_minus_normal": observed, "two_sided_permutation_p_value": p_value}


def _draw_chart(summary: dict[str, Any], output_path: Path) -> None:
    width, height = 1420, 820
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, subtitle_font = _font(38, True), _font(20)
    axis_font, value_font = _font(20), _font(22, True)
    draw.text((70, 42), "Oct.12 自动语义统计与定位失败类型", font=title_font, fill="#172B4D")
    draw.text(
        (70, 96),
        "这里的比例来自自动语义分割，只能说明相关性，不能代替人工确认的人员/动静标签。",
        font=subtitle_font,
        fill="#52606D",
    )
    left, top, right, bottom = 120, 210, 1340, 625
    draw.line((left, bottom, right, bottom), fill="#334E68", width=3)
    draw.line((left, top, left, bottom), fill="#334E68", width=3)
    for value in np.linspace(0.0, 0.12, 5):
        y = bottom - int((bottom - top) * value / 0.12)
        draw.line((left, y, right, y), fill="#D9E2EC", width=1)
        draw.text((30, y - 12), f"{value * 100:.0f}%", font=axis_font, fill="#52606D")
    colors = ["#4C78A8", "#F58518", "#E45756", "#54A24B"]
    spacing = (right - left) // len(GROUPS)
    for index, ((key, label), color) in enumerate(zip(GROUPS, colors)):
        metric = summary["groups"][key]
        ratio = float(metric["mean_ratio"] or 0.0)
        center = left + spacing * index + spacing // 2
        bar_width = 150
        bar_top = bottom - int((bottom - top) * ratio / 0.12)
        draw.rounded_rectangle((center - bar_width // 2, bar_top, center + bar_width // 2, bottom), radius=8, fill=color)
        draw.text((center - 45, bar_top - 38), f"{ratio * 100:.2f}%", font=value_font, fill="#172B4D")
        lines = label.split("（")
        draw.text((center - 100, bottom + 24), lines[0], font=axis_font, fill="#334E68")
        if len(lines) > 1:
            draw.text((center - 116, bottom + 55), "（" + lines[1], font=_font(16), fill="#52606D")
        draw.text((center - 42, bottom + 90), f"n={metric['frames']}", font=_font(17), fill="#52606D")
    test = summary["high_vs_normal_permutation_test"]
    note = (
        f"高误差帧与正常帧的平均比例差为 {test['mean_difference_high_minus_normal'] * 100:+.2f}%；"
        f"随机置换检验 p={test['two_sided_permutation_p_value']:.4f}。"
    )
    draw.rounded_rectangle((70, 700, 1350, 775), radius=12, fill="#F0F7FF", outline="#B8D6F2", width=2)
    draw.text((95, 723), note, font=subtitle_font, fill="#172B4D")
    image.save(output_path)


def run(input_csv: str | Path, output_dir: str | Path) -> dict[str, Any]:
    rows = list(csv.DictReader(Path(input_csv).open(encoding="utf-8-sig")))
    summary: dict[str, Any] = {
        "input": str(input_csv),
        "normal_error_threshold_m": NORMAL_ERROR_M,
        "high_error_threshold_m": HIGH_ERROR_M,
        "semantic_trigger_ratio": TRIGGER_RATIO,
        "groups": {},
        "interpretation": (
            "Automatic semi-dynamic exposure is a scene-level proxy only. It must not be used as a person or motion label."
        ),
    }
    for key, _ in GROUPS:
        summary["groups"][key] = _statistics(_group_rows(rows, key))
    normal = np.asarray([_ratio(row) for row in _group_rows(rows, "normal") if _ratio(row) is not None], dtype=float)
    high = np.asarray(
        [_ratio(row) for row in rows if float(row["selected_position_error_m"]) >= HIGH_ERROR_M and _ratio(row) is not None],
        dtype=float,
    )
    summary["high_vs_normal_permutation_test"] = _permutation_test(high, normal)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "semantic_exposure_failure_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _draw_chart(summary, output_dir / "semantic_exposure_by_failure.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze automatic semantic exposure by frozen failure mechanism.")
    parser.add_argument(
        "--input-csv",
        default="outputs/target_motion_20260812/candidate_error_mechanisms/oct12_candidate_error_rows.csv",
    )
    parser.add_argument("--output-dir", default="outputs/target_motion_20260829/semantic_exposure_failure")
    args = parser.parse_args()
    print(json.dumps(run(args.input_csv, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
