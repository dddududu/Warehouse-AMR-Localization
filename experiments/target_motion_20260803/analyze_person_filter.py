from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = ["C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _metrics(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "frames": int(array.size),
        "mean_position_error_m": float(array.mean()),
        "median_position_error_m": float(np.median(array)),
        "p95_position_error_m": float(np.quantile(array, 0.95)),
        "below_0p5m_ratio": float(np.mean(array < 0.5)),
    }


def _draw(path: Path, metrics: dict[str, dict[str, dict[str, float | int]]]) -> None:
    canvas = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((65, 45), "人员语义硬过滤：十月 CCW 完整盲测对比", fill="#172B4D", font=_font(38, True))
    draw.text((65, 108), "人员出现由独立语义审计确定；两种方法使用同一粗定位、深度网络、ICP 与在线跟踪配置。", fill="#52606D", font=_font(21))
    groups = [("全序列", "all"), ("人员出现帧", "person"), ("无人员帧", "no_person")]
    x_values = [330, 770, 1210]
    maximum = max(float(metrics[key][method]["mean_position_error_m"]) for _, key in groups for method in ("baseline", "person_only"))
    for (label, key), center in zip(groups, x_values, strict=True):
        draw.text((center - 85, 215), label, fill="#243B53", font=_font(25, True))
        baseline = float(metrics[key]["baseline"]["mean_position_error_m"])
        person_only = float(metrics[key]["person_only"]["mean_position_error_m"])
        for x, value, color, method in ((center - 90, baseline, "#5B8FF9", "原模型"), (center + 15, person_only, "#D64545", "仅移除人员")):
            height = int(410 * value / maximum)
            draw.rectangle((x, 650 - height, x + 75, 650), fill=color)
            draw.text((x - 8, 663), method, fill="#52606D", font=_font(17))
            draw.text((x + 5, 650 - height + 12), f"{value:.4f}", fill="white", font=_font(18, True))
        delta = person_only - baseline
        draw.text((center - 120, 730), f"变化：{delta:+.4f} m", fill="#B42318" if delta > 0 else "#27864B", font=_font(22, True))
    draw.rectangle((65, 800, 92, 827), fill="#5B8FF9")
    draw.text((105, 800), "原模型", fill="#52606D", font=_font(21))
    draw.rectangle((250, 800, 277, 827), fill="#D64545")
    draw.text((290, 800), "仅移除人员点云", fill="#52606D", font=_font(21))
    draw.text((590, 800), "结论：不应将“人员被识别到”等同于“整帧几何证据应被删除”。", fill="#334E68", font=_font(21, True))
    canvas.save(path)


def run(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    baseline_path = Path("outputs/aisle_generic_20260521/fine_localization_oct12_aisle_ccw_generic_v14a_samepatchrelease_full.json")
    person_path = root / "person_only_oct12_aisle_ccw.json"
    observations_path = root / "audit" / "frame_target_observations.csv"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["frame_results"]
    person_only = json.loads(person_path.read_text(encoding="utf-8"))["frame_results"]
    person_present = {
        int(row["frame_idx"]): row["present"].lower() == "true"
        for row in csv.DictReader(observations_path.open(encoding="utf-8-sig"))
        if row["sequence"] == "oct12_aisle_ccw" and row["target_name"] == "person"
    }
    baseline_errors = {int(row["frame_idx"]): float(row["position_error_m"]) for row in baseline}
    person_errors = {int(row["frame_idx"]): float(row["position_error_m"]) for row in person_only}
    shared = sorted(set(baseline_errors).intersection(person_errors))
    groups = {
        "all": shared,
        "person": [frame_idx for frame_idx in shared if person_present.get(frame_idx, False)],
        "no_person": [frame_idx for frame_idx in shared if not person_present.get(frame_idx, False)],
    }
    metrics = {
        group: {
            "baseline": _metrics([baseline_errors[frame_idx] for frame_idx in frame_indices]),
            "person_only": _metrics([person_errors[frame_idx] for frame_idx in frame_indices]),
        }
        for group, frame_indices in groups.items()
    }
    summary = {
        "sequence": "Oct.12 Aisle_CCW",
        "method": "Only label-13 points are removed with five-pixel mask dilation; all other localization parameters remain frozen.",
        "metrics": metrics,
        "decision": "rejected",
        "reason": "The all-frame, person-frame, and no-person-frame mean errors all worsen; do not run the symmetric full CW replication.",
    }
    (root / "person_filter_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _draw(root / "person_filter_comparison.png", metrics)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the CCW person-only semantic filter experiment.")
    parser.add_argument("--output-dir", default="outputs/target_motion_20260803")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
