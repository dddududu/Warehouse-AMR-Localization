# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize active manual-annotation queue coverage.")
    parser.add_argument("--summary", default="outputs/target_motion_20260812/active_person_annotation_queue/active_person_annotation_queue_summary.json")
    parser.add_argument("--output", default="outputs/target_motion_20260812/active_person_annotation_queue/active_annotation_queue_coverage.png")
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4), dpi=180)
    groups = [
        ("训练/验证覆盖", summary["selected_by_split"], "#4F81BD"),
        ("路线覆盖", summary["selected_by_sequence"], "#70AD47"),
        ("动静证据提示", summary["selected_by_motion_hint"], "#ED7D31"),
    ]
    for axis, (title, values, color) in zip(axes, groups, strict=True):
        labels = list(values)
        counts = [values[label] for label in labels]
        axis.bar(range(len(labels)), counts, color=color)
        axis.set_title(title, fontsize=11)
        axis.set_xticks(range(len(labels)), labels, rotation=26, ha="right", fontsize=8)
        axis.set_ylim(0, max(counts) + 2)
        axis.grid(axis="y", alpha=0.2)
        for index, count in enumerate(counts):
            axis.text(index, count + 0.15, str(count), ha="center", va="bottom", fontsize=9)
    figure.suptitle("首批 24 个主动人工核验片段的覆盖情况", fontsize=13)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")


if __name__ == "__main__":
    main()
