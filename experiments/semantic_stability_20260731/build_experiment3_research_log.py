from __future__ import annotations

import argparse
import json
from pathlib import Path

from docx import Document
from PIL import Image, ImageDraw, ImageFont

from experiments.semantic_stability_20260731.build_experiment1_research_log import (
    _add_body,
    _add_figure,
    _add_heading,
    _add_table,
    _add_title,
    _clear_document_body,
)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _draw_comparison(output_path: Path, classifier: dict, geometry_risk: dict, semantic_risk: dict) -> None:
    baseline = float(classifier["validation"]["baseline_mean_position_error_m"])
    oracle = float(classifier["validation"]["oracle_mean_position_error_m"])
    entries = [
        ("原启发式", baseline, "#64748b"),
        ("分类门控", float(classifier["validation"]["mean_position_error_m"]), "#dc2626"),
        ("风险回归", float(geometry_risk["validation"]["mean_position_error_m"]), "#f59e0b"),
        ("语义风险回归", float(semantic_risk["validation"]["mean_position_error_m"]), "#2563eb"),
        ("候选内上限", oracle, "#16a34a"),
    ]
    canvas = Image.new("RGB", (1280, 780), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(31, True)
    body_font = _font(22)
    small_font = _font(18)
    draw.text((52, 34), "实验三：学习式可靠性门控的跨日期验证", font=title_font, fill="#1d2939")
    draw.text((52, 82), "训练：Jun.15 Aisle Run2；验证：Jun.23 Aisle Run2；每 8 帧取样", font=body_font, fill="#475467")
    left, top, width, height = 105, 165, 1030, 400
    draw.line((left, top, left, top + height), fill="#98a2b3", width=2)
    draw.line((left, top + height, left + width, top + height), fill="#98a2b3", width=2)
    y_max = 0.17
    for value in (0.00, 0.05, 0.10, 0.15):
        y = top + height - height * value / y_max
        draw.line((left, y, left + width, y), fill="#e2e8f0", width=1)
        draw.text((left - 67, y - 10), f"{value:.2f}", font=small_font, fill="#667085")
    bar_width = 112
    gap = 72
    for index, (label, value, color) in enumerate(entries):
        x = left + 70 + index * (bar_width + gap)
        bar_height = height * min(value, y_max) / y_max
        draw.rounded_rectangle(
            (x, top + height - bar_height, x + bar_width, top + height),
            radius=6,
            fill=color,
        )
        draw.text((x + 2, top + height - bar_height - 32), f"{value:.4f}", font=small_font, fill="#344054")
        label_x = x + bar_width / 2 - draw.textlength(label, font=small_font) / 2
        draw.text((label_x, top + height + 16), label, font=small_font, fill="#344054")
    draw.text((left - 12, top - 36), "平均位置误差（m）", font=small_font, fill="#667085")
    draw.rounded_rectangle((68, 640, 1212, 718), radius=13, fill="#f8fafc", outline="#cbd5e1")
    draw.text(
        (92, 661),
        "结论：两种学习门控在跨日期验证上均劣于原启发式，不能进入十月盲测；候选内仍有 0.0182 m 的可达余量。",
        font=body_font,
        fill="#344054",
    )
    canvas.save(output_path)


def build_log(template_path: Path, output_path: Path, result_dir: Path) -> None:
    classifier = json.loads((result_dir / "reliability_gate_summary.json").read_text(encoding="utf-8"))
    geometry_risk = json.loads((result_dir / "reliability_geometry_risk_gate_summary.json").read_text(encoding="utf-8"))
    risk = json.loads((result_dir / "reliability_risk_gate_summary.json").read_text(encoding="utf-8"))
    figure_path = result_dir / "reliability_gate_validation_comparison.png"
    _draw_comparison(figure_path, classifier, geometry_risk, risk)
    document = Document(template_path)
    _clear_document_body(document)
    _add_title(document, "研究日志（2026.07.31）")

    _add_heading(document, "一、实验目的")
    _add_body(
        document,
        "实验一和实验二说明，语义稳定性能够描述地图的长期可靠程度，但直接将语义分数用于候选 patch 重排的收益很小。进一步分析表明，精定位中真正需要回答的问题不是“某个候选是否静态”，而是“当前这一帧应相信 BEV 初始化、深度引导初始化，还是相信 tracker 初始化”。因此本实验把已有三条 ICP 初始化支路视为三个可行动作，利用训练日期的真值反事实结果监督一个学习门控，并严格在跨日期数据上决定其是否可部署。")

    _add_heading(document, "二、数据划分与门控构造")
    _add_body(
        document,
        "训练只使用 Jun.15 的 Aisle_CCW_Run_2 与 Aisle_CW_Run_2，共 296 个每 8 帧抽样的样本；验证固定为 Jun.23 对应的两条 Run_2，共 278 个样本。Oct.12 数据没有参与标签生成、特征归一化、早停、阈值选择或结构修改。每一帧首先按原 v14a 策略选定候选 patch，再保留同一 patch 下 BEV、深度引导和 tracker 三次 ICP 的最终位姿；以真值位置误差最小的支路生成反事实监督，同时规定当原支路与最优支路的差异不足 0.03 m 时保持原支路，以避免为微小噪声频繁切换。")
    _add_table(
        document,
        ["版本", "输入", "学习目标", "部署约束"],
        [
            ["V1 分类门控", "深度匹配概率、位姿置信度、三支路 ICP 残差和时序一致性，共 16 维", "预测应选择的初始化类别", "最大类别概率低于 0.55 时保持原策略"],
            ["V2 风险门控", "V1 特征", "回归三支路的 log(1+位置误差)", "仅在预测改善超过阈值时切换"],
            ["V3 语义风险门控", "V2 特征加人员、动态物体和半动态物体双目语义覆盖率，共 19 维", "与 V2 相同", "阈值只在 Jun.23 验证集扫描"],
        ],
        [1450, 3000, 2100, 1745],
    )

    _add_heading(document, "三、跨日期验证结果")
    classifier_validation = classifier["validation"]
    risk_validation = risk["validation"]
    _add_body(
        document,
        f"V1 的动作分类准确率达到 {classifier_validation['action_accuracy'] * 100:.2f}%，但平均位置误差从原策略的 {classifier_validation['baseline_mean_position_error_m']:.6f} m 恶化到 {classifier_validation['mean_position_error_m']:.6f} m，95 分位误差增至 {classifier_validation['p95_position_error_m']:.6f} m。原因是类别准确率把所有帧等权处理，而定位系统中少量错误切换会造成远大于普通帧收益的损失。将目标改为直接预测误差后，几何风险门控在验证集的平均误差为 0.132635 m，虽然比分类门控更保守，却仍高于原策略。")
    _add_body(
        document,
        f"V3 额外加入双目语义分割得到的人员、动态物体和半动态物体像素占比，平均误差进一步收敛到 {risk_validation['mean_position_error_m']:.6f} m，95 分位回到 {risk_validation['p95_position_error_m']:.6f} m，与基线相同，但平均误差仍比基线高 {risk_validation['baseline_mean_position_error_m'] - risk_validation['mean_position_error_m']:.6f} m。V3 在验证集实际触发切换的比例为 {risk_validation['gate_apply_rate'] * 100:.2f}%，表明语义动态覆盖能够减少极端误切换，却不足以可靠地区分同一候选 patch 内的错误 ICP 收敛盆地。")
    _add_figure(document, figure_path, "图 1  三个学习门控版本在 Jun.23 跨日期验证集上的平均位置误差对比")

    _add_heading(document, "四、问题分析")
    _add_body(
        document,
        f"候选内 oracle 的平均误差为 {classifier_validation['oracle_mean_position_error_m']:.6f} m，较原策略仍有 {classifier_validation['oracle_headroom_m']:.6f} m 的理论余量，说明不是三条初始化全部无效；问题在于当前 19 维门控特征无法预测“哪条支路将进入错误局部最优”。人员或推车的语义面积只能描述遮挡强弱，不能描述被遮挡的点云是否恰好破坏货架边缘、当前 patch 是否本身错误，以及 tracker 是否已经沿错误轨迹积累。因此，若直接把这个门控接入十月测试，会把验证集已观察到的退化风险带入盲测，结论不可信。")

    _add_heading(document, "五、结论与下一步")
    _add_body(
        document,
        "本实验的有效产出不是一个可部署的门控模型，而是明确排除了两条看似合理但泛化不足的路线：仅依赖几何/深度置信度的动作分类，以及加入语义面积后的单帧风险回归。后续应把学习对象从“初始化来源”前移到“当前 patch 是否正确”和“当前帧中哪些点可用于配准”：一方面用局部点云—双目图像联合特征预测候选 patch 的可辨识性，另一方面对连续多帧的静态点构建短时子图，再用鲁棒核 ICP 仅对稳定对应关系做细配准。只有将 patch 正确性、动态点剔除和多帧约束同时纳入，才可能避免少量错误切换主导均值误差。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the Experiment 3 research log from the July 18 template.")
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    build_log(Path(args.template), Path(args.output), Path(args.result_dir))


if __name__ == "__main__":
    main()
