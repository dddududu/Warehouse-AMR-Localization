# 研究日志：定位失败病例—正常对照的目标与动静核验设计

**日期：2026.08.13**
**研究主题：让人员识别与动静分类研究直接回答“动态目标是否造成定位失败”。**

## 一、为什么要改变标注队列

先前 48 个五帧人员候选是按检测时序、三维证据和自动运动提示排序的，适合建立通用人员样本库，但几乎没有覆盖真正的定位失败：Jun.15 的 Aisle_CCW_Run2 有 5 个误差不小于 0.20 m 的帧、Aisle_CW_Run2 有 22 个，而旧队列中只有 1 条片段位于这些高误差帧附近。因此，即使完成旧队列，也很难回答“人从机器人旁边经过是否导致误差尖峰”。

## 二、病例—对照设计

本队列从 Jun.15 训练路线与 Jun.23 验证路线的完整定位报告中提取连续高误差段；每个段选择误差峰值帧作为 `high_error` 病例，再从同一路线中寻找距离该段至少 30 帧、误差不超过 0.10 m 的 `normal_control`。优先选择相同 patch 的对照，使两类片段尽量处于可比的地图局部区域，而非简单比较不同仓库区域。每条记录提供中心帧前后各两帧、路线和 patch 信息，但不提供自动人员框作为真值。

离线位置误差只用于抽样与最终统计：例如比较病例组和对照组中人工确认人员、推车、叉车以及主要货架几何遮挡的比例。它绝不能进入人员检测器、动静分类器或定位门控器的训练数据；训练时只使用人工确认的框、实例编号和 `moving/static/uncertain` 标签。

## 三、预期可回答的问题

完成这批核验后，可以用同一标注协议检验三个具体假设：第一，高误差病例中的真实人员/动态物体比例是否高于正常对照；第二，若存在人员，它是否遮挡了机器人前方货架、柱子等高信息量结构，而不是只出现在远处背景；第三，Top-5 选择失败病例是否比候选缺失病例更容易伴随动态遮挡。只有前两项成立，才说明动态目标的识别与动静分类有机会真正改善定位；若病例与对照差异不明显，应优先研究货架重复结构、地图陈旧和粗定位召回，而不是继续扩大目标识别训练。

## 四、可复现产物

- 队列生成：`experiments/target_motion_20260812/build_error_conditioned_annotation_queue.py`
- 人工核验队列：`outputs/target_motion_20260812/error_conditioned_annotation_queue/error_conditioned_person_motion_annotation_queue.csv`
- 核验说明：`outputs/target_motion_20260812/error_conditioned_annotation_queue/annotation_instructions.md`
- 统计摘要：`outputs/target_motion_20260812/error_conditioned_annotation_queue/error_conditioned_queue_summary.json`
