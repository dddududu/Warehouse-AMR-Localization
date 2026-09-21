# 研究日志：面向目标识别与动静分类的主动人工核验队列

**日期：2026.08.12**
**研究主题：在不伪造动静真值的前提下，优先构建高价值人员实例标注样本。**

## 一、研究问题

前期实验已经明确：Aisle 场景中人员出现与定位误差尖峰具有统计关联，但原始语义标签 13 只能提供类别像素，COCO 检测框和点云/深度反投影也只能提供候选和运动提示，均不能直接充当人员实例框或动静训练标签。此前 48 个五帧候选全部处于 `pending_manual_verification` 状态，训练数据构建器正确地将它们全部拦截。因此，当前瓶颈不是继续训练检测器，而是如何在有限人工标注预算下优先获得同时服务“人员识别”和“动静分类”的高价值真值。

## 二、主动核验方法

对每个候选片段只使用六月训练/验证数据中已经存在的、部署时可获得的信息进行排序：第一，连续三维观测数量；第二，COCO 检测器在五帧内的平均置信度和相邻框重叠度；第三，自动运动提示的类型。自动运动提示仍只是排序依据：具有至少三帧三维中心且提示为 `potentially_moving` 或 `ambiguous_motion` 的片段进入 A 类，优先确认真实人员及其运动状态；具有连续三维证据但相对静止或不确定的片段进入 B 类，用作动静对照；三维证据不足的片段进入 C 类，主要用于排除检测误检并补充识别样本。

排序过程不使用十月盲测数据、十月定位误差、未来帧，且不把 motion hint、检测框或三维轨迹写成标签。为了避免队列只集中在单一轨迹，还加入了覆盖约束：首批样本同时覆盖训练/验证划分、Aisle_CW/Aisle_CCW 两个方向以及 Jun.15 的四条路线。

## 三、首批人工核验队列

从 48 个待核验候选中选出 24 个首批片段。队列包含 14 个训练片段、10 个验证片段；按路线分别为 Aisle_CCW_Run2 10 个、Aisle_CW_Run2 8 个、Aisle_CCW_Run1 3 个、Aisle_CW_Run1 3 个。按证据层级，A 类 23 个、B 类 1 个；按自动提示，16 个为 `potentially_moving`、7 个为 `ambiguous_motion`、1 个为 `likely_static_relative_to_map`。这意味着首批人工工作优先覆盖近距离、连续三维证据充分、最可能受动态人员影响的片段，同时仍保留跨日期验证样本和少量静态对照。

![首批主动核验队列覆盖](../../outputs/target_motion_20260812/active_person_annotation_queue/active_annotation_queue_coverage.png)

## 四、人工核验与训练准入

标注应严格按 `review_order` 执行。每条片段先填写 `manual_person_label`；只有确认是真实人员时，才确认或修订 `manual_box_json`，并填写五帧一致的 `manual_instance_id`。随后根据五帧可见轨迹、遮挡情况和三维证据填写 `manual_motion_label`：证据不足或遮挡严重时必须填写 `uncertain`，不得强制赋予 `moving` 或 `static`。完成后使用既有训练数据构建器检查：训练集与验证集必须都有真实人员框、至少存在可用运动标签，并且不完整记录仍会被阻断。只有通过这一准入检查，才能启动检测器微调和动静分类训练。

## 五、结论与下一步

本阶段没有宣称得到新的人员检测或动静分类精度，因为人工真值尚未产生；它解决的是前一阶段“自动候选无法安全进入训练”的工程与实验设计问题。首批 24 个片段将把标注工作集中到最可能解释定位尖峰的连续受扰样本，并保留跨路线验证。人工核验完成后，下一步依次是：构建仅含人工确认框的训练/验证集；先评估人员检测器的跨日期召回与 F1；再仅在检测质量达标时训练 `moving/static/uncertain` 分类头；最后在六月验证集测试保守软门控，十月数据仍只保留为最终盲测。

## 六、可复现产物

- 队列生成脚本：`build_active_person_annotation_queue.py`
- 首批队列：`outputs/target_motion_20260812/active_person_annotation_queue/active_person_annotation_queue.csv`
- 队列说明：`outputs/target_motion_20260812/active_person_annotation_queue/annotation_plan.md`
- 覆盖统计：`outputs/target_motion_20260812/active_person_annotation_queue/active_person_annotation_queue_summary.json`
