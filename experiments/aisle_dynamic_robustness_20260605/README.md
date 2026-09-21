# Aisle 动态行人干扰与误差尖峰优化记录

## 1. 目标

本实验只关注 `Aisle_CCW` 和 `Aisle_CW`，以当前通用版 `v14a` 为基线，解决两个问题：

1. 找出测试集上剩余的大误差尖峰具体出现在哪些帧；
2. 重点降低行人从机器人旁边经过时对定位结果的影响，同时不使用指定路线、指定 patch 或指定帧号规则。

本实验目录与之前的 `experiments/aisle_generic_20260521` 分开，原始最优版本保持不变。

## 2. 基线

基线结果：

| 路线 | 平均位置误差 | 中位位置误差 | 平均角度误差 | 小于 0.5m |
| --- | ---: | ---: | ---: | ---: |
| Aisle_CCW | 0.106878m | 0.082058m | 0.433495° | 98.3607% |
| Aisle_CW | 0.087989m | 0.068596m | 0.395880° | 98.9927% |

对应结果文件：

- `outputs/aisle_generic_20260521/fine_localization_oct12_aisle_ccw_generic_v14a_samepatchrelease_full.json`
- `outputs/aisle_generic_20260521/fine_localization_oct12_aisle_cw_generic_v14a_samepatchrelease_full.json`

## 3. 全测试集尖峰诊断

使用 `analyze_dynamic_spikes.py` 对每一帧统计：

- 位置误差和角度误差；
- 左右图像中的人员像素比例；
- 动态类别像素比例；
- 图像拉普拉斯清晰度；
- 当前 patch、初始化来源、深度匹配概率；
- ICP 内点率和 ICP 均方根误差；
- 连续误差尖峰区间。

### 3.1 Aisle_CCW

误差大于等于 `0.5m` 的主要区间：

| 区间 | 峰值帧 | 峰值误差 | 行人情况 | 初步判断 |
| --- | ---: | ---: | --- | --- |
| 369–378 | 371 | 0.9773m | 几乎无人 | 重复结构和 tracker/ICP 累积偏差 |
| 380–382 | 381 | 0.6076m | 无人 | patch 交界处的历史偏差没有及时消除 |
| 479–480 | 480 | 0.6741m | 大面积近距离行人遮挡 | 深度初始化受图像遮挡影响，滞回保留了错误状态 |

`CCW 480` 中，人员最大占据约 `10.32%` 的单目图像。将激光投影到图像后，有约 `1205` 个投影点落在人员掩码内，人员点距离机器人中位数约 `1.04m`。

### 3.2 Aisle_CW

主要区间：

| 区间 | 峰值帧 | 峰值误差 | 行人情况 | 初步判断 |
| --- | ---: | ---: | --- | --- |
| 265–267 | 266 | 1.0073m | 无明显行人 | patch 切换和 tracker 突然漂移 |
| 672–679 | 672 | 0.8116m | 有远处行人，但画面占比很小 | 所有候选均带有相近历史偏差，并非单纯动态遮挡 |

因此，行人是重要风险源，但不是所有尖峰的统一解释。

## 4. 论文与公开研究借鉴

### 4.1 TorWIC 自带语义信息

TorWIC 官方仓库给出了语义类别表，其中 `13` 是人员，`12/14/15` 分别是其他非静态物体、叉车/卡车和其他动态物体。官方也说明，数据集中的语义掩码由语义分割模型推理产生，并非完全无误。

- [TorWICDataset 官方仓库](https://github.com/Viky397/TorWICDataset)

### 4.2 DynaSLAM

DynaSLAM 使用深度学习语义检测与多视图几何共同识别动态区域，并在跟踪和建图前排除动态物体。它还讨论了被动态物体遮挡后的背景修复。

对本项目的启发：不能只删除动态激光点，还要降低动态遮挡对图像特征和位姿初始化的可信度。

- [DynaSLAM](https://arxiv.org/abs/1806.05620)

### 4.3 SuMa++

SuMa++ 使用神经网络给激光点预测语义标签，在扫描匹配时过滤移动物体，并使用语义一致性改善匹配。

对本项目的启发：语义不应只作为预处理删除器，也可以作为匹配权重和置信度来源。

- [SuMa++](https://arxiv.org/abs/1908.06214)

### 4.4 Dynamic Object Aware LiDAR SLAM

该方法训练实时网络检测任意动态物体，并通过自动生成训练标签降低人工标注成本。论文报告显式处理动态物体后，激光里程计性能得到明显改善。

对本项目的启发：后续应把当前数据集语义掩码作为教师信号，训练可在线运行的动态可靠性网络，而不是在部署时依赖测试集标签文件。

- [Dynamic Object Aware LiDAR SLAM](https://arxiv.org/abs/2104.03657)

### 4.5 ERASOR

ERASOR 使用跨帧占据变化检测动态区域，不依赖固定的已知物体类别。

对本项目的启发：除人员语义外，还需要加入地图占据不一致和多帧持续性，识别叉车、货物移动等未知动态变化。

- [ERASOR](https://arxiv.org/abs/2103.04316)

### 4.6 鲁棒 ICP

Babin 等人系统比较了多种 ICP 离群点处理方式，指出可变裁剪、Cauchy 和 Cauchy-MAD 在不同环境中更稳定。

对本项目的启发：后续 ICP 应增加鲁棒核和残差裁剪，但本次实验表明，`CCW 480` 的主要问题在深度初始化和状态切换，单纯删除人员激光点并不能解决尖峰。

- [Analysis of Robust Functions for Registration Algorithms](https://arxiv.org/abs/1810.01474)

## 5. 迭代实验

### 5.1 只删除人员激光点

实现：

1. 读取左右灰度语义图；
2. 使用相机内参与激光到相机外参投影激光点；
3. 删除落入人员掩码的点；
4. 使用过滤后的点继续生成 BEV、深度匹配和 ICP。

结果：

- `CCW` 平均位置误差从 `0.106878m` 变为 `0.106849m`；
- `479–480` 的误差几乎没有变化。

结论：

人员点虽然数量多且距离近，但经过 `0.3m` 体素化后，对最终 ICP 的直接影响有限。真正的问题是图像被大面积遮挡后，深度网络仍给出了错误的深度位姿初始化，后续状态机又接受并保留了该结果。

### 5.2 全局要求 challenger 连续两帧

修复了一个逻辑缺口：旧代码虽然统计 challenger 连续帧数，但普通候选只要单帧分数超过 margin，仍然可以立即切换，连续支持要求会被绕过。

完整重跑后：

- `479–480` 从 `0.5462/0.6741m` 降到 `0.2004/0.2047m`；
- 小于 `0.5m` 的比例从 `98.3607%` 提升到 `98.5792%`；
- 但平均误差上升到 `0.110327m`。

原因：

全局连续帧要求会延迟正常 patch 切换。它能抑制瞬时错误，也会让真正的 patch 转换变慢。

### 5.3 语义遮挡时直接禁用深度初始化

当人员占图比例大于等于 `5%` 时，直接从候选位姿假设中删除深度初始化。

该版本降低了行人尖峰，但也降低了目标 patch 的候选总分，导致 patch 选择和位姿来源被错误地绑定在一起，机器人长时间停留在旧 patch。

结论：

“哪个 patch 正确”和“当前 patch 应使用哪个位姿初值”必须拆开处理。

### 5.4 最终版本：深度选 patch，tracker 接管遮挡期位姿

最终机制：

1. 深度网络仍正常计算每个候选 patch 的匹配概率、深度引导 BEV 分数和候选分数；
2. patch 排名不因遮挡门控而被削弱；
3. 当人员占图比例大于等于 `5%`，且当前候选原本准备使用深度位姿初始化，同时 tracker 初始化有效时：
   - 保留深度网络提供的 patch 评分；
   - 将最终位姿初值替换为 tracker 初始化；
   - 再执行 ICP；
4. 输出中使用 `deep_score_tracker_pose` 标记这种“深度负责选区域、tracker负责姿态”的情况。

这个机制没有使用帧号、路线方向或 patch 编号，是数据驱动的通用遮挡可靠性门控。

## 6. 最终全量结果

### 6.1 Aisle_CCW

| 指标 | v14a 基线 | 遮挡解耦门控 | 变化 |
| --- | ---: | ---: | ---: |
| 平均位置误差 | 0.106878m | **0.105106m** | -0.001772m |
| 中位位置误差 | 0.082058m | **0.081484m** | -0.000573m |
| 平均角度误差 | 0.433495° | **0.419672°** | -0.013823° |
| 中位角度误差 | 0.354409° | **0.348554°** | -0.005855° |
| 小于 0.5m | 98.3607% | **98.5792%** | +0.2186 个百分点 |
| 小于 1m | 100% | 100% | 不变 |

关键人员帧：

| 帧 | 基线误差 | 新误差 | 改善 |
| ---: | ---: | ---: | ---: |
| 478 | 0.3397m | 0.0856m | -0.2542m |
| 479 | 0.5462m | 0.0530m | -0.4932m |
| 480 | 0.6741m | 0.1181m | -0.5560m |
| 481 | 0.2619m | 0.1112m | -0.1507m |

结果文件：

- `outputs/aisle_dynamic_robustness_20260605/fine_localization_oct12_aisle_ccw_v14a_occlusion_gate_decoupled_full.json`

### 6.2 Aisle_CW

`Aisle_CW` 没有人员占图比例达到 `5%` 的帧，因此遮挡门控没有触发。完整重跑结果没有位置精度回退：

| 指标 | v14a 基线 | 遮挡解耦门控 |
| --- | ---: | ---: |
| 平均位置误差 | 0.087989m | **0.087842m** |
| 中位位置误差 | 0.068596m | **0.068172m** |
| 小于 0.5m | 98.9927% | 98.9927% |
| 小于 1m | 99.9084% | 99.9084% |

平均角度误差由 `0.395880°` 变为 `0.397619°`，变化约 `0.00174°`，属于非常小的运行波动。

结果文件：

- `outputs/aisle_dynamic_robustness_20260605/fine_localization_oct12_aisle_cw_v14a_occlusion_gate_decoupled_full.json`

## 7. 当前剩余瓶颈

最大的剩余问题不再是行人，而是以下无明显动态遮挡的区间：

- `Aisle_CCW 369–382`
- `Aisle_CW 265–267`
- `Aisle_CW 672–679`

反事实候选检查表明：

1. 这些帧大多不是简单选错 patch；
2. 多个候选都使用 tracker 初始化，并收敛到相近的错误位置；
3. ICP 内点率仍可达到 `0.97` 左右，说明高内点率在重复货架和长直通道中不等于位姿正确；
4. 误差来自此前几帧积累的 tracker 偏差、单帧几何退化和 ICP 局部极值。

下一阶段应从“单帧候选切换”转向“多帧静态证据和不确定性建模”。

## 8. 下一步实施路线

### 第一优先级：把测试标签门控替换为可部署网络

1. 使用 TorWIC 提供的语义掩码作为教师标签；
2. 训练轻量人员/动态区域分割头；
3. 输入左右图像，输出人员掩码、人员占图比例和遮挡置信度；
4. 将遮挡置信度用于位姿初始化门控，而不是写死测试标签路径；
5. 训练时加入人员剪贴、近距离遮挡、运动模糊和左右图不对称增强。

### 第二优先级：学习式可靠性门控

训练一个小型门控网络，输入：

- 人员占图比例；
- 人员掩码与激光投影的重叠比例；
- 深度匹配概率；
- 深度初始化和 tracker 初始化的 ICP 内点率、均方根误差差值；
- 当前速度、角速度和历史位姿创新量。

输出：

- 深度初始化权重；
- tracker 初始化权重；
- 是否需要保持上一帧位姿状态；
- 当前定位置信度。

这样可以把固定 `5%` 阈值改成学习得到的连续权重。

### 第三优先级：多帧静态子图

针对剩余无行人尖峰：

1. 累积最近 `3–5` 帧经过动态过滤的点云；
2. 使用 tracker 先验将多帧点云变换到当前帧；
3. 对跨帧持续出现的点提高权重，对只出现一帧的点降低权重；
4. 使用多帧子图执行 scan-to-map 配准；
5. 当单帧几何退化时，用多帧观测扩大纵向约束范围。

### 第四优先级：鲁棒 ICP

在现有 ICP 中加入：

- Cauchy 或 Cauchy-MAD 权重；
- 可变比例残差裁剪；
- 空间覆盖率，而不仅是总内点率；
- 近距离动态簇降权；
- 双向一致性检查。

该部分主要解决未知动态物体和局部异常点，但不会替代多帧约束。

## 9. 复现命令

尖峰分析：

```powershell
.\.venv\Scripts\python.exe experiments\aisle_dynamic_robustness_20260605\analyze_dynamic_spikes.py `
  --sequence-name Aisle_CCW `
  --sequence-root "<TORWIC_ROOT>\Oct. 12, 2022\Aisle_CCW\Aisle_CCW" `
  --result-json outputs\aisle_generic_20260521\fine_localization_oct12_aisle_ccw_generic_v14a_samepatchrelease_full.json `
  --output-json outputs\aisle_dynamic_robustness_20260605\aisle_ccw_dynamic_spikes.json `
  --output-csv outputs\aisle_dynamic_robustness_20260605\aisle_ccw_dynamic_spikes.csv
```

最终 `CCW`：

```powershell
.\.venv\Scripts\python.exe -m localization.deep_fine_localizer `
  --config experiments\aisle_dynamic_robustness_20260605\fine_localization_oct12_aisle_ccw_v14a_occlusion_gate.yaml `
  --output-json outputs\aisle_dynamic_robustness_20260605\fine_localization_oct12_aisle_ccw_v14a_occlusion_gate_decoupled_full.json `
  --no-resume
```

最终 `CW`：

```powershell
.\.venv\Scripts\python.exe -m localization.deep_fine_localizer `
  --config experiments\aisle_dynamic_robustness_20260605\fine_localization_oct12_aisle_cw_v14a_occlusion_gate.yaml `
  --output-json outputs\aisle_dynamic_robustness_20260605\fine_localization_oct12_aisle_cw_v14a_occlusion_gate_decoupled_full.json `
  --no-resume
```
