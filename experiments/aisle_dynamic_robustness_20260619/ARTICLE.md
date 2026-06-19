# 从测试标签门控到可部署门控：动态遮挡与无图像消融实验

## 1. 实验目标

这次实验接着 `experiments/aisle_dynamic_robustness_20260605/README.md` 第 8 节继续做两件事：

1. 把上一版依赖测试集语义标签的遮挡门控，改造成可部署的学习式遮挡预测器；
2. 在完成上述工作后，做一次去掉所有图像数据的消融实验，观察当前最优精定位到底有多依赖视觉输入。

这里的“去掉所有图像数据”采用最严格定义：不使用左右 RGB 图像，也不使用深度图/双目几何，只保留点云 BEV、地图候选 BEV、粗定位候选、tracker 和 ICP。

## 2. 实现内容

### 2.1 可部署遮挡预测器

新增轻量网络 `StereoOcclusionPredictor`：

- 输入：左右 RGB 图拼接成 `6 × H × W`；
- 输出：
  - `ratio`：预测人员遮挡占图比例；
  - `occlusion_logit`：是否超过遮挡阈值的分类分支；
- 训练标签：TorWIC 自带语义分割灰度图中 `13` 类人员像素比例；
- 训练序列：`Jun15_Aisle_CCW_Run_1`、`Jun15_Aisle_CW_Run_1`、`Jun15_Hallway_Full_CCW`、`Jun15_Hallway_Straight_CCW`；
- 验证序列：`Jun15_Aisle_CCW_Run_2`、`Jun15_Aisle_CW_Run_2`、`Jun15_Hallway_Full_CW`。

对应文件：

- `models/occlusion_predictor.py`
- `trainers/train_occlusion_predictor.py`
- `experiments/aisle_dynamic_robustness_20260619/occlusion_predictor_train.yaml`

### 2.2 定位管线集成

在 `DeepFineLocalizer` 中新增 `semantic_occlusion_predictor_checkpoint_path`：

1. 如果配置了遮挡预测器 checkpoint，则定位时只读左右 RGB 图，预测遮挡比例；
2. 如果没有配置 checkpoint，则回退到上一版的语义标签比例；
3. 如果预测比例超过阈值，并且当前候选原本会使用 `deep_init`，则保持深度网络对 patch 的评分，但将位姿初始化改为 tracker；
4. 输出 `semantic_occlusion_ratio` 和 `semantic_occlusion_ratio_source`，便于事后分析触发来源。

这实现了“深度网络负责选区域，tracker 负责遮挡期姿态”的结构，但遮挡信号从测试标签换成了可部署网络。

### 2.3 BEV-only 消融模型

为了真正去掉所有图像数据，不能直接复用原 checkpoint，因为原 `FinePoseMatcher` 的全连接维度会因图像/深度分支改变。

因此重新训练了一版 BEV-only 精定位模型：

- `use_query_image: false`
- `use_stereo_query_image: false`
- `use_query_depth: false`
- `use_stereo_geometry: false`

对应文件：

- `experiments/aisle_dynamic_robustness_20260619/fine_pose_matcher_train_jun15_fullroutes_bev_only.yaml`
- `outputs/aisle_dynamic_robustness_20260619/fine_pose_matcher_jun15_fullroutes_bev_only.pt`

训练日志显示最佳验证轮是第 1 轮：

| epoch | 验证匹配准确率 | pose L1 |
| ---: | ---: | ---: |
| 1 | 0.7556 | 2.3506 |
| 2 | 0.6593 | 2.4386 |
| 3 | 0.6639 | 2.6427 |

## 3. 遮挡预测器结果

### 3.1 六月验证集

遮挡预测器在六月验证集上的回归误差较小：

| 指标 | 数值 |
| --- | ---: |
| MAE | 0.0062 |
| RMSE | 0.0140 |

但以 `0.05` 作为高遮挡阈值时，召回率为 `0`。这说明网络学到了“大多数帧人员比例很低”的整体分布，但没有稳定识别少量高遮挡帧。

### 3.2 十月 Aisle 泛化检查

对十月 `Aisle_CCW 478–481` 的关键帧，标签人员比例与预测如下：

| 帧 | 标签人员比例 | 预测比例 | 定位效果 |
| ---: | ---: | ---: | --- |
| 478 | 0.0702 | 0.0007 | 未触发门控 |
| 479 | 0.0799 | 0.0005 | 未触发门控 |
| 480 | 0.1032 | 0.0003 | 未触发门控 |
| 481 | 0.0931 | 0.0002 | 未触发门控 |

完整学习门控定位结果：

| 路线 | 基线均值 | 标签门控均值 | 学习门控均值 | 结论 |
| --- | ---: | ---: | ---: | --- |
| Aisle_CCW | 0.106878m | **0.105106m** | 0.106878m | 学习门控未触发关键近距离行人 |
| Aisle_CW | 0.087989m | 0.087842m | 0.087842m | 只触发 1 帧，结果基本等同标签门控 |

学习门控没有替代标签门控。原因不是集成代码问题，而是训练数据分布与十月近距离模糊行人不一致：六月训练样本中高遮挡帧很少，且十月 `CCW 480` 的行人非常近、模糊、偏画面边缘，模型没有学到这种视觉形态。

## 4. 无图像数据消融结果

### 4.1 Aisle_CCW

| 方法 | 平均位置误差 | 中位位置误差 | 平均角度误差 | 小于 0.5m | 小于 1m |
| --- | ---: | ---: | ---: | ---: | ---: |
| v14a 基线 | 0.106878m | 0.082058m | 0.433495° | 98.3607% | 100.0000% |
| 标签遮挡门控 | **0.105106m** | 0.081484m | **0.419672°** | **98.5792%** | 100.0000% |
| BEV-only 消融 | 0.105584m | **0.080618m** | 0.422913° | 98.4699% | 99.8907% |

`Aisle_CCW` 上，BEV-only 没有明显崩溃，平均误差甚至略优于 v14a 基线，但仍弱于标签遮挡门控。最大问题仍在 `369–382`，其中 `371` 达到 `1.014m`，说明无图像模型对重复货架区的约束仍不足。

### 4.2 Aisle_CW

| 方法 | 平均位置误差 | 中位位置误差 | 平均角度误差 | 小于 0.5m | 小于 1m |
| --- | ---: | ---: | ---: | ---: | ---: |
| v14a 基线 | 0.087989m | 0.068596m | 0.395880° | 98.9927% | 99.9084% |
| 标签遮挡门控 | 0.087842m | **0.068172m** | 0.397619° | 98.9927% | 99.9084% |
| BEV-only 消融 | **0.086936m** | 0.068922m | **0.393479°** | **99.0842%** | **100.0000%** |

`Aisle_CW` 上，BEV-only 反而略好，尤其 `266` 从 `1.007m` 降到 `0.531m`。这说明当前视觉分支并不是所有尖峰的救星；某些场景下视觉/深度分支会给候选打分或初始化带来额外波动，而 BEV + tracker + ICP 更稳。

## 5. 关键结论

### 5.1 标签门控仍是当前行人遮挡上限

上一版基于语义标签的门控能把 `CCW 480` 从 `0.674m` 降到 `0.118m`。这证明“遮挡期让 tracker 接管位姿初始化”是正确机制。

但本次学习遮挡预测器未能复现这个收益。问题在于遮挡检测泛化，而不是定位门控逻辑。

### 5.2 视觉贡献没有想象中大

BEV-only 消融结果说明：

- 当前 Aisle 精定位的主体能力来自点云 BEV、粗定位候选、tracker 和 ICP；
- RGB/深度分支主要在部分 patch 选择和遮挡恢复上提供增益；
- 视觉分支如果遇到强遮挡或域偏移，也可能制造错误初始化。

### 5.3 后续重点应转向可靠性学习，而不是盲目加视觉

第 8 节中“学习式可靠性门控”的方向仍然正确，但输入不能只靠 RGB 遮挡比例。下一版门控应同时使用：

- RGB 遮挡预测；
- 激光投影到疑似人员区域的点比例；
- 深度初始化与 tracker 初始化的 ICP 残差差值；
- 深度匹配概率和 tracker 连续稳定性；
- 最近 `3–5` 帧的位姿创新量。

这样才能判断“这一帧应不应该相信深度初始化”，而不是只判断“画面里有没有人”。

## 6. 下一步建议

1. 不把当前 `stereo_occlusion_predictor_jun15.pt` 作为最终生产模型，只保留为失败消融；
2. 继续保留标签门控结果作为算法上限；
3. 训练第二版可靠性门控，目标不是预测人员比例，而是直接预测 `deep_init` 相对 `tracker_init` 是否更可靠；
4. 引入多帧静态子图，优先解决 `CCW 369–382` 和 `CW 672–679` 这类无明显行人但 tracker/ICP 退化的尖峰；
5. 做更强的数据增强：近距离人体剪贴、运动模糊、左右图不一致、边缘遮挡。

## 7. 复现命令

训练遮挡预测器：

```powershell
.\.venv\Scripts\python.exe -m trainers.train_occlusion_predictor `
  --config experiments\aisle_dynamic_robustness_20260619\occlusion_predictor_train.yaml
```

训练 BEV-only 精定位模型：

```powershell
.\.venv\Scripts\python.exe -m trainers.train_fine_pose_matcher `
  --config experiments\aisle_dynamic_robustness_20260619\fine_pose_matcher_train_jun15_fullroutes_bev_only.yaml `
  --output-checkpoint outputs\aisle_dynamic_robustness_20260619\fine_pose_matcher_jun15_fullroutes_bev_only.pt
```

跑学习遮挡门控：

```powershell
.\.venv\Scripts\python.exe -m localization.deep_fine_localizer `
  --config experiments\aisle_dynamic_robustness_20260619\fine_localization_oct12_aisle_ccw_v14a_learned_occlusion.yaml `
  --output-json outputs\aisle_dynamic_robustness_20260619\fine_localization_oct12_aisle_ccw_v14a_learned_occlusion_full.json `
  --no-resume
```

跑 BEV-only 消融：

```powershell
.\.venv\Scripts\python.exe -m localization.deep_fine_localizer `
  --config experiments\aisle_dynamic_robustness_20260619\fine_localization_oct12_aisle_ccw_v14a_bev_only_ablation.yaml `
  --output-json outputs\aisle_dynamic_robustness_20260619\fine_localization_oct12_aisle_ccw_v14a_bev_only_ablation_full.json `
  --no-resume
```
