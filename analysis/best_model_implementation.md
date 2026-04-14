# TorWIC 项目当前最优模型实现说明

本文档总结截至目前仓库里**效果最好**的一组实现，重点回答四个问题：

1. 当前最优方案到底是什么。
2. 模型网络结构具体长什么样。
3. 这个结果是如何一步一步迭代出来的。
4. `ICP` 和其他方法分别解决什么问题，各自优势是什么。

---

## 1. 先说结论

当前项目已经形成了两条不同定位目标下的最优主线：

- **A. 可泛化、可全测试集运行的主线**
  - 目标：不依赖路线方向性的 patch 专项规则，在 `Oct. 12` 全测试集上稳定工作。
  - 形式：**粗定位检索 + 深度候选打分/初值 + BEV 粗配准 + ICP 精配准 + 在线时序稳定**
  - 当前全测试集最佳结果：
    - 总帧数：`9630`
    - 加权平均位置误差：`4.6724 m`
    - 加权平均航向误差：`22.6473 deg`
    - `<1m` 比例：`78.75%`
    - `<0.5m` 比例：`78.34%`
  - 汇总文件：`outputs/oct12_full_generic_jun15_v3_summary.json`
  - 说明：该主线在**粗定位层**按场景使用了更合适的 coarse checkpoint：
    - `Aisle` 使用 `classifier-head stride10` 最优 coarse
    - `Hallway_Full` 与 `Hallway_Straight_CW` 使用 `Jun.15 full-routes coarse`

- **B. 单路线封顶的专项最优**
  - 目标：在某些固定路线把结果压到极致。
  - 形式：在通用主线之上叠加少量路线相关的 patch 级修正。
  - 代表结果：
    - `Aisle_CW`：
      - 位置误差均值：`0.3879 m`
      - 位置误差中位数：`0.0752 m`
      - 航向误差均值：`0.9824 deg`
      - `<1m`：`98.17%`
      - 文件：`outputs/fine_localization_oct12_aisle_cw_deep_v6_guidedbev_pairresolver_corepatch_v2_trackerinit_full.json`

如果目标是**工程可部署、可泛化**，应该看 A。  
如果目标是**单条路线冲极限指标**，可以参考 B。

---

## 2. 当前最优系统由哪几部分组成

当前最好用的整套系统，不是单一网络，也不是单一几何模块，而是一个分层系统：

1. **粗定位（Coarse Retrieval）**
2. **深度精定位候选打分与位姿先验（Deep Fine Matcher）**
3. **局部几何初始化与精配准（BEV + ICP）**
4. **在线时序稳定（online stabilizer / hysteresis / tracker_init gating）**

核心思想是：

- 粗定位先把**全局搜索空间缩小**到 `topK patch`
- 深度模型负责：
  - 判断哪个 patch 更像真值
  - 给出相对位姿初值
  - 提供置信度
- 几何模块负责：
  - 在局部子图里做真正的精配准
  - 输出米级以下甚至分米级的几何解
- 时序模块负责：
  - 防止长序列里跳到错误 patch
  - 防止 tracker 把错误初值持续锁死

---

## 3. 粗定位：当前最优实现

### 3.1 当前最优粗定位模型

在 `Aisle_CCW` 上，当前单模型最优粗定位仍然是：

- 训练配置：`configs/coarse_retrieval_classifier_head_e2_stride10.yaml`
- checkpoint：`checkpoints/coarse_retrieval_classifier_head_e2_stride10_top1_0p4426_r3_0p7672.pt`
- 指标：
  - `Top1 = 0.4426`
  - `Recall@3 = 0.7672`
  - `MRR = 0.6463`

在 `Hallway` 上，更有效的是后续补做的 `Jun.15 full-routes coarse`：

- 训练配置：`configs/coarse_retrieval_jun15_fullroutes_stride10.yaml`
- checkpoint：`outputs/coarse_retrieval_jun15_fullroutes_stride10.pt`

它不是在 `Aisle` 上绝对最强，但对 `Hallway` 的 coarse 召回提升更明显，因此最终全测试集主线里对 `Hallway` 使用它更合理。

### 3.2 粗定位输入表征

粗定位不是直接吃原始点云，而是先把 query 点云裁成局部区域，再转成 `4` 通道 BEV：

- `count`
- `max_height`
- `mean_height`
- `occupancy`

之后做固定归一化：

- 点数用 `log1p` + clip
- 高度用固定 `z_min/z_max` 归一化
- 占据直接裁剪到 `[0, 1]`

### 3.3 粗定位主干网络结构

粗定位 backbone 在 `models/coarse_encoder_backbone.py`，结构如下：

```text
Input BEV [4, H, W]
│
├─ FixedBEVNormalizer
│
├─ Stem
│  └─ Conv(4→32, 3x3) + BN + ReLU
│
├─ Stage1
│  ├─ ResidualBlock(32→32, stride=1)
│  └─ ResidualBlock(32→32, stride=1)
│
├─ Stage2
│  ├─ ResidualBlock(32→64, stride=2)
│  └─ ResidualBlock(64→64, stride=1)
│
├─ Stage3
│  ├─ ResidualBlock(64→128, stride=2)
│  └─ ResidualBlock(128→128, stride=1)
│
├─ Stage4
│  ├─ ResidualBlock(128→192, stride=2)
│  └─ ResidualBlock(192→192, stride=1)
│
├─ GeM Pooling
│
└─ Projection MLP
   ├─ Linear(192→256) + ReLU
   └─ Linear(256→256)
      └─ L2 normalize
```

输出：

- `stage2 / stage3 / stage4` 多尺度特征图
- `256` 维归一化 descriptor

### 3.4 粗定位整体结构

粗定位整体定义在 `models/coarse_retrieval_model.py`：

```text
Query BEV ──> Query Encoder ──> query descriptor ──┐
                                                  ├─ descriptor similarity
Patch BEV ──> Patch Encoder ──> patch descriptor ─┘

query descriptor ──> query classifier ──> patch logits

final coarse score = descriptor score + classifier score fusion
```

可选还支持：

- `local matcher` 二阶段重排
- query / patch encoder 共享
- patch 分类辅助监督

但最终留下来的单模型最优，是：

- `20m x 20m` patch
- `stride = 10m`
- `legacy CNN descriptor`
- `classifier head` 融合

### 3.5 粗定位为什么最后选这套

因为实验反复证明：

- 单纯 descriptor：不够稳
- 多日期训练：对跨日期泛化极其重要
- 分类头融合：明显提升 `Top1`
- patch 改成 `20m / stride10` 后，必须重训；重训后的 `classifier-head` 反而成为最强单模型

---

## 4. 精定位：当前最优实现

### 4.1 当前保留下来的精定位主线

现在真正有效的精定位主线不是“纯深度回归位姿”，而是：

```text
topK coarse patches
    ↓
build local submap for each candidate
    ↓
Deep Fine Matcher:
    - match probability
    - relative pose prior
    - pose confidence
    ↓
for each candidate:
    BEV coarse alignment
    deep pose init
    tracker pose init
    ↓
    ICP refine
    ↓
candidate score fusion
    ↓
online stabilizer / patch hysteresis
    ↓
final pose
```

也就是说：

- **深度学习主导候选判别和初值生成**
- **ICP 负责最后的几何精修**

### 4.2 精定位深度网络结构

深度匹配网络定义在 `models/fine_pose_matcher.py`，名字是 `FinePoseMatcher`。

它不是单纯的一个 BEV 网络，而是一个多输入、多头输出的网络。

#### 4.2.1 输入

对于每个 query 和每个候选 local submap，输入包括：

- `query BEV`
- `candidate BEV`
- `query left image`
- `query right image`
- `query stereo geometry / depth features`

注意：

- 地图侧仍然是点云子图转 BEV
- 图像只在 query 侧进入，用于帮助识别重复结构

#### 4.2.2 BEV 编码器

`FineBEVEncoder` 结构如下：

```text
Input BEV [4, H, W]
│
├─ Conv(4→32, stride=2) + BN + ReLU
├─ Conv(32→64, stride=2) + BN + ReLU
├─ Conv(64→96, stride=2) + BN + ReLU
├─ Conv(96→128, stride=2) + BN + ReLU
│
├─ Global Average Pool
└─ MLP Projection
   ├─ Linear(128→descriptor_dim) + ReLU
   └─ Linear(descriptor_dim→descriptor_dim)
      └─ L2 normalize
```

输出：

- 局部特征图 `feature_map`
- 全局描述子 `descriptor`

#### 4.2.3 图像编码器

`QueryImageEncoder`：

```text
RGB image
│
├─ Conv(3→32, 5x5, stride=2) + BN + ReLU
├─ Conv(32→64, 3x3, stride=2) + BN + ReLU
├─ Conv(64→96, 3x3, stride=2) + BN + ReLU
├─ Conv(96→128, 3x3, stride=2) + BN + ReLU
│
└─ GAP + MLP projection + L2 normalize
```

左右图共享同一个编码器。

#### 4.2.4 几何编码器

`QueryGeometryEncoder`：

```text
Stereo depth / geometry features
│
├─ Conv(3→24, 5x5, stride=2) + BN + ReLU
├─ Conv(24→48, 3x3, stride=2) + BN + ReLU
├─ Conv(48→96, 3x3, stride=2) + BN + ReLU
├─ Conv(96→128, 3x3, stride=2) + BN + ReLU
│
└─ GAP + MLP projection + L2 normalize
```

#### 4.2.5 Query-Candidate 融合结构

对于每个 query-candidate 对，网络做两类融合：

##### A. 局部特征图融合

把 query 与 candidate 的特征图按以下方式拼接：

- `Q`
- `C`
- `|Q - C|`
- `Q * C`

得到 `4 x 128 = 512` 通道，然后过 `spatial_fusion`：

```text
Conv(512→hidden_dim, 3x3) + BN + ReLU
Conv(hidden_dim→hidden_dim, 3x3) + BN + ReLU
GAP
```

##### B. 全局描述子融合

拼接：

- query descriptor
- candidate descriptor
- `|q - c|`
- `q * c`
- left image descriptor
- right image descriptor
- geometry descriptor

然后与局部融合特征拼到一起，进入 `pair_head`：

```text
Linear(pair_dim→hidden_dim) + ReLU
Linear(hidden_dim→hidden_dim) + ReLU
```

#### 4.2.6 输出头

网络不是只输出一个分数，而是同时输出：

```text
pair embedding
match_logit
x_bin_logits
y_bin_logits
yaw_bin_logits
pose_residual
pose_confidence
```

位姿不是纯回归，而是：

- `x` 分箱分类 + 残差
- `y` 分箱分类 + 残差
- `yaw` 分箱分类 + 残差

这是后来验证下来比“直接回归 pose”更稳的一种表示。

### 4.3 精定位几何层

#### 4.3.1 BEV 粗配准

`localization/bev_matcher.py` 做的是局部子图上的模板匹配：

- 对 query 点云做旋转采样
- 转成 BEV
- 在 candidate submap 的 BEV 上做 `cv2.matchTemplate`
- 搜索：
  - 平移
  - yaw

优点：

- 能在**候选 patch 内做局部全局搜索**
- 比直接上 ICP 更不容易陷入局部最优

#### 4.3.2 ICP 精配准

`localization/icp_refiner.py` 做的是平面约束下的 ICP：

- 先 voxel downsample
- 用 FLANN 建立 map 点云近邻索引
- 每轮：
  - 用当前 pose 变换 query
  - 找最近邻
  - 估计平面 `SE(2)+z` 增量
  - 迭代更新

输出：

- `pose_4x4`
- `num_inliers`
- `inlier_ratio`
- `rmse`
- `converged`

### 4.4 当前精定位真正有效的关键：三初值竞争

当前主线里，每个 candidate 不再只有一种初始化，而是三种初始化竞争：

1. `bev_init`
2. `deep_init`
3. `tracker_init`

其中：

- `bev_init` 来自局部 BEV 相关性匹配
- `deep_init` 来自深度网络预测的 `dx, dy, yaw`
- `tracker_init` 来自上一帧位姿的常速度预测

然后三者都进 ICP，比较：

- `bev_score`
- `icp_inlier_ratio`
- `icp_rmse`
- `temporal_score`
- 与预测轨迹的一致性

最后选分数最高的那一个。

这个改动是后期精度显著提升的决定性因素之一。

### 4.5 在线稳定与错误抑制

长序列真正难的不是前 50 帧，而是：

- 错误 patch 会被连续维持
- tracker 错了以后会持续把系统往错误局部极值里带

所以当前主线里又加了两层在线约束：

- `online pose stabilizer`
- `online patch hysteresis`

它们的作用是：

- 优先保留高分且运动连续的解
- 只有当 challenger 明显更好时才切 patch
- 减少长序列抖动和错误 patch 翻来覆去切换

---

## 5. 发现问题、解决问题的迭代过程

这是当前结果真正重要的部分。项目不是一次调参出来的，而是靠逐层定位瓶颈得到的。

### 阶段 1：最早 baseline 很弱

最早粗定位 baseline：

- `Top1 = 0.1738`
- `Recall@3 = 0.4033`

问题：

- 训练日期太单一
- 跨日期泛化差
- 纯 descriptor 的 patch 判别能力不足

### 阶段 2：先把 coarse 做对

做了三件关键事：

1. 多日期训练
2. query 侧增强
3. patch 分类头融合

随后粗定位大幅提升：

- `Top1 = 0.3388`
- `Recall@3 = 0.7038`

结论：

- 问题首先不是 ICP，而是 coarse 候选本身不够好

### 阶段 3：patch 划分重构

后来确认 patch 划分不合理，改成：

- `20m x 20m`
- `stride = 10m`

改完以后发现：

- 旧模型直接复用会退化
- 必须重训

重训后最佳单模型变成：

- `classifier-head stride10`
- `Top1 = 0.4426`
- `Recall@3 = 0.7672`

这一步把粗定位主线定下来了。

### 阶段 4：纯几何精定位先跑通

先实现了最传统的链路：

```text
topK coarse -> local submap -> BEV matching -> ICP -> smoothing
```

结果：

- 在 `Aisle` 的短窗口上很好
- 但长序列容易跳到错误货架/错误通道

结论：

- 单靠几何，重复结构下的 patch 判别和长序列稳定性不够

### 阶段 5：直接让深度网络回归 pose，不成功

先后试过：

- `V1`: pairwise 深度 pose 回归
- `V2`: topK candidate-set 分类 + pose
- `V3`: pose bin classification + residual
- `V4`: 加单目 / 双目图像
- `V5/V6`: 真正 coarse topK 候选流

这些实验说明了一个事实：

- **深度模型直接端到端回归最终位姿，不稳定**
- 真正适合深度模型做的是：
  - 候选判别
  - 候选排序
  - 位姿初值

而不是完全替代几何优化。

### 阶段 6：转成“深度主导 + 几何收尾”

后续把策略改成：

- 深度模型负责候选与初值
- BEV 和 ICP 负责局部几何求精

这条路开始稳定奏效。

### 阶段 7：定位长序列大误差的真正根因

之后发现大误差不止一种：

1. **patch 选错**
2. **patch 选对了，但 tracker 把初值带进错误局部极值**

这一步非常关键，因为之前很多优化都把这两类问题混在一起处理，效果不稳定。

### 阶段 8：tracker_init 变成关键增益点

在 `CW` 上先验证到：

- 加 `tracker_init` 作为第三初始化源后
- 很多“patch 选对但 pose 错”的窗口被修掉了

随后再迁移到 `CCW`，结果继续显著提升。

### 阶段 9：Hallway 的通用改法不是关 tracker，而是门控 tracker

在 `Hallway_Full` 与 `Hallway_Straight_CW` 上继续定位发现：

- tracker 并不是“全错”或“全对”
- 真问题是：**系统没有判别什么时候该信 tracker**

于是改成：

- 只有 `tracker_init` 自身 `ICP inlier / rmse` 足够好时
- 才允许它参与竞争或主导

这个改动直接把全测试集 generic 指标进一步推高。

---

## 6. 当前最优结果一览

## 6.1 全测试集可泛化主线

汇总：`outputs/oct12_full_generic_jun15_v3_summary.json`

| 路线 | 帧数 | 均值位置误差 | 中位位置误差 | 均值航向误差 | <1m |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Aisle_CCW` | 915 | `0.1073 m` | `0.0828 m` | `0.4314 deg` | `100.00%` |
| `Aisle_CW` | 1092 | `0.2581 m` | `0.0713 m` | `0.3965 deg` | `94.23%` |
| `Hallway_Full_CW_Run_1` | 2318 | `8.2763 m` | `0.1487 m` | `45.9240 deg` | `60.83%` |
| `Hallway_Full_CW_Run_2` | 2320 | `4.0132 m` | `0.0777 m` | `13.2431 deg` | `85.78%` |
| `Hallway_Straight_CCW` | 1175 | `8.0271 m` | `0.2656 m` | `55.0601 deg` | `58.81%` |
| `Hallway_Straight_CW` | 1810 | `3.6948 m` | `0.0889 m` | `8.5055 deg` | `85.58%` |

全测试集加权结果：

- `mean_position_error = 4.6724 m`
- `mean_yaw_error = 22.6473 deg`
- `<1m = 78.75%`
- `<0.5m = 78.34%`

### 6.2 单路线专项上限

`Aisle_CW` 路线专项版本：

- 文件：`outputs/fine_localization_oct12_aisle_cw_deep_v6_guidedbev_pairresolver_corepatch_v2_trackerinit_full.json`
- 指标：
  - `mean_position_error = 0.3879 m`
  - `median_position_error = 0.0752 m`
  - `mean_yaw_error = 0.9824 deg`
  - `<1m = 98.17%`
  - `<0.5m = 94.23%`

这个版本更强，但包含路线相关修正，不作为全测试集 generic 主线。

---

## 7. ICP 与其他方法的区别

这是本项目最容易混淆的地方。

### 7.1 ICP 是什么

`ICP` 是一个**局部几何优化器**，它做的事情是：

- 给定一个初始位姿
- 在这个初始位姿附近
- 通过最近邻对应关系不断迭代
- 把 query 点云和地图子图对齐

它的特点：

- **优点**
  - 几何意义直接
  - 不依赖训练
  - 最终米级/分米级精修能力强
- **缺点**
  - 非常依赖初值
  - 容易陷入局部最优
  - 重复结构场景下容易配到错误位置

### 7.2 BEV 模板匹配是什么

BEV 匹配不是最终精修器，而是局部全局搜索器。

它做的事是：

- 先把点云投影成 BEV
- 在 candidate submap 里搜索平移和旋转

特点：

- **优点**
  - 比 ICP 更能做局部范围内的“全局搜索”
  - 对初值要求没那么高
- **缺点**
  - 分辨率有限
  - 最终精度不如 ICP

### 7.3 深度匹配器是什么

深度匹配器的作用不是直接替代所有几何，而是学习：

- 哪个候选更像真值
- 候选和 query 的相对位姿大概是多少
- 这个候选值不值得信

特点：

- **优点**
  - 能学习跨日期、跨模态差异
  - 能识别几何上很像、但统计上不一样的区域
  - 能提供几何方法没有的置信度
- **缺点**
  - 训练数据与分布敏感
  - 单独用来回归最终位姿时不稳定

### 7.4 为什么最终不是“纯 ICP”，也不是“纯深度”

因为两者各自有明显短板：

- **纯 ICP**
  - 精修强
  - 但不会全局选 patch，也不会解决重复结构的语义歧义

- **纯深度回归**
  - 有学习能力
  - 但在这个仓库场景里，单独直接回归最终 pose 不够稳定

所以最终保留下来的最优路线是：

```text
深度学习负责“选谁、先大致放哪”
几何方法负责“最后精确对齐”
```

这也是当前工程上最稳、效果最好的组合。

---

## 8. 这套方案各模块的优势

### 8.1 粗定位检索

- 优势：快速把搜索空间从全图压到少数 patch
- 适合解决：全局定位 / patch 召回

### 8.2 深度候选匹配器

- 优势：能利用 LiDAR + 图像 + 几何特征进行跨日期判别
- 适合解决：重复结构下的候选歧义

### 8.3 BEV 匹配

- 优势：局部范围内做更稳的旋转和平移搜索
- 适合解决：给 ICP 提供靠谱初值

### 8.4 ICP

- 优势：最后一步精修精度高
- 适合解决：局部子图内的精确配准

### 8.5 tracker_init + 在线稳定器

- 优势：长序列连续性强，能显著减少跳变
- 适合解决：时序漂移和错误 patch 持续锁定

---

## 9. 当前最重要的经验结论

1. **先把 coarse 做强，再谈 fine。**
   - 如果 `topK` 里真值都不在，后面什么都救不了。

2. **不要让深度网络直接完全替代几何。**
   - 在这个项目里，深度直接回归最终 pose 的稳定性不够。

3. **真正有效的是“深度主导 + 几何收尾”。**
   - 深度负责候选与初值，几何负责最终落位。

4. **长序列大误差通常不是单一原因。**
   - 必须区分：
     - patch 选错
     - patch 选对但 pose 错

5. **tracker 不该全开，也不该全关。**
   - 最有效的通用策略是：
     - 只在 tracker 自己的 ICP 质量足够高时才信它

---

## 10. 复现入口

### 10.1 粗定位训练

`Aisle` 最优 coarse：

```bash
python -m trainers.train_coarse_retrieval --config configs/coarse_retrieval_classifier_head_e2_stride10.yaml --output-checkpoint outputs/coarse_retrieval_classifier_head_e2_stride10.pt
```

`Hallway` 通用 coarse：

```bash
python -m trainers.train_coarse_retrieval --config configs/coarse_retrieval_jun15_fullroutes_stride10.yaml --output-checkpoint outputs/coarse_retrieval_jun15_fullroutes_stride10.pt
```

### 10.2 深度精定位训练

```bash
python -m trainers.train_fine_pose_matcher --config configs/fine_pose_matcher_train_jun15_fullroutes.yaml --output-checkpoint outputs/fine_pose_matcher_jun15_fullroutes_generic.pt
```

### 10.3 深度精定位评测

例如 `Hallway_Straight_CW` 当前 generic 最优：

```bash
python -m localization.deep_fine_localizer --config configs/fine_localization_oct12_hallway_straight_cw_deep_v6_guidedbev_generic_jun15coarse_trackerquality.yaml --output-json outputs/fine_localization_oct12_hallway_straight_cw_deep_v6_guidedbev_generic_jun15coarse_trackerquality_full.json
```

---

## 11. 总结

截至目前，这个项目最重要的不是“找到了一个神奇大模型”，而是把整个系统整理成了一条有效的工程链路：

- 粗定位：`20m / stride10` 的 BEV 检索 + classifier fusion
- 精定位：多模态 `FinePoseMatcher`
- 初始化：`bev_init + deep_init + tracker_init`
- 精修：`ICP`
- 长序列稳定：`online stabilizer + hysteresis + tracker quality gate`

这个结果不是靠一次性大改得到的，而是通过反复定位瓶颈、区分错误类型、逐个替换无效设计得到的。

如果后续继续优化，最值得继续投入的方向有两个：

1. `Hallway_Straight_CCW` 的通用化提升
2. 将当前 `Hallway` 上有效的 tracker quality gating 再进一步做成更强的学习式时序决策
