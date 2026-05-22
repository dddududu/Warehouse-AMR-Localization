# Aisle 泛化版问题分析与通用机制优化（2026-05-21 / 2026-05-22）

## 目标

这一轮实验只关注 `Aisle_CCW` 和 `Aisle_CW`，目标是：

- 以当前 **可泛化主线** 为基线；
- 找出坏帧和连续坏片段；
- 判断问题到底是场景变化太大，还是在线决策机制出了问题；
- 在 **不破坏原有能力** 的前提下，只用 **通用机制** 提升坏片段能力；
- 所有新实验都放在本目录，和旧主线隔离。

---

## 一、当前泛化版 baseline

### `Aisle_CCW`
- 结果文件：`outputs/fine_localization_oct12_aisle_ccw_deep_v6_guidedbev_trackerinit_generic.json`
- `mean_position_error_m = 0.10729356583712467`
- `median_position_error_m = 0.08283090002360899`
- `mean_yaw_error_deg = 0.43135020801649326`
- `median_yaw_error_deg = 0.35686440796597796`
- `<1m = 1.0`
- `<0.5m = 0.9846994535519126`

### `Aisle_CW`
- 结果文件：`outputs/fine_localization_oct12_aisle_cw_deep_v6_guidedbev_trackerinit_generic_jun15.json`
- `mean_position_error_m = 0.2580936306428273`
- `median_position_error_m = 0.07126029617578356`
- `mean_yaw_error_deg = 0.39647016057414697`
- `median_yaw_error_deg = 0.3258551778456912`
- `<1m = 0.9423076923076923`
- `<0.5m = 0.9377289377289377`

### baseline 判断
- `Aisle_CCW` 已经非常稳；
- `Aisle_CW` 也不差，但还存在少量连续坏片段；
- 所以优化重点应放在 `Aisle_CW` 的连续坏段，同时不能把 `CCW` 带坏。

---

## 二、坏片段定位

### `Aisle_CCW`
- `>0.5m` 坏帧：`14`
- `>1m` 坏帧：`0`
- 主要片段：
  - `369–378`
  - `380–382`
  - `479`

结论：
- `CCW` 的问题不是大面积失效，而是少量亚米级偏移；
- 没有整段崩坏，不是主战场。

### `Aisle_CW`
- `>0.5m` 坏帧：`68`
- `>1m` 坏帧：`63`
- 主要片段：
  - `265–267`
  - `640–704`（最长、最重）

结论：
- `CW` 的核心问题集中在 `640–704`；
- 这是一个典型的“连续错误状态锁定”片段。

---

## 三、问题成因分析

### 不是“仓库变化太大导致看不懂”

如果真是场景变化太大，我们会看到：

- `CCW` 和 `CW` 都大面积退化；
- 错误分布会很散；
- 深度分数、几何分数、时序信息都会一起失效。

但真实情况不是这样：

- `CCW` 几乎全程很好；
- `CW` 也只有一小段长尾特别差；
- 说明系统整体识别能力还在，失败更像是 **状态机层面的错误延续**。

### 真正的问题：重复货架区里的“错误状态锁定”

对 `Aisle_CW 640–704` 的排查表明：

- challenger 候选其实已经多次出现；
- 深度支持有时已经更偏向 challenger；
- 但系统仍然持续留在旧 patch 状态里；
- `tracker_init` 和 `online_patch_hysteresis` 的组合把旧状态保持得过久。

这说明主矛盾不是“模型不会判别”，而是：

> **系统已经看见了更好的 challenger，但切换得不够果断。**

---

## 四、先排除一条错误方向：Aisle-only 长时间微调

### 做法

- 不改模型结构；
- 从当前 generic checkpoint 继续训练；
- 只使用 `Jun.15` 的 `Aisle_CCW + Aisle_CW` 数据；
- 训练超过 8 小时；
- 目标是看看能否只靠 `Aisle-only` 微调修复 `CW` 坏段。

相关文件：
- 训练配置：`fine_pose_matcher_train_jun15_aisleonly_long.yaml`
- 训练日志：`outputs/aisle_generic_20260521/long_run.log`
- checkpoint：`outputs/aisle_generic_20260521/fine_pose_matcher_aisleonly_long.pt`

### 结果

#### `Aisle_CCW`
- 结果文件：`outputs/aisle_generic_20260521/fine_localization_oct12_aisle_ccw_trackerinit_aisleonly.json`
- `mean_position_error_m = 2.671299301708454`
- `<1m = 0.6043715846994535`
- `<0.5m = 0.5846994535519126`

#### `Aisle_CW`
- 结果文件：`outputs/aisle_generic_20260521/fine_localization_oct12_aisle_cw_trackerinit_aisleonly.json`
- `mean_position_error_m = 1.9023474876144673`
- `<1m = 0.6492673992673993`
- `<0.5m = 0.6465201465201466`

### 结论

这条路失败，而且失败得很明确：

- `CCW` 和 `CW` 一起退化；
- 说明问题不是“把整个 fine matcher 再朝 Aisle 方向训一遍”就能解决；
- 这种做法会破坏 generic 主线原本已经很稳的判别边界。

因此：

> **不能靠重新训练整个 Aisle-only 模型来修坏段，同时保住原有能力。**

---

## 五、正确方向：只改通用在线决策机制

这一轮我们只做通用机制，不做定向 patch 规则，重点对应三件事：

1. challenger 连续多帧支持才触发切换  
2. tracker 主导状态的释放条件  
3. hysteresis 的动态 margin

### 这轮真正落地的关键改动

#### 1）在线 hysteresis 改成“运行中生效”

以前的 hysteresis 很多时候是后处理性质，容易和真实在线状态脱节。  
现在改成：

- 在帧序列运行过程中直接参与决策；
- 让 challenger 的连续支持真的能影响后续状态演化。

#### 2）tracker 释放逻辑缩到“同一已选 patch”内部

之前某些 tracker 释放逻辑过宽，会把 challenger 也卷进去，容易误伤。  
现在收紧成：

- 只针对 **当前已经连续保留的那个 patch**；
- 也就是只在“旧 patch 已经保持太久、深度支持又很低”的情况下，才放松 tracker 主导。

这一步非常关键，它把“释放旧状态”和“错误干扰 challenger”分开了。

#### 3）保留多帧 challenger 机制，不靠单帧强切换

这符合我们一开始的目标：

- 不因单帧抖动乱跳；
- 只有 challenger 连续支持，才更容易接管。

---

## 六、v14a：same-patch tracker release + online hysteresis

### 配置

#### `Aisle_CW`
- `fine_localization_oct12_aisle_cw_trackerinit_generic_v14a_samepatchrelease.yaml`

#### `Aisle_CCW`
- `fine_localization_oct12_aisle_ccw_trackerinit_generic_v14a_samepatchrelease.yaml`

共同特点：
- `apply_online_patch_hysteresis_during_tracking: true`
- `tracker_pose_init_sticky_patch_streak_min: 5`
- `tracker_pose_init_sticky_prev_deep_prob_max: 0.05`
- `tracker_pose_init_sticky_required_margin: 0.00`

这是一套 **通用机制**，不是 route-specific patch 规则。

---

## 七、v14a 结果

### 1）`Aisle_CW` 全序列

baseline：
- `mean_position_error_m = 0.2580936306428273`
- `median_position_error_m = 0.07126029617578356`
- `mean_yaw_error_deg = 0.39647016057414697`
- `median_yaw_error_deg = 0.3258551778456912`
- `<1m = 0.9423076923076923`
- `<0.5m = 0.9377289377289377`

v14a：
- 结果文件：`outputs/aisle_generic_20260521/fine_localization_oct12_aisle_cw_generic_v14a_samepatchrelease_full.json`
- `mean_position_error_m = 0.08798944482013578`
- `median_position_error_m = 0.06859578452361002`
- `mean_yaw_error_deg = 0.39587959540260104`
- `median_yaw_error_deg = 0.33101183709752224`
- `<1m = 0.9990842490842491`
- `<0.5m = 0.98992673992674`

#### `CW` 的核心提升
- `>0.5m` 坏帧：`68 -> 11`
- `>1m` 坏帧：`63 -> 1`
- 主坏段 `640–704`：
  - baseline 平均误差约 `3.0272m`
  - v14a 平均误差约 `0.1914m`
  - `<1m`：`4.62% -> 100%`
  - `<0.5m`：`0% -> 89.23%`

### 2）`Aisle_CCW` 全序列

baseline：
- `mean_position_error_m = 0.10729356583712467`
- `median_position_error_m = 0.08283090002360899`
- `mean_yaw_error_deg = 0.43135020801649326`
- `median_yaw_error_deg = 0.35686440796597796`
- `<1m = 1.0`
- `<0.5m = 0.9846994535519126`

v14a：
- 结果文件：`outputs/aisle_generic_20260521/fine_localization_oct12_aisle_ccw_generic_v14a_samepatchrelease_full.json`
- `mean_position_error_m = 0.10687801356908579`
- `median_position_error_m = 0.08205758340304625`
- `mean_yaw_error_deg = 0.4334947844763753`
- `median_yaw_error_deg = 0.3544091960955288`
- `<1m = 1.0`
- `<0.5m = 0.9836065573770492`

#### `CCW` 结论
- 基本保持不变；
- `mean/median` 位置误差略好；
- `<0.5m` 只有极小幅波动；
- 没有出现任何新的 `>1m` 坏段。

---

## 八、我们从这轮实验学到了什么

### 结论 1：Aisle 的问题不是“模型表示能力不够”

如果模型表示真的不够，我们会看到：
- 深度网络分不清 challenger；
- 几何和时序同时普遍崩；
- 必须重训模型才能解决。

但事实是：
- 只改在线决策机制，就能把 `CW` 的主坏段几乎完全修掉；
- 而且 `CCW` 基本不受损。

所以更准确的结论是：

> **Aisle 的主问题在在线状态切换，不在整体表征能力。**

### 结论 2：通用机制确实可以提升坏帧而不破坏原有能力

这一轮满足了我们的要求：

- 不加 route-specific patch 规则；
- 不重训整模型；
- 只改通用在线决策机制；
- `CW` 大幅提升；
- `CCW` 基本不退化。

这说明方向是对的。

### 结论 3：关键不在“更快切换”，而在“更正确地释放旧状态”

这轮最有效的不是简单调小 hysteresis margin，  
而是：

- 让在线 hysteresis 真正在运行时生效；
- 同时把 tracker 释放逻辑严格限制在 **同一已选 patch** 上。

这避免了：
- challenger 还没站稳就乱切；
- 旧 patch 又被 tracker 过度黏住。

---

## 九、当前推荐结论

如果只看 `Aisle` 泛化版：

- `Aisle_CCW` 基本已经在非常高的稳定水平；
- `Aisle_CW` 的主坏段已经通过 **通用机制** 明显修复；
- 当前最值得保留的方向是：
  - **在线 challenger 连续支持**
  - **same-patch tracker release**
  - **运行时 hysteresis**

而不是：
- 重新训练整个 `Aisle-only` 模型；
- 或者引入 route-specific patch 规则。

---

## 十、目录说明

- `analyze_aisle_failures.py`
  - 统计坏帧、坏段、主导 patch、初始化来源等
- `fine_pose_matcher_train_jun15_aisleonly_long.yaml`
  - 已证伪的 Aisle-only 长跑训练配置
- `fine_localization_oct12_aisle_ccw_trackerinit_generic_v14a_samepatchrelease.yaml`
  - `CCW` 的 v14a 通用机制配置
- `fine_localization_oct12_aisle_cw_trackerinit_generic_v14a_samepatchrelease.yaml`
  - `CW` 的 v14a 通用机制配置

---

## 十一、最终判断

这轮优化说明：

> **在 Aisle 场景里，最有效的提升来自在线状态机的通用修正，而不是重新训练整个深度模型，也不是 patch 定向规则。**

当前 v14a 是一条值得继续保留和推进的方向。
