# Channel x Trial Beta 模式稳定性分析

本子项目用于回答一个更具体的问题：对同一个通道而言，在一个 block 的有效 trial 中，其单 trial beta 动态模式是否会在两类响应之间变化。

## 目标模式

当前只考虑 beta 频带。基于主分类结果和曲线观察，重点区分两种 trial-level 模式：

```text
Type 1:
speech onset 时 beta 功率下降；
言语持续过程中 beta 功率回升；
speech offset 附近再次下降；
言语间隙逐渐回到高功率。

Type 2:
speech onset 时 beta 功率下降；
言语持续过程中维持低功率状态；
speech offset 后在言语间隙回到高功率。
```

真正要验证的是：

```text
对于某一个通道，它在 block 内的每个有效 trial 是 Type 1 还是 Type 2？
这个 trial-wise type 是否稳定，还是会随 trial 顺序发生改变？
```

## 只使用有效 trial

所有分析只使用主流程 `beta_epochs.events` 中保留下来的有效 trial：

- onset/offset 合法；
- duration outlier 已剔除；
- onset 前窗口和 offset 后窗口都在 recording 范围内；
- 坏道不参与通道稳定性判定。

原始 trial 如果已经在主流程中被剔除，不再进入 `channel x trial` 分析。

## 推荐实现方式

不建议每个 trial 独立重新 K-means。这里的目标不是重新发现任意簇，而是判断每个单 trial 更像 Type 1 还是 Type 2。

推荐使用“固定模板或固定原型”的 trial-wise 分类：

1. 从有效 trial 的 beta 曲线中构建 Type 1 和 Type 2 的参考模板。
2. 对每个 `channel x trial` 的 beta 曲线提取形态特征。
3. 将该 trial 与 Type 1/Type 2 模板比较，得到：
   - `trial_type`
   - `type1_score`
   - `type2_score`
   - `confidence`
   - `uncertain` 标记。
4. 对每个通道汇总其 trial-wise type 序列，判断是否稳定或改变。

这样每个 trial 的标签都基于同一套 Type 1/Type 2 定义，避免 trial 间标签漂移。

## Type 1 / Type 2 的专业判定特征

由于两类模式的主要差异在“言语持续过程中是否回升”，建议特征集中关注形态，而不是绝对能量高低。

### 核心形态特征

- `onset_drop`
  - onset 后早期 beta 相对 pre-onset baseline 的下降幅度。

- `mid_speech_rebound`
  - speech 中后段 beta 是否从早期低谷回升。

- `speech_low_maintenance`
  - speech 期间维持低 beta 的程度。

- `offset_drop`
  - offset 附近 beta 是否再次下降。

- `post_offset_recovery`
  - offset 后言语间隙 beta 是否恢复到高功率。

- `rebound_peak_percent`
  - speech 期间回升峰值出现在 trial 进程的百分比位置。

- `beta_slope_audio_slope_corr`
  - beta 斜率与音频包络斜率之间的相关或互相关。

### Type 1 倾向

```text
onset_drop 明显；
mid_speech_rebound 明显；
offset_drop 可见；
post_offset_recovery 明显；
speech_low_maintenance 较弱。
```

### Type 2 倾向

```text
onset_drop 明显；
mid_speech_rebound 弱或不存在；
speech_low_maintenance 明显；
post_offset_recovery 明显；
offset 后才恢复到高功率。
```

## 推荐分类策略

### 首选：形态打分 + 模板匹配

这是当前最可解释的方案。

对每个 `channel x trial` 计算两个分数：

```text
type1_score = rebound_score + post_recovery_score + onset_drop_score + offset_drop_score
type2_score = low_maintenance_score + post_recovery_score + onset_drop_score - rebound_score
```

然后：

```text
trial_type = argmax(type1_score, type2_score)
confidence = abs(type1_score - type2_score)
```

如果两个分数太接近，则标记为：

```text
uncertain
```

优点：

- 直接对应神经生理假设；
- 可解释；
- 不依赖每个 trial 内重新聚类；
- 适合 trial 数不是特别大的情况。

### 备选：全体 channel x trial 上做二类聚类

可以把所有好通道的所有有效 trial 作为样本，使用 beta 曲线形态特征做 `K=2` 聚类，然后人工根据平均曲线把两个 cluster 命名为 Type 1 和 Type 2。

注意：这个 K-means 只能在全体 `channel x trial` 上统一拟合一次，不能每个 trial 单独拟合。

优点：

- 更数据驱动；
- 可发现 Type 1/Type 2 以外的中间模式。

风险：

- 如果 Type 1/Type 2 不平衡，K-means 可能按噪声或幅度分簇；
- 需要再次强调弱化绝对能量高低，强化形态特征。

## 通道稳定性指标

对每个通道，基于其有效 trial 的 type 序列计算：

- `dominant_type`
  - 出现最多的 Type。

- `stability_rate`
  - dominant type 占有效 trial 的比例。

- `switch_count`
  - 相邻有效 trial 中 Type 改变的次数。

- `switch_rate`
  - `switch_count / (n_valid_trials - 1)`。

- `mean_confidence`
  - trial-wise 分类置信度均值。

- `uncertain_rate`
  - 被标记为 uncertain 的 trial 比例。

- `early_late_shift`
  - block 前半段和后半段 Type 1 比例的差异。

建议解释规则：

```text
stability_rate >= 0.75 且 mean_confidence 高:
    该通道模式稳定。

连续 >= 3 个有效 trial 从 Type 1 转到 Type 2，或从 Type 2 转到 Type 1:
    可能存在 block 内模式转换。

switch_rate 高但 mean_confidence 低:
    更可能是边界通道或低信噪比，不应直接解释为真实模式改变。

uncertain_rate 高:
    该通道不适合强行归类，应进入人工复核。
```

## 可视化设计

### 1. Channel x Trial Type 热图

行是通道，列是有效 trial。

颜色：

```text
Type 1: 一种颜色
Type 2: 另一种颜色
Uncertain: 灰色
Bad channel: 白色或不显示
```

通道排序建议：

- 先按 dominant type；
- 再按 stability_rate；
- 最后按 switch_count。

### 2. 单通道 trial 序列图

对某个通道显示：

- 每个有效 trial 的 Type；
- type1_score 和 type2_score；
- confidence；
- speech duration；
- 可选叠加 beta 曲线小 multiples。

### 3. 通道级稳定性散点图

x 轴：

```text
stability_rate
```

y 轴：

```text
mean_confidence
```

点颜色：

```text
dominant_type
```

点大小：

```text
switch_count
```

### 4. Type 比例随 block 变化

按 trial 顺序计算：

```text
Type 1 通道比例
Type 2 通道比例
Uncertain 比例
```

这张图回答整个 block 是否发生群体级模式漂移。

## 计划模块

- `trial_type_features.py`
  - 对每个有效 `channel x trial` 提取 Type 1/Type 2 形态特征。

- `trial_type_classifier.py`
  - 基于形态打分或固定模板进行 trial-wise Type 分类。

- `stability_metrics.py`
  - 计算每个通道的稳定性、切换率、前后半段变化和 uncertain 比例。

- `visualization.py`
  - 绘制 channel x trial heatmap、单通道序列图、稳定性散点图和 block-level type proportion。

- `channelXtrial_pipeline.ipynb`
  - 从主流程复用 `beta_epochs` 和有效 trial，执行完整分析。

## 当前已实现

当前版本已经实现：

- `trial_type_features.py`
- `trial_type_classifier.py`
- `stability_metrics.py`
- `visualization.py`
- `pipeline.py`
- `channelXtrial_pipeline.ipynb`

## 运行方式

在项目根目录启动 JupyterLab：

```powershell
cd "I:\贝塔频带汇报-2026-10"
.\.venv\Scripts\python.exe -m jupyter lab
```

然后打开：

```text
channelXtrial/channelXtrial_pipeline.ipynb
```

默认使用：

```text
I:\ECoG_data\HS0015\blocks\B03\standardized
```

## 输出文件

默认输出目录：

```text
outputs/HS0015/B03/channelXtrial
```

主要文件：

- `channel_trial_type_features.csv`
  - 每个 `channel x trial` 的形态特征。

- `channel_trial_type_labels.csv`
  - 每个 `channel x trial` 的 Type1/Type2/Uncertain 判定、分数和置信度。

- `channel_type_stability.csv`
  - 每个通道的 dominant type、稳定率、切换次数、uncertain 比例和前后半段变化。

- `channel_trial_type_heatmap.png`
  - 行为通道、列为有效 trial 的 Type 热图。

- `block_type_proportions.png`
  - 每个有效 trial 中 Type1、Type2 和 Uncertain 通道比例。

- `*_type_sequence.png`
  - 自动选择的示例通道 trial-wise type、分数和形态特征轨迹。

- `*_trial_curves.png`
  - 自动选择的示例通道单 trial beta 曲线小面板。

HS0015/B03 当前 smoke test 结果：

```text
有效 trial: 44
好通道: 126
channel x trial 样本: 5544
Type2: 2564
Type1: 1829
Uncertain: 1151
```
