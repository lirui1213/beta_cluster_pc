# 全通道两轮全局 beta 聚类

打开 `all_channel_pipeline.ipynb`，选择能运行根项目的 Python 内核。只需设置数据根目录与手动病人/block 选择：

```python
DATA_ROOT = Path(r"I:\ECoG_data")
INCLUDED_BLOCKS = {
    "HS0005": ["B01", "B03"],
    "HS0006": ["B07", "B08", "B09"],
}
FIRST_N_CLUSTERS = 10
SECOND_N_CLUSTERS = 4
```

上述仅为格式示例；更新保留了你已填写的 Notebook 病人/block 参数。先运行初始化和参数区，再运行文件检查和第一轮分类；第 6 节先自动筛除一次，再默认弹出三维 PCA 人工复审窗口，点击 Apply 后执行第二轮。两轮类数分别由 FIRST_N_CLUSTERS、SECOND_N_CLUSTERS 手动指定，默认 2，窗口内也可指定第二轮 K。每次输出使用独立时间戳目录，不覆盖旧运行。

## 第一轮 PCA 复审与第二轮

第一轮使用所有质量合格样本，PCA 只用于查看分布，聚类仍使用筛选后的加权特征。PCA 距离表列出每个 sample_id 的前三个 PC 和到所属簇中心的欧氏距离，按距离排序辅助定位异常点。

自动筛除默认开启，只剔除同时满足以下两项的第一轮簇：

- 簇成员数 / 第一轮合格样本总数 ≤ `AUTO_SMALL_CLUSTER_FRACTION`（默认 0.05）。这里成员是病人 × block × 通道样本，不是 trial 数或去重电极数。
- 前三个 PC 的簇内样本方差之和（协方差迹，ddof=1）> 非小簇方差中位数 × `AUTO_PCA_VARIANCE_RATIO`（默认 3.0），附加数值精度下限。

自动筛除只做一次，没有中间重新聚类，保留第一轮坐标和簇编号。单成员簇无法估计方差，不自动剔除；没有非小簇参照时记录原因并不自动剔除。小但紧密的簇、成员多的簇不会因本规则被删除。默认值是可调整的复审启发式参数，不是生理异常或稳定性检验标准。

随后窗口中手动勾选坏簇或输入样本 ID，自动与手动排除取并集：

```python
BAD_CLUSTERS = (3,)  # 排除第一轮整个异常簇
MANUAL_BAD_SAMPLES = ("HS0012:B01:ECoG_124",)  # 病人:block:通道
PCA_DISTANCE_THRESHOLD = None  # 默认关闭；填写正数后排除距离 > 阈值的样本
AUTO_PRUNE_ENABLED = True
AUTO_SMALL_CLUSTER_FRACTION = 0.05
AUTO_PCA_VARIANCE_RATIO = 3.0
OPEN_REVIEW_WINDOW = True  # 默认沿用原 Notebook 的 Tk 三维 PCA 复审窗口
```

以上样本 ID 和簇号为示例，必须从实际第一轮结果选择。单独写 ECoG_124 会报错，防止误删其它病人或 block 的同名通道。窗口中手动电极字段也填写 sample_id，窗口返回的选择与单元参数合并；在窗口内也可修改第二轮 K（窗口范围 2–10）。自动剔除点显示为白色 Bad，其簇不再提供勾选，自动剔除不能被人工恢复。关闭窗口未 Apply 时终止该单元，第二轮不会执行；重新运行第 6 节即可再复审。无图形桌面时可明确关闭弹窗，使用手动参数。

额外的 PCA_DISTANCE_THRESHOLD 仍是可选的点到簇中心绝对距离规则，与本次“小且高方差簇”规则独立，默认关闭。不足三个可计算 PC 时，未估计维度仅在筛除/窗口坐标副本中补零，原模型和输出 PCA 不改动。孤立点若已单独形成一个簇，其簇内距离可能很小，需要人工按整簇或 sample_id 排除。自动、手动和质量排除理由分别保存到 review_audit.csv。

第二轮重新筛选特征，并独立拟合 imputer、scaler、K-means 与 PCA，避免沿用被排除点贡献的统计量。原始质量排除不会恢复，所有排除样本仍保留 -1 标签。复用第一次提取的特征和曲线，修改自动阈值或复审只需重跑第 6–8 节，无需重读 FIF；修改输入或特征提取参数则重新执行第一轮，并重新复审。

## 数据路径

每个所选组合自动解析以下路径：

```text
DATA_ROOT/<subject>/blocks/<block>/standardized/ECoG/ecog_TEST_clean_car-raw.fif
DATA_ROOT/<subject>/blocks/<block>/standardized/events/onset.npy
DATA_ROOT/<subject>/blocks/<block>/standardized/events/offset.npy
DATA_ROOT/<subject>/blocks/<block>/standardized/audio/analog1.wav
DATA_ROOT/<subject>/baseline/baseline.fif
```

基线默认 `file`，与根项目当前 Notebook 一致；用于原流程的质量和诊断检查。新分类/图形的数值直接来自连续 Hilbert 功率，不使用基线 Z。同一病人所有 block 使用同一个文件；自动读取相邻 `baseline__segments.csv`（若有）。基线与各任务记录必须使用一致的参考和预处理；原接口会核查通道及采样率等元数据，但无法证明所有历史处理一致。明确设置 `BASELINE_MODE="pre_onset"` 时不要求基线文件，改用逐 trial 的 onset 前 `[-1.3,-0.3]` 秒。所有 block 使用同一模式，不自动退回另一种基线。

文件缺失会在正式提取前报错并列出路径。事件、采样率、基线等内容检查由现有提取流程执行；任一 block 失败会终止运行，保留已导出的诊断，`run_status.json` 标为 failed。不能将失败目录中的部分结果作为完整全局结果。

## 样本与聚类

每行是 **病人 × block × 通道**。读取所有 ECoG 类型通道（包括原始坏道），排除音频/刺激等其他类型。trial 先按现有合法性、时长和边界规则过滤；从连续 beta 13–30 Hz Hilbert 功率直接生成配对 trial 曲线。连续功率和音频先统一平滑，然后每个 trial 一次取样；不拼接分别平滑的窗口。再按通道平均，去水平、去幅度，完整缓存与图形均为无量纲变化。语音长度仍用于统一进程，前后仍为原生连续记录中的真实时间上下文。

原始坏道、无效 trial 比例超过 0.25、有效 trial 少于 2 的通道保留排除原因，标签为 -1，PCA 为 NaN。各病人的同名通道及同一通道的不同 block 使用不同 sample_id；病人、block、trial 数及质量字段不进入特征筛选或聚类。

分类输入是直接取自连续信号的完整曲线数值，不是图片像素。每条曲线统一减去该通道语音前平均功率，去掉整体水平偏移，再除以该通道整条 pre/speech/post 曲线的 RMS，**去除响应幅度差异**。形状相同而仅幅度不同的通道，其曲线分类特征相同；保留正负方向和各段相对形状，平坦曲线保持零。三段使用同一个逐通道尺度，不分别归一化。原始绝对功率、能量及旧 `beta_progress__` 特征保留作诊断，全部不进入此模式的模型。

逐通道归一化后再应用合格样本池的共享 RMS，方便与音频关系指标配权；该共享步骤不会恢复原始幅度差异。另加入 beta/audio 平均曲线斜率的零延迟相关、最大绝对互相关对应的带符号相关及延迟（搜索 ±10% 语音段；正延迟表示 audio 比 beta 晚）。采用加权欧氏距离与 K-means，比较归一化曲线形状。pre/speech/post 权重预算为 0.2/0.6/0.2，音频关系预算为 0.25，各组内按保留列数分配。去常量和过多缺失的列，保留相邻相关时间点；`CORRELATION_THRESHOLD` 仅为旧模式兼容参数，在曲线模式不用于删除时间点。第二轮重新拟合共享尺度与全部模型。

`CURVE_CONTEXT_SECONDS=0.75`：提取 onset 前和 offset 后固定 0.75 秒，`CURVE_CONTEXT_POINTS=51`；speech 用 `PROGRESS_POINTS=101` 个进程点。`PRE_POST_WINDOW_SECONDS=None` 使用上述固定窗口，明确填写更短值则使用更短窗口。边界不足的 trial 由原提取流程过滤。曲线生成与分类使用 `DISPLAY_SMOOTH_SIGMA_SECONDS=0.10` 秒平滑；旧 `FEATURE_SMOOTH_SIGMA_SECONDS=0.06` 仅作用于原诊断指标。

样本等权；block 多、通道多的病人贡献较大。两轮 K 均是人工参数，不自动选择，不称为已通过稳定性评估。

本项目执行两轮同一粒度的全局聚类，不执行跨 block 汇总、单 trial 分类、分半/重采样、留一病人及其它稳定性分析。全局簇只命名 Cluster 0、1…，不自动解释为 Type1/Type2。两轮及各次运行标签编号可能变化。

## 输出

第一轮输出位于 `all-channel/outputs/<运行时间>/`，第二轮位于其 `round2_<运行时间>/` 子目录。两轮各自保存标签、模型、PCA、beta/音频图和构成统计；第一轮逐 block 的特征和基线诊断无需重复导出：

每次第 6 节在 `review_<运行时间>/` 保存 `auto_cluster_screening.csv`、`auto_sample_screening.csv`、`auto_policy.json`，先保存报告再打开窗口。第二轮目录也保存对应的自动筛除报告及参数快照，便于复核每个簇的成员数、方差、参照方差和剔除原因。

- `input_manifest.csv`、`run_config.json`、`run_status.json`：实际输入、参数和完成状态。
- `blocks/<病人>/<block>/`：通道特征、保留/剔除 trial、基线诊断。
- `all_channel_features.csv`：完整样本特征；`global_assignments.csv`：全局标签、身份、质量及 PCA；`excluded_channels.csv`：排除通道。
- `selected_feature_weights.csv`：实际进入模型的特征及权重；`global_model.joblib`：拟合预处理器、K-means 和 PCA。
- `global_beta_curves.png`、`global_beta_curve_summary.csv`：语音进程下的各簇均值与样本间 SEM。每个 block/channel 先平滑并平均其配对有效 trial，再在簇内按样本等权平均。显示和曲线分类均使用物理时间 sigma=0.10 秒。此文件只显示语音段，完整图见下项。所有 beta 曲线均为无量纲归一化变化。
- `cluster_grid_beta_audio.png`、`complete_curve_summary.csv`：示例样式的完整 pre/speech/post 图，彩色 beta 均值与样本 SEM、橙色音频双轴、onset/offset 虚线。图和分类均使用减去语音前均值并逐曲线 RMS 归一化的无量纲 beta 形状，不保留 beta Z 显示；语音毫秒轴映射到全部纳入 trial 时长中位数，pre/post 保留真实秒数。
- `channel_figures/<病人>/<block>/*.png`、`channel_figure_index.csv`：第一轮逐通道完整图与 trial SEM，默认 `SAVE_CHANNEL_FIGURES=True`，可关闭以减少保存耗时；完全无有效曲线的通道在索引记录原因。第二轮复用这些通道曲线，不重复生成通道图。
- `beta_audio_dynamics.png`、`.csv`：每簇三列图，beta/音频包络、beta/音频斜率、同 trial beta 斜率 × 音频斜率。音频按各 block 全部保留 speech trial 的包络峰值归一化；音频关系分类使用斜率相关与相对延迟，不使用音频绝对幅度。逐 trial 按实际时长换算平滑，斜率单位为每百分比点。先计算同 trial 乘积，再平均，避免跨 trial 混配。包络和斜率均用独立纵轴，曲线交叉不代表量纲相等。
- `mean_speech_curves.npz`：按 sample_id 对齐的 beta、归一化音频、两种斜率、斜率乘积、pre/post beta/audio、trial SEM、百分比轴及显示时长。使用与通道 beta 相同的有效 trial，避免配对错位。
- `sample_audio_relationship.csv`：标签及原诊断特征中的音频相关性、耦合和斜率乘积指标；没有新拟合或稳定性分析。
- `pca_review_distances.csv`：本轮 PCA 距离表；第二轮额外 `review_audit.csv` 保存第一轮距离、三种排除标记、最终排除和第二轮标签。
- `pca_clusters.png`：全局特征 PCA；`cluster_subject_counts.*`、`cluster_block_counts.*`：各簇的数据来源构成。
- `block_summary.csv`、`global_summary.json`：样本数量、排除情况、最小簇规模及描述性轮廓系数。轮廓系数最多固定随机抽取 5000 个样本计算；它不代表分类稳定性。

同一电极跨 block 的样本并不独立，样本 SEM 仅用于描述分散程度，不能解释为对病人总体的置信区间或稳定性证据。

测试：在根目录运行 `python -m pytest tests/test_all_channel.py tests/test_all_channel_review.py tests/test_all_channel_auto_prune.py tests/test_all_channel_popup.py tests/test_all_channel_curve_similarity.py tests/test_all_channel_native_curves.py -q`。弹窗测试构造真实 Tk 控件并在隐藏窗口中触发 Apply/取消回调，不弹出测试用可见窗口。
