# ECoG Beta 进程特征与两轮聚类管道

本项目实现连续侵入式脑电（ECoG）的 beta 频带功率提取、语音 onset/offset 对齐、trial 时长过滤、0-100% 语音进程归一化、形态与音频斜率耦合特征、带权两轮 K-means 聚类及 trial 稳定性评估，并支持脑表面映射。

代码中仍保留原三频带特征提取函数，便于复现早期分析；当前 Notebook 默认使用 beta-only 百分比进程流程。

脑表面数据不是运行主流程的必要条件。没有 FreeSurfer、MRI、pial surface 或肿瘤掩膜时，仍可完整执行特征提取、聚类、PCA、动态曲线绘制和坏道复审。

## 项目结构

- `audio_cross_3band_utils.py`：加载 FIF、事件和 WAV，计算 Hilbert 功率，支持原三频带固定窗特征，也支持 beta-only 百分比 trial 进程特征。
- `clustering_pipeline.py`：仅使用好通道执行缺失值填充、标准化、带权 K-means、PCA 和两轮坏道复审；第二轮会重新拟合全部预处理器，避免数据泄漏。
- `signal_processing.py`：与功率尺度匹配的基线质量检查、NaN 安全平滑、多相抗混叠重采样和 Savitzky-Golay 导数。
- `cluster_evaluation.py`：K=2-6 的轮廓系数、独立分半 trial ARI、trial bootstrap ARI 和簇内波形一致性。
- `visualization.py`：绘制簇级均值和 SEM 动态曲线、beta 百分比进程曲线、两个互不重叠 trial 示例图，提供 Tk 坏道复审窗口，以及可选的电极三维显示和 FreeSurfer/PyVista 脑表面渲染。
- `ECoG_3band_clustering_pipeline.ipynb`：中文 Notebook 入口，默认运行 beta-only 百分比进程流程，从参数设置到特征提取、带权两轮聚类、导出结果和可选三维显示。
- `channelXtrial/`：单通道跨 trial 的 beta Type1/Type2 动态模式稳定性分析，输出 channel x trial 热图、通道稳定性表和单通道轨迹图。
- `tests/test_pipeline.py`：事件、分段、特征维度、坏道标签和二轮重拟合测试。
- `requirements.txt`：核心运行依赖。
- `requirements-dev.txt`：测试依赖。
- `requirements-notebook.txt`：Notebook 与 JupyterLab 依赖。
- `requirements-brain.txt`：可选脑表面显示依赖。

## 输入数据

主流程支持以下输入：

- ECoG：`*_clean_car-raw.fif`，保留 `raw.info["bads"]`。
- 事件：CSV、DataFrame、二维 NPY，或分别存储的 `onset.npy` 与 `offset.npy`。
- 音频：单声道或多声道 WAV。
- 可选解剖数据：Scanner RAS 电极坐标 CSV、FreeSurfer 参考 MRI 和 pial surface，以及可选肿瘤掩膜。

## 本地环境

项目环境已创建在当前目录的 `.venv` 中。

在 PowerShell 中激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

也可以不激活环境，直接调用项目解释器：

```powershell
.\.venv\Scripts\python.exe your_script.py
```

如需重新创建环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Notebook 运行

推荐直接打开：

```text
ECoG_3band_clustering_pipeline.ipynb
```

在 VS Code 中选择当前目录的 `.venv` 作为 Notebook kernel，然后从上到下运行。当前已填入 `HS0015` / `B03` 数据路径，通常只需要确认 Notebook 第 2 节“参数区”中的 `SUBJECT_ID`、`BLOCK_ID`、`DATA_DIR`、`EVENT_SOURCE`、`OFFSET_SOURCE` 和 `WAV_PATH`。

如果希望在浏览器中运行 JupyterLab：

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

Notebook 运行依赖可通过以下命令安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
```

## 最小分析示例

```python
import pandas as pd

from audio_cross_3band_utils import build_beta_progress_feature_table_from_files
from clustering_pipeline import (
    cluster_electrodes,
    default_beta_progress_feature_weights,
)

features, beta_epochs = build_beta_progress_feature_table_from_files(
    fif_path="ecog_TEST_clean_car-raw.fif",
    event_source="onset.npy",
    offset_source="offset.npy",
    wav_path="analog1.wav",
    progress_points=101,
    duration_mad_threshold=2.5,
)

feature_columns = [
    column for column in features.columns
    if column not in {"channel", "is_bad"}
    and pd.api.types.is_numeric_dtype(features[column])
]
weights = default_beta_progress_feature_weights(feature_columns)

result = cluster_electrodes(
    features,
    n_clusters=5,
    feature_weights=weights,
)

assignments = result.to_frame()
```

返回的 `features` 每行对应一个电极，包含：

- `channel`：电极名称。
- `is_bad`：是否属于 FIF 中的初始坏道。
- beta-only 百分比进程特征：能量水平、beta 斜率、音频包络斜率交互、斜率互相关、同向比例、交叉点和 trialwise slope-audio 耦合等。

当前默认只使用：

- Beta：13–30 Hz。

trial 的 onset 到 offset 会被重采样到 0-100% 语音进程。onset 前和 offset 后的真实时间窗口由保留 trial 中最小的 `onset[n+1] - offset[n]` 决定。默认使用 duration 的 median/MAD 规则剔除时长变异过大的 trial。

## 坏道交互复审

第一轮聚类后，可打开 Tk 复审窗口：

```python
from visualization import BadChannelReviewWindow

review = BadChannelReviewWindow(result.first_round).show()
```

窗口支持：

- 勾选需要整体排除的异常簇。
- 手动输入需要追加的坏电极。
- 将第二轮聚类数量设置为 2–10。

第二轮聚类会在排除全部坏道后重新拟合 `SimpleImputer`、`StandardScaler` 和 `KMeans`。所有坏道标签始终为 `-1`，PCA 坐标为 NaN。

## 动态曲线

### 2026-10-03 分析优化

优化前源码已保存到 GitHub 分支 `2026/10/03`，提交 `3b8d161`。优化在本地分支 `codex/beta-optimization-2026-10-03` 中继续，保留远端快照。

基线分母改为真实的基线功率标准差，不再默认加 `1e-6`。质量下限由每个电极的典型基线标准差乘 `BASELINE_RELATIVE_FLOOR`（默认 `1e-3`）决定，同时使用相对于基线功率的数值精度下限。低方差或非有限基线会使该 trial/channel 的数据成为 NaN，并保存到 `baseline_quality.csv`。无效 trial 比例超过 `MAX_BASELINE_INVALID_FRACTION`（默认 0.25），或不足两个有效 trial 的电极不参与聚类。`baseline_invalid_fraction` 和 `valid_trial_count` 仅用于质量检查，不参与聚类。修正后 Z 值的数值尺度会改变，应从初始化 cell 开始重新运行，不能沿用旧的特征文件或标签。

分类用的平滑参数 `FEATURE_SMOOTH_SIGMA_SECONDS` 默认 0.06 秒，按每个 trial 的实际时长换算；先平滑 beta 功率和音频包络，再用 `FEATURE_SLOPE_METHOD="savgol"` 估计导数。导数仍以说话进程百分比为单位。语音段降采样采用多相抗混叠滤波。

新增早/中/晚响应、正负响应面积、正负峰及位置、主峰半突出度宽度、offset 回落/反弹等形态特征。特征筛选只使用好电极，优先保留可解释的形态和相关性特征，剔除常量、缺失过多及绝对相关性达到 `FEATURE_CORRELATION_THRESHOLD`（默认 0.95）的重复特征。`NORMALIZE_FEATURE_GROUP_WEIGHTS=True` 时，权重表示各特征组的总预算，避免同一组仅因列数更多而占据更大的距离权重。第二轮在坏道复审后重新进行特征筛选。

`RUN_CLUSTER_STABILITY=True` 会在第二轮前比较 K=2-6。每种 trial 切分或重采样都重新提取特征，并独立拟合 imputer、scaler、KMeans。好电极名单由原始数据质量确定，不随重采样次数改变；无法支持固定电极名单的重采样会记为失败，而不会永久排除电极。至少需要四个有效 trial 才能分为两个独立 trial 子集；较少时保留人工设置的 K。完整但出现小簇的重采样仍计入 ARI，避免仅保留稳定结果而高估一致性。候选 K 的全数据最小簇需至少有 max(2, ceil(0.05 * 好电极数)) 个成员，且两种评估分别至少有 80% 完整有效重采样；符合条件者按轮廓系数、分半 ARI 和 bootstrap ARI 的平均名次推荐 K，同名次选较小 K。`AUTO_SELECT_K=True` 使用推荐 K；设为 False 可保留复审窗口设置。指标和逐次重采样结果分别保存到 `cluster_stability_summary.csv` 与 `cluster_stability_resamples.csv`，图为 `cluster_stability.png`。这些指标衡量描述性分离度和重现性，不是显著性检验或固定生理类别。

Cell 7 先把 pre、speech、post 映射到统一的等间隔显示网格，再对每个电极的 trial 平均曲线平滑，最后计算簇均值和电极间 SEM。`DISPLAY_SFREQ=100.0`，`BETA_CURVE_SMOOTH_SIGMA_SECONDS=0.10`。额外的 `beta_smoothing_comparison.png` 对比 0.06、0.10、0.15 秒三个 sigma。图中的语音区间经过时长归一化，毫秒轴是按有效 trial 的中位时长映射的显示轴，不代表所有 trial 的真实物理时间。

将 Cell 7 的 `CURVE_UNCERTAINTY` 改为 `"trial_bootstrap"` 可显示逐点 95% trial 重采样区间，`CURVE_BOOTSTRAP_ITERATIONS` 默认 200。此区间固定电极和簇成员，反映 trial 变化，不包括重新选择 K 或电极归类的不确定性，也不是全曲线同时置信带。

完整数值与绘图验证可运行 `python scripts/verify_beta_optimization.py --mode inprocess`，在当前 Python 环境中顺序执行 Notebook 的分析 cell，关闭验证副本中的复审和三维弹窗，采用 Agg 后端导出图。结果保存在 `outputs/optimization_verification/`，不会覆盖正常分析输出或修改原 Notebook 的交互设置。若环境支持 Jupyter 内核，也可使用 `--mode kernel --kernel ecog_lr_v1`。

```python
from visualization import (
    plot_beta_progress_cluster_dynamics,
    plot_two_nonoverlapping_beta_trials,
)

figure, axes = plot_beta_progress_cluster_dynamics(
    beta_epochs,
    result.labels,
)

trial_figure, trial_axes = plot_two_nonoverlapping_beta_trials(beta_epochs)
```

主要输出 `cluster_grid_beta_only.png`，效果为每个簇一个子图，只保留 beta：彩色曲线是簇内 ECoG beta Z-power 均值，阴影是 SEM，橙色曲线是语音包络。图中虚线标记 onset 和归一化后的 median offset，灰色背景显示 onset 前与 offset 后窗口，窗口大小由有效 trial 的最小 intertrial gap 决定。额外的 `cluster_dynamics_beta_progress_diagnostic.png` 会显示 beta 斜率及 beta 斜率与音频包络斜率的交互项。两个 trial 示例图用于直观看到互不重叠 trial 在相同百分比进程下的 beta 和音频轨迹。

## 可选三维显示

特征提取、聚类、PCA、动态曲线和坏道复审均不需要安装脑表面依赖。

仅在需要三维电极或脑表面显示时安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-brain.txt
```

### 仅显示电极

没有脑表面数据时，可直接在 Scanner RAS 空间显示电极：

```python
from visualization import render_electrode_clusters

label_mapping = dict(
    zip(final_assignments["channel"], final_assignments["label"])
)

render_electrode_clusters(
    "electrodes.csv",
    label_mapping,
)
```

### 叠加 FreeSurfer 脑表面

只有显式提供 `subjects_dir` 和 `subject` 时，程序才会检查 FreeSurfer 文件：

```python
render_electrode_clusters(
    r"I:\ECoG_data\HS0015\recon\electrodes\electrodes.csv",
    label_mapping,
    subjects_dir=r"I:\ECoG_data\HS0015",
    subject="recon",
    reference_mri=r"I:\ECoG_data\HS0015\recon\mri\T1.mgz",
    hemispheres=("lh",),
    tumor_nifti=None,
)
```

当前 `HS0015/recon` 默认使用：

- `electrodes/electrodes.csv`
- `mri/T1.mgz`
- `surf/lh.pial`
- 可选的肿瘤掩膜

坐标使用以下矩阵从 Scanner RAS 转换到 FreeSurfer tkRAS：

```text
vox2ras_tkr @ inv(vox2ras)
```

前五个簇依次使用红、蓝、绿、紫、灰，坏道固定使用白色。

## 测试

安装开发依赖并运行测试：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

当前测试覆盖事件加载、Nyquist 边界、分段基线、原三频带特征、beta 百分比进程特征、duration outlier 剔除、音频插值、特征权重、全缺失特征处理、坏道合并和第二轮独立重拟合。
