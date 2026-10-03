# ECoG Beta 进程特征与两轮聚类管道

本项目实现连续侵入式脑电（ECoG）的 beta 频带功率提取、语音 onset/offset 对齐、trial 时长过滤、0-100% 语音进程归一化、音频包络斜率耦合特征工程、带权两轮 K-means 聚类与可选的脑表面映射。

代码中仍保留原三频带特征提取函数，便于复现早期分析；当前 Notebook 默认使用 beta-only 百分比进程流程。

脑表面数据不是运行主流程的必要条件。没有 FreeSurfer、MRI、pial surface 或肿瘤掩膜时，仍可完整执行特征提取、聚类、PCA、动态曲线绘制和坏道复审。

## 项目结构

- `audio_cross_3band_utils.py`：加载 FIF、事件和 WAV，计算 Hilbert 功率，支持原三频带固定窗特征，也支持 beta-only 百分比 trial 进程特征。
- `clustering_pipeline.py`：仅使用好通道执行缺失值填充、标准化、带权 K-means、PCA 和两轮坏道复审；第二轮会重新拟合全部预处理器，避免数据泄漏。
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
