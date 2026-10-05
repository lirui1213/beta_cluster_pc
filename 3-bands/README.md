# 四频带联合分类

入口：`ECoG_4band_clustering_pipeline.ipynb`。目录沿用 `3-bands` 名称，实际分析为四个频带，原项目根目录的 Beta Notebook 不受影响。

| 频带 | 频率范围 |
| --- | --- |
| Theta | 4-8 Hz |
| Beta1 | 13-24 Hz |
| Beta2 | 24-40 Hz |
| High-gamma | 70-150 Hz |

每个频带分别带通滤波、计算 Hilbert 功率和基线 Z-score，再提取形态、斜率及音频耦合特征。四频带合并后，每个电极得到一套联合标签，不是四份独立分类。有限阶滤波器的过渡带可能重叠，24 Hz 不是理想硬切边界。

## 参数和两轮操作

1. 在参数区设置患者、block 和第一轮 `FIRST_N_CLUSTERS`，从初始化开始顺序执行。`SECOND_N_CLUSTERS` 仅作为弹窗首次打开时的 K 初值，最终第二轮 K 由窗口确认。
2. 第一轮结果给出 PCA、四频带功率/音频曲线及每个频带的音频关系图。
3. 第 6 节准备 PCA 距离表，第 7 节默认打开复审弹窗。在窗口中启用/关闭距离阈值、输入阈值、选择全局/簇内参考、勾选整类、输入指定电极并调整第二轮 K。距离阈值默认关闭，不自动猜测。全局距离表示前三个 PC 到合格电极坐标中位数的欧氏距离，簇内距离表示到所属簇中心的距离。孤立点可能具有零簇内距离；前三个 PC 也不能覆盖全部特征差异。
4. 窗口实时预览 PCA 剔除标记、原始坏道、新增剔除及剩余数量。所有剔除条件取并集，重复选择不重复计数，原始坏道不恢复。K 必须为 2 至剩余电极数之间的整数；非法输入会禁止确认。
5. 点击“确认并重新聚类”后重新筛选特征、拟合填补器、标准化、K-means 和 PCA，仍展示四频带与音频的关系。点击取消或关闭窗口不会执行第二轮，也不会显示旧第二轮的脑表面结果。重跑第 7 节会再次弹出窗口，并继承同一第一轮模型下最后确认的设置；重新拟合第一轮后复审设置重置。

分类使用完整加权特征，PCA 仅用于观察和剔除，并非直接在两个 PC 上分类。特征冗余筛选在每个频带内部进行，默认每个频带总权重均为 1，避免因特征数量不同而隐含增权。`BAND_WEIGHTS` 可手动调整。任何频带没有可用变化特征都会报错，不会偷偷退化成单频带分类。

改变频带、基线、患者、block、特征平滑等提取参数，必须重新执行文件检查、提取及两轮分类。仅改变第一轮 K 时重跑第一轮，再复审及第二轮；仅改变复审条件或第二轮 K 时重跑第 7 节并在弹窗中修改。显示平滑参数只影响曲线，不影响已提取分类特征。`OPEN_REVIEW_WINDOW=False` 留给明确的批处理验证使用，不是日常默认流程。

## 基线和显示

- `BASELINE_MODE="file"`：使用 `I:\ECoG_data\患者\baseline\baseline.fif` 的合格区间，逐电极、逐频带计算均值和标准差。`BASELINE_SEGMENTS_CSV` 可指定区间表；边缘默认排除 0.3 秒。
- `BASELINE_MODE="pre_onset"`：每个 trial 使用 onset 前 `[-1.3, -0.3]` 秒，共 1 秒。只取到 onset 前 0.3 秒，不使用临近 onset 的最后 0.3 秒。短窗口对 Theta 功率标准差的估计不如长且干净的独立 baseline 稳定；零相位滤波/Hilbert 有时间扩散，0.3 秒间隔不保证基线完全不受事件影响，尤其是 Theta。因此默认仍使用独立 baseline 文件。
- 归一化为 `(功率 - 基线均值) / 基线标准差`。低方差或不可用基线标记为无效，而非用固定大 epsilon 压缩信号。任一频带原始/QC 不合格都会排除该电极。
- 特征提取平滑默认 0.06 秒，显示平滑 0.10 秒；显示统一约 100 Hz 网格。先平滑电极平均曲线，再算簇均值和电极间 SEM。
- 音频显示按所有保留 trial 的语音段全局峰值归一化，右轴独立。斜率图按语音进程百分比求导，不是每秒斜率；乘积先在匹配的同一 trial 内求值，再平均，避免跨 trial 混合。
- High-gamma 上限 150 Hz 要求采样率严格高于 300 Hz；不满足时明确报错。默认数据 400 Hz。必要时应额外检查工频谐波、肌电等伪迹，曲线好看并不保证可靠。

## 输出和验证

输出到 `outputs/患者/block/时间戳/`，每次分类保存独立轮次目录，不覆盖旧复审。包含基线质量报告、完整特征、选中特征权重、标签与排除原因、PCA 距离、复审记录、模型以及两轮曲线/音频关系 CSV 和 PNG。`summary.json` 的 silhouette 仅为描述性数值，**未进行稳定性选 K**，标签和颜色不具有固定生理意义。

音频关系导出表使用明确前缀：`feature_mean_curve` 为特征提取时平均曲线的关系（音频保留提取单位），`feature_trialwise` 为提取时 trial 级描述，`display_paired_trial_slope_product_mean` 为显示设置下先在同一 trial 内计算斜率乘积、再平均的量（音频按全局峰值归一化），与音频动态图一致。这些量不应直接互换；方法和单位记录于 `relationship_methods.json`。

`RUN_3D_RENDER`、`RUN_SELECTABLE_CLUSTER_RENDER` 可开启原有 PyVista 脑表面及分类选择弹窗，使用第二轮联合标签。调整 `SURFACE_COLOR` / `SURFACE_OPACITY` 不改变分类。

验证脚本只运行无弹窗副本，不修改原 Notebook：

```powershell
& 'E:\anaconda\envs\ECoG_LR_v1\python.exe' 3-bands/verify_notebook.py --baseline-mode file
& 'E:\anaconda\envs\ECoG_LR_v1\python.exe' 3-bands/verify_notebook.py --baseline-mode pre_onset --first-k 3 --second-k 2
```

可选 `--subject`、`--block`、`--manual-bad 电极名` 和 `--distance-threshold 数值`；运行记录存放于 `outputs/verification/`。不要在已修改 Notebook 后直接运行 `create_notebook.py`，无参数会重新生成初始版本。`--update-review` 只用于更新本次复审相关单元及导入/脑表面确认检查，会保留数据和分析参数。
