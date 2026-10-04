# All-channel Implementation Plan

**Goal:** 根据手动病人/block 选择，建立一次全局 beta 聚类及可审阅的结果。

**Architecture:** all-channel/inputs.py 校验选择和生成路径；pipeline.py 复用根项目特征与聚类接口；plots.py 绘制全局结果；all_channel_pipeline.ipynb 提供中文入口。

**Tech Stack:** Python、MNE、NumPy、pandas、scikit-learn、Matplotlib、Jupyter。

## Global Constraints
- 不改根项目算法；不执行其他分析方案；空选择不读取真实记录。
- 每行病人 × block × 通道；保留被排除通道；仅 beta 特征参与拟合。
- 任一输入缺失时在提取前停止；输出使用新建运行目录防止旧结果混入。

## Tasks
1. 输入：先写 tests/test_all_channel.py 的选择、重复、缺失和文件/事件路径用例；运行 pytest 确认接口缺失；实现 build_manifest(DATA_ROOT, INCLUDED_BLOCKS, baseline_mode) 与 validate_manifest(manifest)。测试空选择、路径越界和真实目录可用清单。
2. 全局拟合：测试重复通道跨病人/block 的身份保留、数值元数据排除及坏道不影响模型。实现 fit_global(features, n_clusters, random_state, correlation_threshold)，显式只传 beta__ 特征；输出标签 -1 及 PCA NaN。
3. 批处理与可视化：写包含真实合成 FIF、事件和 WAV 的两 block 端到端测试；实现 run_analysis(manifest, output_dir, ...)，逐 block 导出质量诊断，缓存 trial 平均 speech 曲线，汇总统一聚类和输出。
4. Notebook 与说明：生成中文参数、输入检查和执行单元；配置 INCLUDED_BLOCKS={}；验证 Notebook 单元语法及输入检查，运行新项目测试与原有相关回归测试。扫描真实数据文件，但不选择/运行真实患者分析。

Verification: `python -m pytest tests/test_all_channel.py -q`，随后根项目完整 `python -m pytest -q`。Notebook 使用 nbformat.validate 并编译所有代码单元。端到端测试必须确认导出的标签、曲线、质量记录和配置可读取。
