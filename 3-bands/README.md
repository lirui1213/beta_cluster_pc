# 四频带联合分类

入口：`ECoG_4band_clustering_pipeline.ipynb`。目录沿用 `3-bands` 名称，实际分析为四个频带，原项目根目录的 Beta Notebook 不受影响。

| 频带 | 频率范围 |
| --- | --- |
| Theta | 4-8 Hz |
| Beta1 | 13-24 Hz |
| Beta2 | 24-40 Hz |
| High-gamma | 70-150 Hz |

每个频带分别带通滤波、计算 Hilbert 功率和基线 Z-score，再提取形态、功率及自身斜率特征。**两轮分类都只使用 ECoG 自身特征**，不使用与音频包络有关的相关性、回归、时滞、斜率乘积、同向比例或交叉次数。音频关系仍可计算和显示，但仅为描述性对照。四频带合并后，每个电极得到一套联合标签，不是四份独立分类。有限阶滤波器的过渡带可能重叠，24 Hz 不是理想硬切边界。

## 参数和两轮操作

1. 在参数区设置患者、block 和第一轮 `FIRST_N_CLUSTERS`，从初始化开始顺序执行。`SECOND_N_CLUSTERS` 仅作为弹窗首次打开时的 K 初值，最终第二轮 K 由窗口确认。
2. 第一轮结果给出 PCA、四频带功率/音频曲线及每个频带的音频关系图。
3. 第 6 节准备 PCA 距离表，第 7 节默认打开复审弹窗。在窗口中启用/关闭距离阈值、输入阈值、选择全局/簇内参考、勾选整类、输入指定电极并调整第二轮 K。距离阈值默认关闭，不自动猜测。全局距离表示前三个 PC 到合格电极坐标中位数的欧氏距离，簇内距离表示到所属簇中心的距离。孤立点可能具有零簇内距离；前三个 PC 也不能覆盖全部特征差异。
4. 窗口实时预览 PCA 剔除标记、原始坏道、新增剔除及剩余数量。所有剔除条件取并集，重复选择不重复计数，原始坏道不恢复。K 必须为 2 至剩余电极数之间的整数；非法输入会禁止确认。
5. 点击“确认并重新聚类”后重新筛选特征、拟合填补器、标准化、K-means 和 PCA，仍展示四频带与音频的关系。点击取消或关闭窗口不会执行第二轮，也不会显示旧第二轮的脑表面结果。重跑第 7 节会再次弹出窗口，并继承同一第一轮模型下最后确认的设置；重新拟合第一轮后复审设置重置。

分类使用完整加权特征，PCA 仅用于观察和剔除，并非直接在两个 PC 上分类。特征冗余筛选在每个频带内部进行，默认每个频带总权重均为 1，避免因特征数量不同而隐含增权。`BAND_WEIGHTS` 可手动调整。任何频带没有可用变化特征都会报错，不会偷偷退化成单频带分类。

音频耦合列在冗余筛选之前就被排除，旧特征表也不能把它们带回分类。去掉这些特征可能改变分类标签，不保证与旧结果相同。现有多 trial 平均曲线按语音时长归一化；它们不是两个 trial 的连续记录。

改变频带、基线、患者、block、特征平滑等提取参数，必须重新执行文件检查、提取及两轮分类。仅改变第一轮 K 时重跑第一轮，再复审及第二轮；仅改变复审条件或第二轮 K 时重跑第 7 节并在弹窗中修改。显示平滑参数只影响曲线，不影响已提取分类特征。`OPEN_REVIEW_WINDOW=False` 留给明确的批处理验证使用，不是日常默认流程。

## 基线和显示

- `BASELINE_MODE="file"`：使用 `I:\ECoG_data\患者\baseline\baseline.fif` 的合格区间，逐电极、逐频带计算均值和标准差。`BASELINE_SEGMENTS_CSV` 可指定区间表；边缘默认排除 0.3 秒。
- `BASELINE_MODE="pre_onset"`：每个 trial 使用 onset 前 `[-1.3, -0.3]` 秒，共 1 秒。只取到 onset 前 0.3 秒，不使用临近 onset 的最后 0.3 秒。短窗口对 Theta 功率标准差的估计不如长且干净的独立 baseline 稳定；零相位滤波/Hilbert 有时间扩散，0.3 秒间隔不保证基线完全不受事件影响，尤其是 Theta。因此默认仍使用独立 baseline 文件。
- 归一化为 `(功率 - 基线均值) / 基线标准差`。低方差或不可用基线标记为无效，而非用固定大 epsilon 压缩信号。任一频带原始/QC 不合格都会排除该电极。
- 特征提取平滑默认 0.06 秒，显示平滑 0.10 秒；显示统一约 100 Hz 网格。先平滑电极平均曲线，再算簇均值和电极间 SEM。
- 音频显示按所有保留 trial 的语音段全局峰值归一化，右轴独立。斜率图按语音进程百分比求导，不是每秒斜率；乘积先在匹配的同一 trial 内求值，再平均，避免跨 trial 混合。
- High-gamma 上限 150 Hz 要求采样率严格高于 300 Hz；不满足时明确报错。默认数据 400 Hz。必要时应额外检查工频谐波、肌电等伪迹，曲线好看并不保证可靠。

## 输出和验证

输出到 `outputs/患者/block/时间戳/`，每次分类保存独立轮次目录，不覆盖旧复审。包含基线质量报告、完整特征、选中特征权重、标签与排除原因、PCA 距离、复审记录、模型以及两轮曲线/音频关系 CSV 和 PNG。`summary.json` 的 silhouette 仅为描述性数值，**不进行自动选 K**。第 10 节另生成固定 K 的稳定性报告；标签和颜色不具有固定生理意义。

音频关系导出表使用明确前缀：`feature_mean_curve` 为特征提取时平均曲线的关系（音频保留提取单位），`feature_trialwise` 为提取时 trial 级描述，`display_paired_trial_slope_product_mean` 为显示设置下先在同一 trial 内计算斜率乘积、再平均的量（音频按全局峰值归一化），与音频动态图一致。这些量不应直接互换；方法和单位记录于 `relationship_methods.json`。

`DISPLAY_BACKEND="browser"` 为显示默认值。`RUN_3D_RENDER`、`RUN_SELECTABLE_CLUSTER_RENDER` 开启本地浏览器中的统一脑表面视图，使用第二轮联合标签；两项都开启也只使用一个脑视图。调整 `SURFACE_COLOR` / `SURFACE_OPACITY` 不改变分类。需要旧显示时显式设置 `DISPLAY_BACKEND="native"`，浏览器失败不会自动切换到旧窗口。

脑表面的 `Channel labels` 开关显示电极 CSV 中的实际通道名，不用行号代替通道号。选择类别时只显示当前可见电极的编号；切换编号不改变相机、颜色或分类。

## 真实连续双 trial

第 9 节从原始连续记录读取四频带 ECoG，显示两次相邻语音及中间真实间隔，保留音频包络。横轴为相对第一次 onset 的真实秒数，不压缩或拉伸语音持续时间，也不拼接语音进度数组。图横向加宽，原有平均图保留作补充。

- `TRIAL_PAIR_IDS=None`：默认第一对原始事件中真正相邻、且均保留的 trial。可改为配对表中的 `(原始编号1, 原始编号2)`；编号从 0 开始，不能把筛选后两行当作原始相邻事件。
- `TRIAL_PRE_SECONDS=1.0`、`TRIAL_POST_SECONDS=1.0`：第一次 onset 前及第二次 offset 后额外显示的秒数，可以增大。超出录制范围会截到边界并记录实际范围；窗口涉及的其他语音也标注。
- `TRIAL_DISPLAY_ROUND="both"`：显示两轮结果，可改为 `"first"` 或 `"second"`。第二轮未确认时不显示旧标签；新第一轮会取消旧第二轮的就绪状态。
- `DISPLAY_SMOOTH_SIGMA_SECONDS`、`DISPLAY_SFREQ` 控制连续显示的物理时间平滑与网格，不重新分类。
- file 基线沿用提取阶段的患者基线参数；pre_onset 模式的整幅图统一采用第一个所选 trial 的基线，并在标题/JSON 中明确标注。不会在第二次 onset 切换基线制造假跳变，分类仍沿用各 trial 自身基线。
- 阴影为该次连续记录中同类电极间 SEM，电极只有一个时不绘制 SEM，不是两个 trial 间的置信区间。

第 9 节可以单独重跑选择另一对 trial，每次在对应轮次目录下生成独立 `continuous_trials_时间戳` 文件夹，保存 PNG、曲线 CSV 和包含事件/窗口/基线/平滑/通道标签的 JSON。改变患者、block、频带或基线参数后，必须从提取和第一轮重新运行。

验证脚本只运行无弹窗副本，不修改原 Notebook：

```powershell
& 'E:\anaconda\envs\ECoG_LR_v1\python.exe' 3-bands/verify_notebook.py --baseline-mode file
& 'E:\anaconda\envs\ECoG_LR_v1\python.exe' 3-bands/verify_notebook.py --baseline-mode pre_onset --first-k 3 --second-k 2
```

可选 `--subject`、`--block`、`--manual-bad 电极名`、`--distance-threshold 数值`、`--weight-preset balanced`、`--exclude-trial ECoG_1:0`、`--stability-resamples 5`；运行记录存放于 `outputs/verification/`。验证副本关闭弹窗，默认不剔除候选伪迹；`--exclude-trial` 是验证时明确指定的剔除，不是自动检测剔除。现有进程显示版 Notebook 使用 `python 3-bands/create_notebook.py --update-browser` 迁移浏览器设置，仅补充缺少的 backend 参数，保留用户参数、代码编辑、所有输出、执行计数、单元格顺序和元数据，也不会恢复已删除的空白单元格。重复迁移不会重复添加设置，已显式选择的 native 后端也保留。无参数运行时，文件存在则迁移，只有文件不存在才生成初始版本；旧 `--update-review`、`--update-continuous`、`--update-reliability` 入口仍可使用，已有进程显示版走相同的源码迁移路径。

## Trial 质量人工复审

第 4.1 节打开通道 × 原始 trial 编号的质量复审窗口。所有候选项初始都不勾选，可以查看原始电压、手动勾选未标记项、选择全部候选、清空和撤销。只有点击 `Confirm exclusions` 才应用，取消后不能继续用旧结果重新分类。

检测包括尖峰、相邻采样跳变、平直段、重复极值的截幅候选、非有限采样及已有逐频带基线标记。截幅只称为候选，不假定已知放大器量程；阈值不是经过验证的生理拒绝标准，不增加自动陷波。

`TRIAL_QC_SETTINGS=None` 使用默认值，或在第 2.1 节填写覆盖字典，例如：

```python
TRIAL_QC_SETTINGS = {
    "spike_z": 10.0,
    "jump_z": 10.0,
    "flat_run_seconds": 0.05,
    "flat_tolerance_v": 1e-12,
    "clipping_fraction": 0.10,
    "clipping_run_seconds": 0.05,
    "clipping_tolerance_v": 1e-12,
    "interval": "speech_context",
}
```

尖峰参考该通道整段记录的中位数/缩放 MAD，跳变参考相邻差分的中位数/MAD；电压单位为伏。默认检测语音及已有前后窗口；可用 `pre_seconds` / `post_seconds` 覆盖。`interval="speech"` 只检测语音，前后边界须为零。

确认的通道-trial 对在四频带的 speech/pre/post 数组中一起掩蔽，并重新计算特征，不修改原始 FIF。基线无效比例、人工剔除比例和剩余有效 trial 数分别记录。原始坏道或原基线不合格通道不会被恢复；剩余有效 trial 少于 2 会明确标记为不合格电极。重跑复审从未剔除源数据开始，默认重新清空勾选，可恢复此前人工剔除。修改阈值后须重新复审并重跑两轮分类。

复审输出包括候选报告、确认清单、阈值/单位/来源指纹和复审后特征。每轮另保存 `fitted_electrode_features.csv`、`feature_group_budgets.csv` 及确认的 trial 剔除记录。

## 形态、幅度和斜率权重

第 2.1 节的 `WEIGHT_PRESET`：

| 设置 | 形态 : 幅度 : 斜率 |
| --- | --- |
| `legacy` | 保持原有权重，为迁移默认值 |
| `balanced` | 1 : 1 : 1 |
| `morphology` | 3 : 1 : 1 |
| `amplitude` | 1 : 3 : 1 |
| `custom` | 使用 `GROUP_WEIGHTS` |

三个组互斥，零权重组在冗余筛选之前移除；组内按保留特征数分配预算，频带总预算仍由 `BAND_WEIGHTS` 决定。某组无可用特征时，预算在该频带其余可用组内重新分配，实际预算和成员有 CSV 记录。每个频带必须至少有一组可用。

形态组包括早/中/晚均值及差值、正负面积、峰位置、峰宽和回落/反弹等；幅度组包括 energy、前后功率及峰值幅度；自身斜率归斜率组。形态特征仍可能依赖幅度，`morphology` 不等于完全去除幅度的波形聚类。修改预设、组权重、频带权重或第一轮 K 后重跑第一轮及第二轮。音频列仍不能进入模型。

## 稳定性与同类内部差异

第 10 节对每个已确认轮次的固定 K、固定电极集合评估：按时间分半、随机分半及 trial bootstrap。每次重新聚合特征、筛选、填补、标准化和拟合；同一次抽样在四频带/所有通道中一致，重复抽中的 trial 保留次数。默认 `STABILITY_RESAMPLES=30`，每轮 92 次比较拟合，另验证一次全数据参考；计算可能需要几分钟。进度窗可取消，已有分类保留，部分报告会标记取消。

`stability_summary.json`、`stability_replicates.csv`、`stability_channels.csv` 和 `stability.png` 保存 ARI、最大重叠匹配后的通道一致度、支持覆盖率及失败原因。样本支持不足不改变电极集合；小簇/塌缩结果计入已支持重复并单独标记，不隐藏不利结果。没有支持重复时指标不可用，但不会触发 `No eligible K` 或自动换 K。通道一致度不是生理分类正确的概率。

同目录的四张 `*_inspection.png` 包括最多 40 条电极平均细线、类别均值、电极热图和选定通道的 trial 热图；完整电极统计仍导出到 CSV。横轴明确为时长归一化语音进程，而非原始秒数；trial 编号不重排。`INSPECTION_BOOTSTRAP_DRAWS=100` 生成固定标签/电极条件下的逐点 trial 重采样 95% 置信带，设为 0 可关闭。电极 SEM、trial 置信带和归类稳定度不同，不作为显著性检验。

## 交互查看与缓存

第 8 节和第 11 节都把 `backend=DISPLAY_BACKEND` 传给 `fourband.display_process.run_display`。默认在系统浏览器打开本地会话，并在 Notebook 打印可重新打开的 URL。服务只监听 `127.0.0.1` 的随机空闲端口，URL 包含本次会话的随机访问令牌；Three.js、Plotly 和其他前端资源全部随项目本地提供，不使用 CDN，也不把患者数据上传到外部服务。

第 11 节 `OPEN_EXPLORER=True` 打开仅有曲线的浏览器视图（`explorer_only=True`）。相邻配对、上一对/下一对、类别、通道及 `Continuous` / `Cluster detail` 可在页面切换。第 8 节开启 `LINK_SURFACE_EXPLORER=True` 时，点击电极按实际通道名联动四频带曲线；关闭联动时只显示脑表面，不计算连续曲线。页面保留类别可见性、颜色、通道编号、脑表面外观和相机控制；秒单位 sigma 经 Apply 后应用，导出保存当前图和查看设置。

刷新页面可以恢复同一会话，不重新拟合、不丢弃已有分类标签。**显示 Cell 在会话活动期间保持阻塞**：点击页面的 **Close Session** 停止服务和它拥有的数值子进程，随后返回 Notebook；中断 Cell 也会清理本次拥有的进程树。**只关闭浏览器标签页不会清理会话**，需重新打开打印的 URL 后点击 Close Session，或在 Notebook 中断该 Cell。

Notebook 只传递可信本地提取/模型快照、源文件路径及显示设置，独立服务子进程拥有会话和数值子进程；源记录读取、SciPy 滤波、Hilbert 功率、平滑及有界缓存留在数值子进程。浏览器显示流程不创建 Tk、Qt 或 VTK 渲染窗口。Trial 质量复审与第二轮 PCA 复审仍使用原有流程；基线、特征、权重、手动 K、trial 剔除和已拟合标签不因更换显示后端而改变，音频仍仅供显示和描述性对照。

启动器沿用 Notebook 内核的 Python 环境。Windows 启动门控先建立 Job 对象的进程树归属，再放行服务代码；中断或失败时清理本次会话的服务/数值子进程。返回的 `surface_display_run` / `explorer_display_run` 仍以 `directory`、`pid`、`returncode` 记录会话。

对应轮次的 `display_sessions/时间戳/` 保留可信本地快照、`browser_session.json` 中的 URL/PID、`stdout.log`、`stderr.log`、`numerical_stdout.log`、`numerical_stderr.log` 及状态/退出记录；页面导出位于该轮次的 `browser_exports/`。服务或数值进程失败会在 Notebook 抛出普通 Python 异常并提供日志位置，已有分类仍在内核。缺失解剖、无效设置或 WebGL 不可用会显示错误；WebGL 失败时可用的 2D 曲线仍保留。浏览器失败不会自动重试 native 后端。`request.joblib` 只用于可信本地数据，快照可能占用数百 MB，不会自动清除。

原始连续曲线不会因分类剔除而删掉真实记录：被剔除的选定通道 trial 标成 QC，类别平均连续图标注该事件中剔除的成员数。它们是原始信号对照，不应误读为重新纳入分类。

首次连续功率计算仍可能等待约一分钟，取决于记录长度和电脑性能；更换浏览器后端不承诺加快数值计算。页面显示实际加载/计算阶段及已等待秒数，完成后显示真实曲线。相同设置共用正在执行的计算；仅切换通道、类别、颜色或编号不重新滤波，更改配对或 sigma 会请求新计算，过时结果不会覆盖当前视图。

`CACHE_MAX_MB=512` 分别设置第 9 节内核静态图缓存和会话数值进程中的有界功率缓存。每个新会话从冷缓存开始，结束后释放；刷新页面复用仍活动的会话。`POWER_CACHE.clear()` / `POWER_CACHE.info()` 只影响第 9 节，页面的 Clear cache 清理当前会话的功率和最近结果缓存。缓存之后按现有配置应用基线；信号内容、采样率、通道次序、频带或滤波参数改变会使缓存失效，缓存不足时报告并退回逐频带计算。

## 显式 Native 回退与排查

确需旧 PyVista/Tk 显示时，在参数区显式设置 `DISPLAY_BACKEND="native"` 后重跑显示 Cell；所需 PyVista、Tk 等依赖须在内核环境中可用。旧后端同样通过独立进程启动器运行，其 GIL/GUI 兼容性限制仍可能存在。此前真实独立 GUI 测试复现过 `PyEval_RestoreThread` 崩溃，不能把进程隔离理解为保证旧显示不会失败。

若原有复审或显式 native 窗口日志出现 `invalid command name "...idle_draw"`，表示已关闭的 Tk 画布仍有待执行回调。复审和查看窗口会在销毁前取消 Matplotlib 绘图/事件循环回调，并先解绑 GUI 画布；两轮静态 PNG 导出独立使用 Agg 画布。

已有当前模型且内核尚未重启时，先在一个临时 Cell 热更新启动器，再重跑显示 Cell，无需重新提取或聚类；仅改变后端变量不足以更新旧内核缓存的函数签名。

```python
import importlib
import fourband.display_process as display_module
run_display = importlib.reload(display_module).run_display
DISPLAY_BACKEND = "browser"
```

若重启内核则需重新按初始化、参数、提取和人工复审顺序运行。现有 Notebook 仅修改显示相关源码，保留参数、已有输出和用户编辑，不整体重新生成。其他内核崩溃仍需根据 Jupyter 日志判断原因。

`scripts/verify_isolated_fourband_display.py --request <可信本地 request.joblib 路径>` 属于旧 native 显示诊断，检查实际 Tk 画布像素；`--explorer-only` 可单独验证旧曲线窗口。这些结果不能代替浏览器流程验证。

复现实际参数和既有人工选择可使用 `scripts/reproduce_fourband_kernel.py`，结果另存于 `outputs/verification/kernel_reproduction/`；不改写用户 Notebook。`--inprocess` 在指定环境中按单元顺序执行，保留真实 Tk/VTK 窗口，但并非 Jupyter 内核测试。诊断工具不启用周期性底层栈转储：该环境中它本身已触发访问冲突，不能把此类诊断崩溃误判为绘图故障。
