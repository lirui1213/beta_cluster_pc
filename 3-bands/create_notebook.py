"""Build the four-band experiment notebook using structured notebook cells."""
from pathlib import Path
import ast
import nbformat as nbf


def build_notebook():
    m, c = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        m("# 四频带联合 ECoG 分类\n\nTheta、Beta1、Beta2、High-gamma 共同产生一套电极标签。分类只使用 ECoG 自身特征，音频仅作显示和描述性对照。两轮 K 手动指定；第一轮 PCA 复审后独立重拟合第二轮。第 9 节显示两个相邻 trial 及中间间隔的真实连续曲线。"),
        m("## 1. 初始化"),
        c('''from pathlib import Path
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

os.environ.setdefault("OMP_NUM_THREADS", "1")
PROJECT_DIR = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / "audio_cross_3band_utils.py").is_file()), None)
if PROJECT_DIR is None:
    raise FileNotFoundError("Open this notebook from the project root or its 3-bands directory.")
FOURBAND_DIR = PROJECT_DIR / "3-bands"
for folder in (PROJECT_DIR, FOURBAND_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
for name in list(sys.modules):
    if name == "fourband" or name.startswith("fourband."):
        sys.modules.pop(name, None)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import visualization
importlib.reload(visualization)
from IPython.display import display, Image
from fourband.analysis import DEFAULT_BANDS, load_fourband, fit_joint
from fourband.review import pca_review_table, refit_reviewed
from fourband.review_window import FourBandReviewWindow, ReviewSettings
from fourband.workflow import save_extraction, save_round, ensure_extraction_config, ensure_first_round
from fourband.continuous import adjacent_trial_pairs, load_continuous_trials, save_continuous_trial_plot
from audio_cross_3band_utils import load_speech_events
from visualization import render_electrode_clusters, render_selectable_electrode_class
SECOND_ROUND_READY = False
pd.set_option("display.max_columns", 24)
def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
print(PROJECT_DIR)'''),
        m("## 2. 参数区\n\n两轮 K 均由下方参数手动决定，不执行自动选 K。频带总权重默认为等权；基线与显示窗口相互独立。"),
        c('''DATA_ROOT = Path(r"I:\\ECoG_data")
SUBJECT_ID = "HS0023"
BLOCK_ID = "B07"
DATA_DIR = DATA_ROOT / SUBJECT_ID / "blocks" / BLOCK_ID / "standardized"
FIF_PATH = DATA_DIR / "ECoG" / "ecog_TEST_clean_car-raw.fif"
ONSET_PATH = DATA_DIR / "events" / "onset.npy"
OFFSET_PATH = DATA_DIR / "events" / "offset.npy"
WAV_PATH = DATA_DIR / "audio" / "analog1.wav"
BASELINE_FIF = DATA_ROOT / SUBJECT_ID / "baseline" / "baseline.fif"

BANDS = {"theta": (4.0, 8.0), "beta1": (13.0, 24.0),
         "beta2": (24.0, 40.0), "high_gamma": (70.0, 150.0)}
BAND_WEIGHTS = {name: 1.0 for name in BANDS}
FIRST_N_CLUSTERS = 10
SECOND_N_CLUSTERS = 4
RANDOM_STATE = 50
FEATURE_CORRELATION_THRESHOLD = .95

BASELINE_MODE = "file"  # "file" or "pre_onset"
BASELINE_WINDOW_SECONDS = (-1.3, -.3)
BASELINE_SEGMENTS_CSV = None
BASELINE_EDGE_SECONDS = .3
BASELINE_RELATIVE_FLOOR = 1e-3
MAX_BASELINE_INVALID_FRACTION = .25
PROGRESS_POINTS = 101
DURATION_MAD_THRESHOLD = 2.5
MIN_TRIALS_AFTER_FILTER = 3
PRE_POST_WINDOW_SECONDS = None
FEATURE_SMOOTH_SIGMA_SECONDS = .06
FEATURE_SLOPE_METHOD = "savgol"
DISPLAY_SMOOTH_SIGMA_SECONDS = .10
DISPLAY_SFREQ = 100.0

OUTPUT_ROOT = FOURBAND_DIR / "outputs" / SUBJECT_ID / BLOCK_ID
RUN_3D_RENDER = False
RUN_SELECTABLE_CLUSTER_RENDER = False
RECON_DIR = DATA_ROOT / SUBJECT_ID / "recon"
ELECTRODE_CSV = RECON_DIR / "electrodes" / "electrodes.csv"
SUBJECTS_DIR = DATA_ROOT / SUBJECT_ID
SURFACE_SUBJECT = "recon"
REFERENCE_MRI = RECON_DIR / "mri" / "T1.mgz"
HEMISPHERES = ("lh",)
SURFACE_COLOR = "lightgray"
SURFACE_OPACITY = .25
def extraction_settings():
    return dict(subject=SUBJECT_ID, block=BLOCK_ID, bands={name: tuple(bounds) for name, bounds in BANDS.items()},
        baseline_mode=BASELINE_MODE, baseline_fif=str(BASELINE_FIF), baseline_window=tuple(BASELINE_WINDOW_SECONDS),
        baseline_segments_csv=str(BASELINE_SEGMENTS_CSV) if BASELINE_SEGMENTS_CSV is not None else None,
        baseline_edge_seconds=BASELINE_EDGE_SECONDS, baseline_relative_floor=BASELINE_RELATIVE_FLOOR,
        max_invalid_fraction=MAX_BASELINE_INVALID_FRACTION, progress_points=PROGRESS_POINTS,
        duration_mad_threshold=DURATION_MAD_THRESHOLD, min_trials=MIN_TRIALS_AFTER_FILTER,
        pre_post_window=PRE_POST_WINDOW_SECONDS, feature_sigma_seconds=FEATURE_SMOOTH_SIGMA_SECONDS,
        slope_method=FEATURE_SLOPE_METHOD,
        fif_path=str(FIF_PATH), onset_path=str(ONSET_PATH), offset_path=str(OFFSET_PATH), wav_path=str(WAV_PATH))
print(SUBJECT_ID, BLOCK_ID, BANDS)'''),
        m("## 3. 文件检查"),
        c('''required = {"FIF": FIF_PATH, "onset": ONSET_PATH, "offset": OFFSET_PATH, "audio": WAV_PATH}
if BASELINE_MODE not in {"file", "pre_onset"}:
    raise ValueError("BASELINE_MODE must be file or pre_onset.")
if BASELINE_MODE == "file":
    required["baseline"] = BASELINE_FIF
    if BASELINE_SEGMENTS_CSV is not None:
        required["baseline_segments"] = Path(BASELINE_SEGMENTS_CSV)
missing = {name: str(path) for name, path in required.items() if not Path(path).is_file()}
if missing:
    raise FileNotFoundError(missing)
display(pd.DataFrame({"input": list(required), "path": list(map(str, required.values()))}))'''),
        m("## 4. 四频带提取与基线检查\n\n各频带分别做实际带通、Hilbert 功率、基线归一化及特征提取。任一频带质量不合格的电极不参与联合分类。频带边界不会因 Nyquist 限制而静默截短。"),
        c('''pending_settings = extraction_settings()
pending_extraction = load_fourband(
    FIF_PATH, ONSET_PATH, OFFSET_PATH, WAV_PATH, bands=BANDS, picks="ecog",
    progress_points=PROGRESS_POINTS, duration_mad_threshold=DURATION_MAD_THRESHOLD,
    min_trials_after_filter=MIN_TRIALS_AFTER_FILTER, pre_post_window=PRE_POST_WINDOW_SECONDS,
    baseline_mode=BASELINE_MODE, baseline_window=BASELINE_WINDOW_SECONDS,
    baseline_fif=BASELINE_FIF if BASELINE_MODE == "file" else None,
    baseline_segments_csv=BASELINE_SEGMENTS_CSV, baseline_edge_seconds=BASELINE_EDGE_SECONDS,
    baseline_relative_floor=BASELINE_RELATIVE_FLOOR,
    smoothing_sigma_seconds=FEATURE_SMOOTH_SIGMA_SECONDS, slope_method=FEATURE_SLOPE_METHOD,
    max_baseline_invalid_fraction=MAX_BASELINE_INVALID_FRACTION,
)
pending_run_dir = OUTPUT_ROOT / timestamp()
config = dict(pending_settings, band_weights=dict(BAND_WEIGHTS), display_sigma_seconds=DISPLAY_SMOOTH_SIGMA_SECONDS,
    display_sfreq=DISPLAY_SFREQ, correlation_threshold=FEATURE_CORRELATION_THRESHOLD,
    first_k=FIRST_N_CLUSTERS, second_k=SECOND_N_CLUSTERS, random_state=RANDOM_STATE)
save_extraction(pending_extraction, pending_run_dir, config)
extraction, extracted_settings, RUN_DIR = pending_extraction, pending_settings, pending_run_dir
features = extraction.features
report = []
for band, epochs in extraction.epochs.items():
    reference = epochs.baseline_reference
    valid = reference.valid
    report.append(dict(band=band, trials=len(epochs.events), raw_bad_channels=len(epochs.bad_channels),
        invalid_pairs=int((~epochs.baseline_diagnostics.valid).sum()),
        baseline_mean_error=float(reference.loc[valid, "normalized_mean"].abs().max()),
        baseline_std_error=float((reference.loc[valid, "normalized_std"] - 1).abs().max())))
display(pd.DataFrame(report))
display(features[["channel", "is_bad", "exclusion_reason"]])
print(f"Joint eligible electrodes: {int((~features.is_bad).sum())}/{len(features)}")
print(RUN_DIR)'''),
        m("## 5. 第一轮 ECoG 联合分类\n\nPCA 用于显示和复审；K-means 只使用四频带 ECoG 的形态、功率和斜率特征，音频耦合特征不参与分类。各频带总权重不随保留特征数增长。下图为多 trial 时长归一化平均；第 9 节为真实连续双 trial 曲线。"),
        c('''SECOND_ROUND_READY = False
ensure_extraction_config(extraction_settings(), extracted_settings)
pending_first = fit_joint(features, n_clusters=FIRST_N_CLUSTERS, bands=BANDS,
    band_weights=BAND_WEIGHTS, correlation_threshold=FEATURE_CORRELATION_THRESHOLD, random_state=RANDOM_STATE)
pending_round1_dir = RUN_DIR / ("round1_" + timestamp())
first_assignments = save_round(extraction, pending_first, pending_round1_dir, n_clusters=FIRST_N_CLUSTERS,
    sigma_seconds=DISPLAY_SMOOTH_SIGMA_SECONDS, display_sfreq=DISPLAY_SFREQ, slope_method=FEATURE_SLOPE_METHOD,
    round_config=dict(round=1, predictor_policy="ecog_only", band_weights=BAND_WEIGHTS, random_state=RANDOM_STATE,
        correlation_threshold=FEATURE_CORRELATION_THRESHOLD))
first_result, first_extraction, ROUND1_DIR = pending_first, extraction, pending_round1_dir
SECOND_ROUND_READY = False
display(first_assignments.groupby("label").size().rename("n_electrodes"))
display(Image(filename=str(ROUND1_DIR / "pca_clusters.png")))
display(Image(filename=str(ROUND1_DIR / "fourband_cluster_grid.png")))
for band in BANDS:
    display(Image(filename=str(ROUND1_DIR / f"{band}_audio_dynamics.png")))'''),
        m("## 6. PCA 复审准备\n\n下一节默认打开复审弹窗，设置距离阈值、参考方式、整类/指定电极剔除及第二轮 K。全局距离为前三个 PC 到所有合格电极坐标中位数的欧氏距离；簇内距离到所属簇中心。孤立电极的簇内距离可能为零，阈值没有通用数值。原始坏道不会恢复。"),
        c('''SECOND_ROUND_READY = False
ensure_first_round(globals().get("first_extraction"), extraction)
OPEN_REVIEW_WINDOW = True
if globals().get("review_model") is not first_result:
    review_settings = ReviewSettings(n_clusters=SECOND_N_CLUSTERS)
    review_model = first_result

pca_distances = pca_review_table(first_result)
display(pca_distances.sort_values("pca_global_distance", ascending=False).head(25))'''),
        m("## 7. 弹窗复审与第二轮联合分类\n\n在窗口中设置第二轮 K 和剔除条件，点击确认后才重新分类。取消或关闭窗口不执行第二轮，也不显示旧的脑表面结果。重跑本节会再次打开窗口，并保留同一第一轮模型下最后确认的设置。若改变提取参数，请重新提取并执行第一轮。类别数仍由你决定，不代表通过稳定性检验。"),
        c('''SECOND_ROUND_READY = False
ensure_extraction_config(extraction_settings(), extracted_settings)
ensure_first_round(globals().get("first_extraction"), extraction)
if globals().get("review_model") is not first_result:
    review_settings = ReviewSettings(n_clusters=SECOND_N_CLUSTERS)
    review_model = first_result
if OPEN_REVIEW_WINDOW:
    selected_review = FourBandReviewWindow(first_result, initial=review_settings).show()
else:
    selected_review = review_settings
if selected_review is None:
    print("已取消复审，未执行第二轮聚类。")
else:
    review_settings = selected_review
    selected_bad_clusters = selected_review.bad_clusters
    selected_manual_channels = selected_review.manual_bad_channels
    selected_second_k = selected_review.n_clusters
    pending_second, pending_audit = refit_reviewed(features, first_result, **selected_review.refit_kwargs(),
        bands=BANDS, band_weights=BAND_WEIGHTS, correlation_threshold=FEATURE_CORRELATION_THRESHOLD, random_state=RANDOM_STATE)
    pending_round2_dir = RUN_DIR / ("round2_" + timestamp())
    pending_assignments = save_round(extraction, pending_second, pending_round2_dir, n_clusters=selected_second_k,
        audit=pending_audit, sigma_seconds=DISPLAY_SMOOTH_SIGMA_SECONDS, display_sfreq=DISPLAY_SFREQ,
        slope_method=FEATURE_SLOPE_METHOD, round_config=dict(round=2, first_round=str(ROUND1_DIR),
        bad_clusters=selected_bad_clusters, manual_bad_channels=selected_manual_channels,
        pca_distance_threshold=selected_review.pca_distance_threshold, distance_reference=selected_review.distance_reference,
        predictor_policy="ecog_only", band_weights=BAND_WEIGHTS, random_state=RANDOM_STATE, correlation_threshold=FEATURE_CORRELATION_THRESHOLD))
    second_result, review_audit, ROUND2_DIR, second_assignments = pending_second, pending_audit, pending_round2_dir, pending_assignments
    SECOND_ROUND_READY = True
    display(review_audit.loc[review_audit.excluded_final])
    display(second_assignments.groupby("label").size().rename("n_electrodes"))
    display(Image(filename=str(ROUND2_DIR / "pca_clusters.png")))
    display(Image(filename=str(ROUND2_DIR / "fourband_cluster_grid.png")))
    for band in BANDS:
        display(Image(filename=str(ROUND2_DIR / f"{band}_audio_dynamics.png")))
    print(ROUND2_DIR)'''),
        m("## 8. 可选脑表面显示\n\n使用第二轮联合标签，绘图参数不影响分类。窗口的 Channel labels 开关显示或隐藏实际通道名，选择类别时编号随可见电极切换。"),
        c('''if RUN_3D_RENDER and SECOND_ROUND_READY:
    render_electrode_clusters(ELECTRODE_CSV, dict(zip(second_result.channels, map(int, second_result.labels))),
        subjects_dir=SUBJECTS_DIR, subject=SURFACE_SUBJECT, reference_mri=REFERENCE_MRI,
        hemispheres=HEMISPHERES, brain_color=SURFACE_COLOR, brain_opacity=SURFACE_OPACITY,
        channel_label_toggle=True, screenshot=ROUND2_DIR / "joint_electrode_clusters.png")
if RUN_SELECTABLE_CLUSTER_RENDER and SECOND_ROUND_READY:
    render_selectable_electrode_class(ELECTRODE_CSV, dict(zip(second_result.channels, map(int, second_result.labels))),
        subjects_dir=SUBJECTS_DIR, subject=SURFACE_SUBJECT, reference_mri=REFERENCE_MRI,
        hemispheres=HEMISPHERES, brain_color=SURFACE_COLOR, brain_opacity=SURFACE_OPACITY,
        channel_label_toggle=True)'''),
        m("## 9. 相邻双 trial 连续显示\n\n横轴为相对第一次 onset 的真实秒数，完整保留两个 trial 及中间间隔。原始 trial 编号从 0 开始，不是筛选后的行号；从下方相邻配对表选取。音频包络保留，分类不使用音频。pre_onset 模式的整幅连续图统一使用第一个 trial 的基线；SEM 表示电极间差异，不是两个 trial 的置信区间。重跑本节不会重新分类。"),
        c('''TRIAL_PAIR_IDS = None  # None: first available pair; or (original_id_1, original_id_2)
TRIAL_PRE_SECONDS = 1.0
TRIAL_POST_SECONDS = 1.0
TRIAL_DISPLAY_ROUND = "both"  # "first", "second", "both"

ensure_extraction_config(extraction_settings(), extracted_settings)
ensure_first_round(globals().get("first_extraction"), extraction)
if TRIAL_DISPLAY_ROUND not in {"first", "second", "both"}:
    raise ValueError("TRIAL_DISPLAY_ROUND must be first, second or both.")
trial_pairs = adjacent_trial_pairs(load_speech_events(ONSET_PATH, OFFSET_PATH), next(iter(extraction.epochs.values())).events)
display(trial_pairs)
display_rounds = []
if TRIAL_DISPLAY_ROUND in {"first", "both"}:
    display_rounds.append(("first", first_result, ROUND1_DIR))
if TRIAL_DISPLAY_ROUND in {"second", "both"} and SECOND_ROUND_READY:
    display_rounds.append(("second", second_result, ROUND2_DIR))
if TRIAL_DISPLAY_ROUND == "second" and not SECOND_ROUND_READY:
    raise ValueError("Confirm a fresh second-round review before displaying its trials.")
if TRIAL_DISPLAY_ROUND == "both" and not SECOND_ROUND_READY:
    print("第二轮未确认，仅显示第一轮连续曲线。")
continuous_trials = load_continuous_trials(FIF_PATH, ONSET_PATH, OFFSET_PATH, WAV_PATH, extraction,
    trial_ids=TRIAL_PAIR_IDS, pre_seconds=TRIAL_PRE_SECONDS, post_seconds=TRIAL_POST_SECONDS,
    sigma_seconds=DISPLAY_SMOOTH_SIGMA_SECONDS, display_sfreq=DISPLAY_SFREQ)
continuous_display_outputs = {}
for round_name, fitted_result, round_dir in display_rounds:
    display_dir = round_dir / ("continuous_trials_" + timestamp())
    save_continuous_trial_plot(continuous_trials, fitted_result, display_dir)
    continuous_display_outputs[round_name] = display_dir
    print(round_name, continuous_trials.trial_ids, continuous_trials.metadata["actual_bounds_seconds"], display_dir)
    display(Image(filename=str(display_dir / "fourband_continuous_trials.png")))'''),
    ]
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python"}
    return notebook


def update_review_cells(output):
    """Update the approved UI cells without replacing user analysis parameters."""
    existing = nbf.read(output, as_version=4)
    revised = build_notebook()
    for index in (11, 12, 13, 14):
        if existing.cells[index].cell_type != revised.cells[index].cell_type:
            raise ValueError("Notebook layout changed; review cells require a targeted manual update.")
        existing.cells[index].source = revised.cells[index].source
        if existing.cells[index].cell_type == "code":
            existing.cells[index].outputs = []
            existing.cells[index].execution_count = None
    initialization = existing.cells[2]
    initialization.source = initialization.source.replace(
        "from visualization import BadChannelReviewWindow, render_electrode_clusters, render_selectable_electrode_class",
        "from visualization import render_electrode_clusters, render_selectable_electrode_class")
    import_line = "from fourband.review_window import FourBandReviewWindow, ReviewSettings"
    if import_line not in initialization.source:
        initialization.source = initialization.source.replace(
            "from fourband.review import pca_review_table, refit_reviewed",
            "from fourband.review import pca_review_table, refit_reviewed\n" + import_line)
    existing.cells[16].source = existing.cells[16].source.replace(
        "if RUN_3D_RENDER:", "if RUN_3D_RENDER and SECOND_ROUND_READY:").replace(
        "if RUN_SELECTABLE_CLUSTER_RENDER:", "if RUN_SELECTABLE_CLUSTER_RENDER and SECOND_ROUND_READY:")
    nbf.validate(existing)
    nbf.write(existing, output)


def _owned_cell(notebook, cell_type, *markers):
    matches = [i for i, cell in enumerate(notebook.cells)
        if cell.cell_type == cell_type and all(marker in cell.source for marker in markers)]
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely identify owned notebook cell: {markers}. No file was changed.")
    return matches[0]


def _preserve_display_parameters(source, revised):
    names = {"TRIAL_PAIR_IDS", "TRIAL_PRE_SECONDS", "TRIAL_POST_SECONDS", "TRIAL_DISPLAY_ROUND"}
    originals = {}
    lines = source.splitlines()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                originals[node.targets[0].id] = "\n".join(lines[node.lineno - 1:node.end_lineno])
    updated = revised.splitlines()
    for node in reversed(ast.parse(revised).body):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in originals:
                updated[node.lineno - 1:node.end_lineno] = [originals[name]]
    return "\n".join(updated)


def update_continuous_cells(output):
    """Migrate display and policy cells without touching user parameters."""
    existing = nbf.read(output, as_version=4)
    revised = build_notebook()
    _owned_cell(existing, "code", "SUBJECT_ID =", "def extraction_settings():")
    owners = {
        0: ("markdown", "# 四频带联合 ECoG 分类"),
        2: ("code", "PROJECT_DIR =", "from fourband.analysis import"),
        9: ("markdown", "## 5."),
        10: ("code", "pending_first = fit_joint("),
        14: ("code", "selected_review = FourBandReviewWindow("),
        15: ("markdown", "## 8."),
        16: ("code", "if RUN_3D_RENDER", "render_electrode_clusters("),
    }
    # Identify source-owned cells before replacing anything; positional type
    # checks cannot distinguish an inserted user analysis from pipeline code.
    positions = {index: _owned_cell(existing, *markers) for index, markers in owners.items()}
    continuous = [i for i, cell in enumerate(existing.cells) if cell.cell_type == "code"
        and "TRIAL_PAIR_IDS =" in cell.source and "load_continuous_trials(" in cell.source]
    if continuous:
        code_index = _owned_cell(existing, "code", "TRIAL_PAIR_IDS =", "load_continuous_trials(")
        markdown_index = _owned_cell(existing, "markdown", "## 9.", "双 trial")
        display_source = _preserve_display_parameters(existing.cells[code_index].source, revised.cells[18].source)
        existing.cells[markdown_index].source = revised.cells[17].source
        existing.cells[code_index].source = display_source
    else:
        insert_at = positions[16] + 1
        existing.cells[insert_at:insert_at] = revised.cells[17:]
    for revised_index, existing_index in positions.items():
        existing.cells[existing_index].source = revised.cells[revised_index].source
    for cell in existing.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbf.validate(existing)
    nbf.write(existing, output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-review", action="store_true")
    parser.add_argument("--update-continuous", action="store_true")
    args = parser.parse_args()
    output = Path(__file__).with_name("ECoG_4band_clustering_pipeline.ipynb")
    if args.update_continuous:
        update_continuous_cells(output)
    elif args.update_review:
        update_review_cells(output)
    else:
        nbf.write(build_notebook(), output)
    print(output)
