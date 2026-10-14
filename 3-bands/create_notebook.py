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
from fourband.display_process import run_display
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
DISPLAY_BACKEND = "browser"  # "native": explicit legacy fallback

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
        c('''if RUN_3D_RENDER or RUN_SELECTABLE_CLUSTER_RENDER:
    if SECOND_ROUND_READY:
        surface_display_run = run_display(extraction, second_result, output_dir=ROUND2_DIR,
            fif_path=FIF_PATH, onset_path=ONSET_PATH, offset_path=OFFSET_PATH, wav_path=WAV_PATH,
            surface_options=dict(electrode_csv=ELECTRODE_CSV, subjects_dir=SUBJECTS_DIR,
                subject=SURFACE_SUBJECT, reference_mri=REFERENCE_MRI, hemispheres=HEMISPHERES,
                brain_color=SURFACE_COLOR, brain_opacity=SURFACE_OPACITY, channel_label_toggle=True),
            show_surface=RUN_3D_RENDER, show_selectable=RUN_SELECTABLE_CLUSTER_RENDER,
            link_explorer=globals().get("LINK_SURFACE_EXPLORER", True),
            explorer_settings=dict(pre_seconds=globals().get("TRIAL_PRE_SECONDS", 1.0),
                post_seconds=globals().get("TRIAL_POST_SECONDS", 1.0),
                sigma_seconds=DISPLAY_SMOOTH_SIGMA_SECONDS, display_sfreq=DISPLAY_SFREQ,
                trial_ids=globals().get("TRIAL_PAIR_IDS", None)),
            cache_max_mb=globals().get("CACHE_MAX_MB", 512), backend=DISPLAY_BACKEND)'''),
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
    if any("WEIGHT_PRESET =" in cell.source for cell in existing.cells):
        return update_reliability_cells(output)
    if _has_process_displays(existing):
        return update_browser_cells(output)
    revised = build_notebook()
    owners = {
        2: ("code", "PROJECT_DIR =", "from fourband.analysis import"),
        11: ("markdown", "## 6.", "PCA"),
        12: ("code", "pca_review_table("),
        13: ("markdown", "## 7."),
        14: ("code", "refit_reviewed("),
        16: ("code", "if RUN_3D_RENDER"),
    }
    # Resolve every owner before editing sources, including the surface cell.
    positions = {index: _owned_cell(existing, *markers) for index, markers in owners.items()}
    for index in (11, 12, 13, 14):
        cell = existing.cells[positions[index]]
        cell.source = revised.cells[index].source
    initialization = existing.cells[positions[2]]
    initialization.source = initialization.source.replace(
        "from visualization import BadChannelReviewWindow, render_electrode_clusters, render_selectable_electrode_class",
        "from fourband.display_process import run_display").replace(
        "from visualization import render_electrode_clusters, render_selectable_electrode_class",
        "from fourband.display_process import run_display")
    if "from fourband.display_process import run_display" not in initialization.source:
        initialization.source += "\nfrom fourband.display_process import run_display"
    import_line = "from fourband.review_window import FourBandReviewWindow, ReviewSettings"
    if import_line not in initialization.source:
        initialization.source = initialization.source.replace(
            "from fourband.review import pca_review_table, refit_reviewed",
            "from fourband.review import pca_review_table, refit_reviewed\n" + import_line)
    existing.cells[positions[16]].source = revised.cells[16].source
    _update_browser_sources(existing)
    nbf.validate(existing)
    nbf.write(existing, output)


def _owned_cell(notebook, cell_type, *markers):
    matches = [i for i, cell in enumerate(notebook.cells)
        if cell.cell_type == cell_type and all(marker in cell.source for marker in markers)]
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely identify owned notebook cell: {markers}. No file was changed.")
    return matches[0]


def _has_process_displays(notebook):
    markers = ["if RUN_3D_RENDER"]
    if any("WEIGHT_PRESET =" in cell.source for cell in notebook.cells):
        markers.append("if OPEN_EXPLORER:")
    for marker in markers:
        matches = [cell for cell in notebook.cells if cell.cell_type == "code" and marker in cell.source]
        if len(matches) != 1 or not any(isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "run_display"
                for node in ast.walk(ast.parse(matches[0].source))):
            return False
    return True


def _update_browser_sources(notebook):
    """Patch only missing settings/keywords; retain every existing cell payload."""
    parameters = notebook.cells[_owned_cell(notebook, "code", "SUBJECT_ID =", "def extraction_settings():")]
    displays = [notebook.cells[_owned_cell(notebook, "code", "if RUN_3D_RENDER")]]
    if any("WEIGHT_PRESET =" in cell.source for cell in notebook.cells):
        displays.append(notebook.cells[_owned_cell(notebook, "code", "if OPEN_EXPLORER:")])
    sources = []
    for cell in displays:
        source = cell.source
        calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "run_display"]
        if len(calls) != 1:
            raise ValueError("Expected one process launcher per display cell. No file was changed.")
        call = calls[0]
        if not any(keyword.arg == "backend" for keyword in call.keywords):
            lines = source.splitlines(keepends=True)
            # AST columns count UTF-8 bytes, including any user text on this line.
            column = len(lines[call.end_lineno - 1].encode("utf-8")[:call.end_col_offset].decode("utf-8"))
            offset = sum(map(len, lines[:call.end_lineno - 1])) + column - 1
            separator = " " if source[:offset].rstrip().endswith(",") else ", "
            source = source[:offset] + separator + "backend=DISPLAY_BACKEND" + source[offset:]
        sources.append(source)
    has_backend = any(isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(isinstance(target, ast.Name) and target.id == "DISPLAY_BACKEND"
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))
        for cell in notebook.cells if cell.cell_type == "code" for node in ast.parse(cell.source).body)
    if not has_backend:
        parameters.source += '\nDISPLAY_BACKEND = "browser"'
    for cell, source in zip(displays, sources):
        cell.source = source


def update_browser_cells(output):
    """Migrate existing process display cells without regenerating the notebook."""
    existing = nbf.read(output, as_version=4)
    _update_browser_sources(existing)
    nbf.validate(existing)
    nbf.write(existing, output)


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
    if any("WEIGHT_PRESET =" in cell.source for cell in existing.cells):
        return update_reliability_cells(output)
    if _has_process_displays(existing):
        return update_browser_cells(output)
    revised = build_notebook()
    _owned_cell(existing, "code", "SUBJECT_ID =", "def extraction_settings():")
    owners = {
        0: ("markdown", "# 四频带联合 ECoG 分类"),
        2: ("code", "PROJECT_DIR =", "from fourband.analysis import"),
        9: ("markdown", "## 5."),
        10: ("code", "pending_first = fit_joint("),
        14: ("code", "selected_review = FourBandReviewWindow("),
        15: ("markdown", "## 8."),
        16: ("code", "if RUN_3D_RENDER"),
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
    _update_browser_sources(existing)
    nbf.validate(existing)
    nbf.write(existing, output)


def build_reliability_notebook():
    """Extend the legacy template without altering its analysis parameter cell."""
    notebook = build_notebook()
    initialization = notebook.cells[2]
    initialization.source += '''
from fourband.trial_qc import detect_trial_quality, apply_trial_exclusions, TrialExclusions
from fourband.trial_review_window import TrialQualityReviewWindow
from fourband.continuous import ContinuousPowerCache
from fourband.inspection import save_inspection
from fourband.stability import save_stability
from fourband.session import ensure_fit_settings, ensure_trial_review, close_explorers, run_stability
from audio_cross_3band_utils import load_ecog_fif
TRIAL_QC_READY = False
EXPLORERS = {}
'''
    settings = nbf.v4.new_code_cell('''WEIGHT_PRESET = "legacy"  # legacy, balanced, morphology, amplitude, custom
GROUP_WEIGHTS = {"morphology": 1.0, "amplitude": 1.0, "slope": 1.0}
OPEN_TRIAL_QC_WINDOW = True
TRIAL_QC_SETTINGS = None  # None: candidate detection defaults; see README
RUN_STABILITY = True
STABILITY_RESAMPLES = 30
STABILITY_PROGRESS_WINDOW = True
INSPECTION_BOOTSTRAP_DRAWS = 100
OPEN_EXPLORER = True
LINK_SURFACE_EXPLORER = True
CACHE_MAX_MB = 512
POWER_CACHE = ContinuousPowerCache(max_bytes=int(CACHE_MAX_MB * 1024**2))

def fit_settings():
    return dict(bands=dict(BANDS), band_weights=dict(BAND_WEIGHTS),
        correlation_threshold=FEATURE_CORRELATION_THRESHOLD, random_state=RANDOM_STATE,
        weight_preset=WEIGHT_PRESET, group_weights=dict(GROUP_WEIGHTS) if WEIGHT_PRESET == "custom" else None)''')
    extraction_cell = notebook.cells[8]
    extraction_cell.source = "TRIAL_QC_READY = False\nSECOND_ROUND_READY = False\nclose_explorers(EXPLORERS)\n" + extraction_cell.source
    extraction_cell.source += "\nUNREVIEWED_EXTRACTION = extraction\n"
    qc = nbf.v4.new_code_cell('''TRIAL_QC_READY = False
SECOND_ROUND_READY = False
close_explorers(EXPLORERS)
ensure_extraction_config(extraction_settings(), extracted_settings)
qc_raw = load_ecog_fif(FIF_PATH, preload=False)
trial_quality_report = detect_trial_quality(qc_raw, UNREVIEWED_EXTRACTION, settings=TRIAL_QC_SETTINGS)
display(trial_quality_report.loc[trial_quality_report.candidate])
if OPEN_TRIAL_QC_WINDOW:
    selected_trial_pairs = TrialQualityReviewWindow(qc_raw, UNREVIEWED_EXTRACTION,
        trial_quality_report, initial_pairs=()).show()
else:
    selected_trial_pairs = ()  # explicit unattended verification: no artifact rejection
if selected_trial_pairs is not None:
    selected_trial_pairs = TrialExclusions(selected_trial_pairs, provenance=trial_quality_report.attrs["provenance"])
    pending_reviewed = apply_trial_exclusions(UNREVIEWED_EXTRACTION, selected_trial_pairs,
        smoothing_sigma_seconds=FEATURE_SMOOTH_SIGMA_SECONDS, slope_method=FEATURE_SLOPE_METHOD,
        max_baseline_invalid_fraction=MAX_BASELINE_INVALID_FRACTION)
    trial_review_dir = RUN_DIR / ("trial_review_" + timestamp())
    trial_review_dir.mkdir()
    trial_quality_report.to_csv(trial_review_dir / "trial_quality_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_trial_pairs, columns=["channel", "original_trial_id"]).to_csv(
        trial_review_dir / "confirmed_exclusions.csv", index=False, encoding="utf-8-sig")
    pending_reviewed.features.to_csv(trial_review_dir / "reviewed_features.csv", index=False, encoding="utf-8-sig")
    import json
    (trial_review_dir / "review_settings.json").write_text(json.dumps(dict(
        settings=trial_quality_report.attrs["settings"], units=trial_quality_report.attrs["units"],
        provenance=trial_quality_report.attrs["provenance"],
        exclusions=list(selected_trial_pairs)), indent=2), encoding="utf-8")
    extraction, features = pending_reviewed, pending_reviewed.features
    from copy import deepcopy
    reviewed_trial_settings = deepcopy(TRIAL_QC_SETTINGS)
    TRIAL_QC_READY = True
    display(features[["channel", "is_bad", "exclusion_reason"]])
    print("Confirmed channel-trial exclusions:", selected_trial_pairs)
else:
    print("Trial review cancelled. Confirm a fresh review before fitting.")''')
    first = notebook.cells[10]
    first.source = first.source.replace("SECOND_ROUND_READY = False\nensure_extraction_config",
        "SECOND_ROUND_READY = False\nclose_explorers(EXPLORERS)\nif not TRIAL_QC_READY:\n    raise ValueError('Confirm trial QC before round one.')\nensure_extraction_config", 1)
    first.source = first.source.replace("random_state=RANDOM_STATE)",
        "random_state=RANDOM_STATE, weight_preset=WEIGHT_PRESET,\n    group_weights=GROUP_WEIGHTS if WEIGHT_PRESET == 'custom' else None)", 1)
    first.source = first.source.replace("first_result, first_extraction, ROUND1_DIR =", "first_fit_settings = dict(fit_settings(), first_k=FIRST_N_CLUSTERS)\nfirst_result, first_extraction, ROUND1_DIR =")
    first.source = first.source.replace("round_config=dict(round=1,", "round_config=dict(weight_preset=WEIGHT_PRESET, group_weights=GROUP_WEIGHTS if WEIGHT_PRESET == 'custom' else None, round=1,")
    second = notebook.cells[14]
    second.source = second.source.replace("SECOND_ROUND_READY = False", "SECOND_ROUND_READY = False\nensure_extraction_config(extraction_settings(), extracted_settings)\nclose_explorers(EXPLORERS)\nif not TRIAL_QC_READY:\n    raise ValueError('Confirm trial QC and rerun round one.')\nensure_fit_settings(fit_settings(), first_fit_settings)", 1)
    second.source = second.source.replace("random_state=RANDOM_STATE)",
        "random_state=RANDOM_STATE, weight_preset=WEIGHT_PRESET,\n        group_weights=GROUP_WEIGHTS if WEIGHT_PRESET == 'custom' else None)", 1)
    second.source = second.source.replace("round_config=dict(round=2,", "round_config=dict(weight_preset=WEIGHT_PRESET, group_weights=GROUP_WEIGHTS if WEIGHT_PRESET == 'custom' else None, round=2,")
    surface = notebook.cells[16]
    surface.source = surface.source.replace("if RUN_3D_RENDER or RUN_SELECTABLE_CLUSTER_RENDER:", '''if RUN_3D_RENDER or RUN_SELECTABLE_CLUSTER_RENDER:
    if not TRIAL_QC_READY:
        raise ValueError("Confirm trial QC and refit before surface display.")
    ensure_extraction_config(extraction_settings(), extracted_settings)
    ensure_first_round(globals().get("first_extraction"), extraction)
    ensure_fit_settings(fit_settings(), first_fit_settings)''', 1)
    surface.source = surface.source.replace('globals().get("LINK_SURFACE_EXPLORER", True)', "LINK_SURFACE_EXPLORER").replace(
        'globals().get("CACHE_MAX_MB", 512)', "CACHE_MAX_MB")
    continuous = notebook.cells[18]
    continuous.source = continuous.source.replace("ensure_extraction_config(extraction_settings(), extracted_settings)",
        "if not TRIAL_QC_READY:\n    raise ValueError('Confirm trial QC and refit before display.')\nensure_fit_settings(fit_settings(), first_fit_settings)\nensure_extraction_config(extraction_settings(), extracted_settings)")
    continuous.source = continuous.source.replace("display_sfreq=DISPLAY_SFREQ)", "display_sfreq=DISPLAY_SFREQ, cache=POWER_CACHE)", 1)
    diagnostics = nbf.v4.new_code_cell('''ensure_extraction_config(extraction_settings(), extracted_settings)
ensure_first_round(globals().get("first_extraction"), extraction)
ensure_fit_settings(fit_settings(), first_fit_settings)
if not TRIAL_QC_READY:
    raise ValueError("Confirm trial QC and refit before diagnostics.")
diagnostic_rounds = [("first", first_result, ROUND1_DIR)]
if SECOND_ROUND_READY:
    diagnostic_rounds.append(("second", second_result, ROUND2_DIR))
diagnostic_outputs = {}
for round_name, fitted_result, round_dir in diagnostic_rounds:
    diagnostic_dir = round_dir / ("diagnostics_" + timestamp())
    diagnostic_dir.mkdir()
    save_inspection(extraction, fitted_result, diagnostic_dir,
        sigma_seconds=DISPLAY_SMOOTH_SIGMA_SECONDS, n_bootstrap=INSPECTION_BOOTSTRAP_DRAWS, random_state=RANDOM_STATE)
    if RUN_STABILITY:
        stability_report = run_stability(extraction, fitted_result, n_resamples=STABILITY_RESAMPLES,
            random_state=RANDOM_STATE, fit_kwargs=fit_settings(),
            feature_kwargs=extraction.feature_settings, show_progress=STABILITY_PROGRESS_WINDOW)
        save_stability(stability_report, diagnostic_dir)
        display(stability_report.channels)
        print(round_name, stability_report.summary)
    diagnostic_outputs[round_name] = diagnostic_dir
    print(diagnostic_dir)''')
    explorer = nbf.v4.new_code_cell('''ensure_extraction_config(extraction_settings(), extracted_settings)
ensure_first_round(globals().get("first_extraction"), extraction)
ensure_fit_settings(fit_settings(), first_fit_settings)
if not TRIAL_QC_READY:
    raise ValueError("Confirm trial QC and refit before exploration.")
if OPEN_EXPLORER:
    fitted_result = second_result if SECOND_ROUND_READY else first_result
    explorer_dir = ROUND2_DIR if SECOND_ROUND_READY else ROUND1_DIR
    explorer_display_run = run_display(extraction, fitted_result, output_dir=explorer_dir,
        fif_path=FIF_PATH, onset_path=ONSET_PATH, offset_path=OFFSET_PATH, wav_path=WAV_PATH,
        explorer_settings=dict(pre_seconds=globals().get("TRIAL_PRE_SECONDS", 1.0),
            post_seconds=globals().get("TRIAL_POST_SECONDS", 1.0),
            sigma_seconds=DISPLAY_SMOOTH_SIGMA_SECONDS, display_sfreq=DISPLAY_SFREQ,
            trial_ids=globals().get("TRIAL_PAIR_IDS", None)),
        cache_max_mb=CACHE_MAX_MB, explorer_only=True, backend=DISPLAY_BACKEND)''')
    notebook.cells[9:9] = [nbf.v4.new_markdown_cell("## 4.1 Trial 质量人工复审\n\n候选伪迹默认只标记，不自动剔除。勾选通道 × 原始 trial 编号并确认后，四频带共同剔除对应响应。取消不会应用待选项；有效 trial 太少的电极仍按已有质量规则排除并报告。"), qc]
    notebook.cells[5:5] = [nbf.v4.new_markdown_cell("## 2.1 可靠性与查看参数\n\n默认 legacy 保持原权重；可改 balanced、morphology、amplitude 或 custom。修改分类权重后重跑第一轮及第二轮。质量阈值只生成候选项，剔除仍须人工确认。"), settings]
    notebook.cells.extend([nbf.v4.new_markdown_cell("## 10. 稳定性与同类细节\n\n按已经拟合的 K 做分半及 trial 重采样，不自动选 K，也不因支持次数少而阻止分类。细线/电极热图反映同类内部差异；trial 热图保留原编号。试次重采样置信带以固定电极和标签为条件，不代表显著性或分类可信概率。"), diagnostics,
        nbf.v4.new_markdown_cell("## 11. 交互式曲线查看\n\n相邻配对选择、上一对/下一对、通道/类别与频带查看。连续曲线保留真实秒数及音频；分类剔除 trial 只作标记，不删除原始记录。脑表面点选电极可联动查看其四频带曲线。"), explorer])
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace("ensure_fit_settings(fit_settings(), first_fit_settings)",
                "ensure_fit_settings(dict(fit_settings(), first_k=FIRST_N_CLUSTERS), first_fit_settings)")
            if "pending_first = fit_joint(" in cell.source:
                cell.source = cell.source.replace("pending_first = fit_joint(",
                    "ensure_trial_review(TRIAL_QC_SETTINGS, reviewed_trial_settings)\npending_first = fit_joint(")
            if "ensure_fit_settings(dict(" in cell.source:
                lines = []
                for line in cell.source.splitlines():
                    if line.lstrip().startswith("ensure_fit_settings(dict("):
                        indent = line[:len(line) - len(line.lstrip())]
                        lines.append(indent + "ensure_trial_review(TRIAL_QC_SETTINGS, reviewed_trial_settings)")
                    lines.append(line)
                cell.source = "\n".join(lines)
    return notebook


def _preserve_assignments(source, revised, names):
    originals = {}
    lines = source.splitlines()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                originals[node.targets[0].id] = "\n".join(lines[node.lineno - 1:node.end_lineno])
    lines = revised.splitlines()
    for node in reversed(ast.parse(revised).body):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in originals:
                lines[node.lineno - 1:node.end_lineno] = [originals[node.targets[0].id]]
    return "\n".join(lines)


def update_reliability_cells(output):
    existing = nbf.read(output, as_version=4)
    if any("WEIGHT_PRESET =" in cell.source for cell in existing.cells) and _has_process_displays(existing):
        return update_browser_cells(output)
    revised = build_reliability_notebook()
    _owned_cell(existing, "code", "SUBJECT_ID =", "def extraction_settings():")
    markers = [("PROJECT_DIR =", "from fourband.analysis import"),
        ("pending_settings = extraction_settings()",), ("pending_first = fit_joint(",),
        ("selected_review = FourBandReviewWindow(",), ("if RUN_3D_RENDER",),
        ("TRIAL_PAIR_IDS =", "load_continuous_trials(")]
    positions = [(_owned_cell(existing, "code", *m), _owned_cell(revised, "code", *m)) for m in markers]
    for old_index, new_index in positions:
        source = revised.cells[new_index].source
        if "TRIAL_PAIR_IDS =" in source:
            source = _preserve_display_parameters(existing.cells[old_index].source, source)
        existing.cells[old_index].source = source
    additional = [("WEIGHT_PRESET =", "def fit_settings():"), ("selected_trial_pairs =", "TrialQualityReviewWindow("),
        ("diagnostic_rounds =", "run_stability("), ("if OPEN_EXPLORER:",)]
    setting_names = {"WEIGHT_PRESET", "GROUP_WEIGHTS", "OPEN_TRIAL_QC_WINDOW", "TRIAL_QC_SETTINGS", "RUN_STABILITY",
        "STABILITY_RESAMPLES", "STABILITY_PROGRESS_WINDOW", "INSPECTION_BOOTSTRAP_DRAWS", "OPEN_EXPLORER",
        "LINK_SURFACE_EXPLORER", "CACHE_MAX_MB"}
    for markers in additional:
        new_index = _owned_cell(revised, "code", *markers)
        matches = [i for i, c in enumerate(existing.cells) if c.cell_type == "code" and all(m in c.source for m in markers)]
        if len(matches) > 1:
            raise ValueError("Multiple reliability cells found; no file was changed.")
        if matches:
            old_index = matches[0]
            source = revised.cells[new_index].source
            if "WEIGHT_PRESET =" in source:
                source = _preserve_assignments(existing.cells[old_index].source, source, setting_names)
            existing.cells[old_index].source = source
        else:
            if "WEIGHT_PRESET =" in markers:
                insert_at = _owned_cell(existing, "code", "def extraction_settings():") + 1
            elif "selected_trial_pairs =" in markers:
                insert_at = _owned_cell(existing, "code", "pending_settings = extraction_settings()") + 1
            else:
                insert_at = len(existing.cells)
            existing.cells[insert_at:insert_at] = [revised.cells[new_index - 1], revised.cells[new_index]]
    _update_browser_sources(existing)
    nbf.validate(existing)
    nbf.write(existing, output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-review", action="store_true")
    parser.add_argument("--update-continuous", action="store_true")
    parser.add_argument("--update-reliability", action="store_true")
    parser.add_argument("--update-browser", action="store_true")
    args = parser.parse_args()
    output = Path(__file__).with_name("ECoG_4band_clustering_pipeline.ipynb")
    if args.update_browser:
        update_browser_cells(output)
    elif args.update_reliability:
        update_reliability_cells(output)
    elif args.update_continuous:
        update_continuous_cells(output)
    elif args.update_review:
        update_review_cells(output)
    elif output.exists():
        update_review_cells(output)
    else:
        nbf.write(build_notebook(), output)
    print(output)
