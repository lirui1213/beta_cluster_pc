"""Headless, file-backed numerical actions for the browser workspace.

Only compact JSON metadata crosses the worker boundary. Trusted extraction and
model objects stay in joblib snapshots under the server-selected run directory.
"""
from datetime import datetime
import json
from numbers import Integral
import os
from pathlib import Path
from uuid import uuid4

os.environ["MPLBACKEND"] = "Agg"

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from audio_cross_3band_utils import load_ecog_fif
from .analysis import load_fourband, fit_joint
from .browser_data import json_safe, session_metadata
from .review import pca_review_table, review_exclusion_audit, refit_reviewed
from .stability import evaluate_stability, save_stability
from .trial_qc import (TrialExclusions, apply_trial_exclusions,
                       detect_trial_quality, validate_trial_report)
from .workflow import save_extraction, save_round
from .workspace_display import validate_source_identity


_ACTIONS = {"extract", "detect_qc", "confirm_trials", "fit_first",
            "preview_review", "fit_second", "waveform", "stability"}
_FIT_KEYS = ("band_weights", "weight_preset", "group_weights",
             "correlation_threshold", "random_state")


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False,
                                        allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_joblib(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _progress(run, stage, **details):
    directory = Path(os.environ.get("FOURBAND_WORKSPACE_TASK_DIR", str(run)))
    _atomic_json(directory / "progress.json", dict(stage=stage, **details))


def _revision(run, prefix):
    return run / (prefix + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                  + "_" + uuid4().hex[:8])


def _state(run):
    path = run / "action_state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _save_state(run, state):
    _atomic_json(run / "action_state.json", {**state, "schema_version": 1})


def _source(run):
    return joblib.load(run / "extraction" / "source.joblib")


def _report(run):
    return joblib.load(run / "extraction" / "trial_report.joblib")


def _feature_settings(config):
    return dict(smoothing_sigma_seconds=config.get("feature_sigma_seconds", .06),
                slope_method=config.get("slope_method", "savgol"),
                max_baseline_invalid_fraction=config.get("max_baseline_invalid_fraction", .25))


def _fit_settings(config):
    return {key: config[key] for key in _FIT_KEYS if key in config}


def _reviewed(run):
    state = _state(run)
    path = run / "extraction" / "reviewed.joblib"
    report = _report(run)
    if (not path.is_file() or not state.get("confirmation") or
            state["confirmation"]["report_fingerprint"] != report.attrs.get("report_fingerprint") or
            state["confirmation"]["provenance"] != report.attrs.get("provenance")):
        raise ValueError("Confirm trials against the current QC report before fitting.")
    return joblib.load(path)


def _accepted(run, number):
    state = _state(run)
    key = "first" if number == 1 else "second"
    if not state.get(key):
        raise ValueError(f"No accepted round {number}; confirm trials and fit it first.")
    # Workers clone only active snapshots, not prior round plot directories.
    return (joblib.load(run / "extraction" / "first.joblib"),
            joblib.load(run / f"{key}.joblib"), Path(state[key]))


def _validate_k(value, count, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 2 <= value <= count:
        raise ValueError(f"{name} must be an integer between 2 and {count} usable electrodes.")
    return int(value)


def _qc_data(extraction, report):
    reference = next(iter(extraction.epochs.values()))
    baseline = {}
    for band, epoch in extraction.epochs.items():
        diagnostics = epoch.baseline_diagnostics
        baseline[band] = dict(mode=epoch.baseline_mode,
            n_trials=len(epoch.events), bad_channels=sorted(epoch.bad_channels),
            invalid_count=int((~diagnostics.valid.astype(bool)).sum())
            if diagnostics is not None and "valid" in diagnostics else None)
    columns = ["channel", "original_trial_id", "candidate", "reasons", "excluded"]
    if "raw_bad" in report:
        columns.append("raw_bad")
    columns.extend(name for name in ("flag_spike", "flag_jump", "flag_flat",
                                    "flag_clipping_candidate", "flag_nonfinite") if name in report)
    exclusions = list(getattr(extraction, "trial_exclusions", ()))
    return dict(qc_rows=report.loc[:, columns].to_dict("records"),
                qc_summary=dict(total_pairs=len(report), candidate_count=int(report.candidate.sum()),
                                excluded_count=len(exclusions)),
                channels=list(reference.channel_names),
                events=reference.events.to_dict("records"), baseline_summary=baseline,
                qc_settings=report.attrs["settings"], qc_units=report.attrs["units"],
                trial_exclusions=exclusions)


def _artifacts(folder):
    return [str(path.resolve()) for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".png", ".csv", ".json"}]


def _extract(run, config, payload):
    sources = config["sources"]
    keys = ("baseline_mode", "baseline_window", "baseline_edge_seconds", "baseline_relative_floor",
            "duration_mad_threshold", "min_trials_after_filter", "progress_points", "pre_post_window")
    options = {key: config[key] for key in keys if key in config}
    options.update(_feature_settings(config))
    options["baseline_auto_discover_segments"] = False
    options.update({key: sources.get(key) for key in ("baseline_fif", "baseline_segments_csv")})
    _progress(run, "Extracting four-band epochs and ECoG features")
    extraction = load_fourband(*(sources[key] for key in
        ("fif_path", "onset_path", "offset_path", "wav_path")), picks="ecog", **options)
    _progress(run, "Detecting raw trial quality candidates")
    raw = load_ecog_fif(sources["fif_path"], preload=False)
    report = detect_trial_quality(raw, extraction)
    _progress(run, "Saving extraction and trial quality report")
    folder = _revision(run, "extraction")
    save_extraction(extraction, folder, config)
    _atomic_joblib(folder / "source.joblib", extraction)
    _atomic_joblib(folder / "trial_report.joblib", report)
    report.to_csv(folder / "trial_quality.csv", index=False, encoding="utf-8-sig")
    data = _qc_data(extraction, report)
    _atomic_joblib(run / "extraction" / "source.joblib", extraction)
    _atomic_joblib(run / "extraction" / "trial_report.joblib", report)
    _save_state(run, {})
    return dict(stage="trial_review", data=data, artifacts=_artifacts(folder))


def _detect_qc(run, config, payload):
    _progress(run, "Loading source extraction and raw recording for QC")
    extraction = _source(run)
    raw = load_ecog_fif(config["sources"]["fif_path"], preload=False)
    _progress(run, "Recomputing trial quality candidates")
    report = detect_trial_quality(raw, extraction, settings=payload.get("settings"))
    data = _qc_data(extraction, report)
    folder = _revision(run, "trial_qc")
    folder.mkdir()
    _atomic_joblib(folder / "trial_report.joblib", report)
    report.to_csv(folder / "trial_quality.csv", index=False, encoding="utf-8-sig")
    _atomic_json(folder / "settings.json", report.attrs)
    _atomic_joblib(run / "extraction" / "trial_report.joblib", report)
    _save_state(run, {})
    return dict(stage="trial_review", data=data, artifacts=_artifacts(folder))


def _confirm_trials(run, config, payload):
    if "pairs" not in payload or not isinstance(payload["pairs"], list):
        raise ValueError("Explicit pairs list is required, including an empty list.")
    _progress(run, "Validating trial report provenance")
    extraction, report = _source(run), _report(run)
    raw = load_ecog_fif(config["sources"]["fif_path"], preload=False)
    provenance = validate_trial_report(raw, extraction, report)
    pairs = TrialExclusions(payload["pairs"], provenance=provenance)
    _progress(run, "Applying confirmed exclusions across all four bands")
    reviewed = apply_trial_exclusions(extraction, pairs, **_feature_settings(config))
    folder = _revision(run, "trial_review")
    folder.mkdir()
    confirmation = dict(pairs=list(reviewed.trial_exclusions), provenance=provenance,
                        report_fingerprint=report.attrs["report_fingerprint"])
    _atomic_joblib(folder / "reviewed.joblib", reviewed)
    _atomic_json(folder / "trial_confirmation.json", confirmation)
    data = _qc_data(reviewed, report)
    selected = set(reviewed.trial_exclusions)
    for row in data["qc_rows"]:
        row["excluded"] = (row["channel"], row["original_trial_id"]) in selected
    _atomic_joblib(run / "extraction" / "reviewed.joblib", reviewed)
    _save_state(run, dict(confirmation=confirmation))
    return dict(stage="trial_confirmed", data=data, artifacts=_artifacts(folder))


def _display_request(run, folder, extraction, model, config, number):
    options = dict(config.get("surface_options") or {})
    warnings = []
    root = (Path(options["subjects_dir"]) / options["subject"]
            if options.get("subjects_dir") and options.get("subject") else None)
    surface = bool(root and options.get("electrode_csv") and
                   Path(options["electrode_csv"]).is_file() and
                   any((root / "surf" / f"{hemi}.pial").is_file() for hemi in ("lh", "rh")))
    if not surface:
        warnings.append("Anatomy is unavailable; brain display is disabled.")
    request = dict(extraction=extraction, result=model, output_dir=str(folder),
        sources=dict(config["sources"]), surface_options=options,
        show_surface=surface, show_selectable=surface, link_explorer=True,
        explorer_only=False, explorer_settings=dict(
            sigma_seconds=config.get("display_sigma_seconds", .1),
            display_sfreq=config.get("display_sfreq", 100.)),
        cache_max_mb=512, backend="browser", subject=config.get("subject"),
        block=config.get("block"), round=number, input_identity=config.get("input_identity"))
    try:
        metadata = session_metadata(request)
    except ValueError as error:
        if "No adjacent retained original trial pairs" not in str(error):
            raise
        request["link_explorer"] = False
        warnings.append("No adjacent retained original trial pairs; continuous curves are disabled.")
        metadata = session_metadata(request)
    warnings.extend(metadata.get("warnings", []))
    request["warnings"] = warnings
    path = folder / "request.joblib"
    _atomic_joblib(path, request)
    return str(path), warnings


def _fit_round(run, config, extraction, model, number, *, audit=None, controls=None):
    _progress(run, f"Exporting round {number} plots and diagnostics")
    folder = _revision(run, f"round{number}")
    fit_config = dict(config, **(controls or {}), round=number)
    save_round(extraction, model, folder, n_clusters=model.kmeans.n_clusters,
               audit=audit, sigma_seconds=config.get("display_sigma_seconds", .1),
               display_sfreq=config.get("display_sfreq", 100.),
               slope_method=config.get("slope_method", "savgol"), round_config=fit_config)
    _progress(run, f"Saving round {number} trusted display snapshot")
    _atomic_joblib(folder / "extraction.joblib", extraction)
    request, warnings = _display_request(run, folder, extraction, model, config, number)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    data = dict(pca=pca_review_table(model).to_dict("records"),
                cluster_sizes=summary["cluster_sizes"], summary=summary,
                plot_paths={path.stem: str(path) for path in sorted(folder.glob("*.png"))},
                round=number, round_dir=str(folder), warnings=warnings)
    # Publish accepted aliases only after numerical work, exports and requests succeed.
    state = _state(run)
    key = "first" if number == 1 else "second"
    state[key] = str(folder)
    state[key + "_fit_settings"] = _fit_settings(config)
    if number == 1:
        state.pop("second", None)
        _atomic_joblib(run / "extraction" / "first.joblib", extraction)
    _atomic_joblib(run / f"{key}.joblib", model)
    _save_state(run, state)
    return dict(stage="first_ready" if number == 1 else "second_ready", data=data,
                artifacts=_artifacts(folder), display_request=request)


def _fit_first(run, config, payload):
    _progress(run, "Loading explicitly reviewed extraction")
    extraction = _reviewed(run)
    overrides = payload.get("fit_settings", {})
    if not isinstance(overrides, dict) or set(overrides) - set(_FIT_KEYS):
        raise ValueError("fit_settings contains unknown classification settings.")
    config = dict(config, **overrides)
    seed = config.get("random_state", 50)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("random_state must be an unsigned 32-bit integer.")
    threshold = config.get("correlation_threshold", .95)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("correlation_threshold must be finite and between zero and one.")
    k = _validate_k(payload.get("first_k", config.get("first_k", 4)),
                    int((~extraction.features.is_bad.astype(bool)).sum()), "first_k")
    _progress(run, f"Fitting first round with manual K={k}")
    model = fit_joint(extraction.features, n_clusters=k, **_fit_settings(config))
    config["first_k"] = k
    response = _fit_round(run, config, extraction, model, 1, controls=dict(first_k=k))
    _atomic_json(run / "config.json", config)
    response["config_update"] = dict(first_k=k, **_fit_settings(config))
    return response


def _review_controls(config, payload):
    controls = dict(bad_clusters=payload.get("bad_clusters", []),
                    manual_bad_channels=payload.get("manual_bad_channels", []),
                    pca_distance_threshold=payload.get("pca_distance_threshold"),
                    distance_reference=payload.get("distance_reference", "global"))
    return controls, payload.get("second_k", config.get("second_k", 4))


def _preview_review(run, config, payload):
    _progress(run, "Computing first-round exclusion union")
    _, first, _ = _accepted(run, 1)
    controls, k = _review_controls(config, payload)
    audit = review_exclusion_audit(first, **controls)
    remaining = int((~audit.excluded_final).sum())
    valid_k = isinstance(k, Integral) and not isinstance(k, bool) and 2 <= k <= remaining
    preview = dict(rows=audit.to_dict("records"), remaining=remaining,
                   excluded=int(audit.excluded_final.sum()), second_k=k, valid_k=valid_k,
                   excluded_channels=audit.loc[audit.excluded_final, "channel"].tolist())
    return dict(stage="first_ready", data=dict(preview=preview), artifacts=[])


def _fit_second(run, config, payload):
    _progress(run, "Validating manual second-round exclusions and K")
    extraction, first, _ = _accepted(run, 1)
    config = dict(config, **_state(run).get("first_fit_settings", _fit_settings(config)))
    controls, k = _review_controls(config, payload)
    audit = review_exclusion_audit(first, **controls)
    k = _validate_k(k, int((~audit.excluded_final).sum()), "second_k")
    _progress(run, f"Independently refitting second round with manual K={k}")
    model, audit = refit_reviewed(extraction.features, first, n_clusters=k,
                                 **controls, **_fit_settings(config))
    config["second_k"] = k
    response = _fit_round(run, config, extraction, model, 2, audit=audit,
                          controls=dict(controls, second_k=k))
    _atomic_json(run / "config.json", config)
    response["config_update"] = dict(second_k=k)
    return response


def _extrema_indices(values, limit=4096):
    if len(values) <= limit:
        return np.arange(len(values))
    # Reserve two extrema and a missing-data marker per bin, plus endpoints.
    edges = np.linspace(1, len(values) - 1, (limit - 2) // 3 + 1, dtype=int)
    indices = [0, len(values) - 1]
    for left, right in zip(edges[:-1], edges[1:]):
        block = values[left:right]
        finite = np.flatnonzero(np.isfinite(block))
        if len(finite):
            indices.extend((left + finite[np.argmin(block[finite])],
                            left + finite[np.argmax(block[finite])]))
        missing = np.flatnonzero(~np.isfinite(block))
        if len(missing):
            indices.append(left + missing[0])
    return np.unique(indices)


def _waveform(run, config, payload):
    _progress(run, "Validating raw waveform channel and original trial ID")
    channel, trial = payload.get("channel"), payload.get("trial_id")
    if isinstance(trial, bool) or not isinstance(trial, Integral):
        raise ValueError("trial_id must be an integer original trial ID.")
    extraction, report = _source(run), _report(run)
    reference = next(iter(extraction.epochs.values()))
    selected = report.loc[(report.channel == channel) & (report.original_trial_id == trial)]
    event = reference.events.loc[reference.events.original_trial_id == trial]
    if channel not in reference.channel_names or len(selected) != 1 or len(event) != 1:
        raise ValueError("Waveform channel/trial must belong to the current report and epochs.")
    row = selected.iloc[0]
    raw = load_ecog_fif(config["sources"]["fif_path"], preload=False)
    _progress(run, "Validating raw waveform against trial report provenance")
    validate_trial_report(raw, extraction, report)
    sfreq = float(raw.info["sfreq"])
    start = max(0., float(row.detection_start_seconds))
    stop = min(float(raw.times[-1]), float(row.detection_stop_seconds), start + 120.)
    left = max(0, int(np.ceil(start * sfreq)))
    right = min(raw.n_times, int(np.floor(stop * sfreq)) + 1)
    if left >= right:
        raise ValueError("Requested trial has no raw samples inside the recording.")
    _progress(run, "Reading bounded original raw voltage samples")
    values = raw.get_data(picks=[channel], start=left, stop=right)[0]
    indices = _extrema_indices(values)
    waveform = dict(channel=channel, trial_id=int(trial), time=(left + indices) / sfreq,
                    voltage=values[indices], units="volts", sfreq=sfreq,
                    onset=float(event.iloc[0].onset), offset=float(event.iloc[0].offset),
                    start_seconds=start, stop_seconds=stop, sample_count=len(values),
                    downsampled=len(indices) < len(values))
    return dict(stage="trial_review", data=dict(waveform=waveform), artifacts=[])


def _stability(run, config, payload):
    number = payload.get("round")
    if isinstance(number, bool) or not isinstance(number, Integral) or number not in (1, 2):
        raise ValueError("round must select accepted round 1 or 2.")
    _progress(run, f"Loading accepted round {number} for fixed-K stability")
    extraction, model, _ = _accepted(run, number)
    report = evaluate_stability(extraction, model, n_resamples=payload.get("n_resamples", 30),
        random_state=config.get("random_state", 50),
        fit_kwargs=_state(run).get("first_fit_settings" if number == 1 else "second_fit_settings",
                                   _fit_settings(config)),
        progress=lambda done, total: _progress(run, f"Round {number} stability resampling",
                                              completed=done, total=total))
    _progress(run, f"Saving round {number} stability diagnostics")
    folder = _revision(run, f"stability_round{number}")
    save_stability(report, folder)
    stability = dict(round=int(number), summary=report.summary,
                     channels=report.channels.to_dict("records"),
                     replicates=report.replicates.to_dict("records"))
    return dict(stage="first_ready" if number == 1 else "second_ready",
                data=dict(stability=stability), artifacts=_artifacts(folder))


def execute_action(run_dir, action, payload):
    """Execute one action from effective config and trusted local snapshots.

    ``confirm_trials`` requires an explicit ``pairs`` list. Fits use manual
    ``first_k``/``second_k``; review controls match ``review_exclusion_audit``.
    Read-only actions return their natural stage; the parent preserves its stage.
    """
    run = Path(run_dir).resolve()
    if action not in _ACTIONS or not isinstance(payload, dict):
        raise ValueError("Unknown numerical action or invalid action payload.")
    _progress(run, f"Loading effective configuration for {action}")
    try:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        validate_source_identity(config["sources"], config.get("input_identity"))
        result = globals()["_" + action](run, config, payload)
        result = json_safe(result)
        _progress(run, f"Completed {action}")
        return result
    except Exception as error:
        _progress(run, f"Failed {action}", error=str(error))
        raise
