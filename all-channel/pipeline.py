"""Extract one row per selected block/channel and fit one global beta model."""
from dataclasses import dataclass
from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from audio_cross_3band_utils import build_beta_progress_feature_table_from_files
from clustering_pipeline import (cluster_electrodes, select_beta_progress_feature_columns,
                                 default_beta_progress_feature_weights)
from inputs import validate_manifest
from plots import save_global_plots
from dynamics import save_audio_plots
from full_curves import save_complete_plots
from curve_similarity import build_curve_features,fit_curve_clusters
from native_curves import summarize_native_files


@dataclass
class GlobalAnalysisResult:
    features: pd.DataFrame
    assignments: pd.DataFrame
    clustering: object
    output_dir: Path


def refresh_saved_plots(output_dir):
    """Redraw existing results without rereading signals or refitting clusters."""
    output=Path(output_dir)
    assignments=pd.read_csv(output/'global_assignments.csv')
    with np.load(output/'mean_speech_curves.npz') as source:
        cache={key:source[key] for key in source.files}
    if tuple(assignments.sample_id)!=tuple(cache['sample_id']):
        raise ValueError('Saved assignments and curves must have the same sample order')
    save_global_plots(assignments,cache['beta'],cache['progress_percent'],output)
    save_audio_plots(assignments,cache,output)
    if 'pre_beta' in cache:
        save_complete_plots(assignments,cache,output)


def save_fitted_round(features, clustering, assignments, cache, output, *, n_clusters, random_state):
    """Export the same complete diagnostics for either fitted round."""
    from review import pca_review_table
    assignments.to_csv(output / "global_assignments.csv", index=False, encoding="utf-8-sig")
    assignments.loc[assignments.is_bad].to_csv(output / "excluded_channels.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": clustering.feature_columns, "weight": clustering.feature_weights}).to_csv(
        output / "selected_feature_weights.csv", index=False, encoding="utf-8-sig")
    pca_review_table(features, clustering).rename(columns={"first_label": "label", "first_pc1": "pc1", "first_pc2": "pc2", "first_pc3": "pc3"}).to_csv(
        output / "pca_review_distances.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(output / "mean_speech_curves.npz", **cache)
    curve_stats = save_global_plots(assignments, cache["beta"], cache["progress_percent"], output)
    save_audio_plots(assignments, cache, output)
    if 'pre_beta' in cache:
        save_complete_plots(assignments,cache,output)
    rows = []
    for label, mean, sem, counts in curve_stats:
        rows.extend({"label": int(label), "progress_percent": float(p), "mean_beta_shape": float(m),
                     "sample_sem": float(s), "n_samples": int(n)}
                    for p, m, s, n in zip(cache["progress_percent"], mean, sem, counts))
    pd.DataFrame(rows).to_csv(output / "global_beta_curve_summary.csv", index=False, encoding="utf-8-sig")
    coupling_columns = [c for c in features if c.startswith('relation__') or
                        (c.startswith("beta_progress__") and ("audio" in c or "slope_product" in c or "same_sign" in c))]
    pd.concat([assignments[["sample_id", "subject_id", "block_id", "channel", "label"]],
               features[coupling_columns].reset_index(drop=True)], axis=1).to_csv(
                   output / "sample_audio_relationship.csv", index=False, encoding="utf-8-sig")
    unique, counts = np.unique(clustering.labels[clustering.clean_mask], return_counts=True)
    silhouette = None
    if 1 < len(unique) < int(clustering.clean_mask.sum()):
        silhouette = float(silhouette_score(clustering.weighted_clean_features,
                                           clustering.labels[clustering.clean_mask],
                                           sample_size=min(5000, int(clustering.clean_mask.sum())),
                                           random_state=random_state))
    summary = {"n_samples": len(features), "n_included": int(clustering.clean_mask.sum()),
               "n_requested_clusters": n_clusters, "n_actual_clusters": len(unique),
               "min_cluster_size": int(counts.min()), "silhouette": silhouette,
               "silhouette_sample_cap": 5000, "stability_evaluated": False,
               "interpretation": "Descriptive clustering and paired beta/audio curves; sample SEM is not a stability test."}
    if any(c.startswith('curve__') for c in clustering.feature_columns):
        summary.update(classification_mode='curve_shape', distance='weighted Euclidean',
                       beta_centering='subtract each sample pre-onset mean; divide full curve by its own RMS',
                       shared_beta_scale=float(clustering.scaler.beta_scale_))
    (output / "global_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(clustering, output / "global_model.joblib")
    if len(unique) != n_clusters:
        warnings.warn("Fewer distinct clusters than requested; inspect global_summary.json", RuntimeWarning)


def fit_global(features, *, n_clusters=2, random_state=50, correlation_threshold=.95):
    """Fit curve/relationship predictors; retain legacy beta-table compatibility."""
    if features.empty:
        raise ValueError("No channel samples")
    required = {"subject_id", "block_id", "channel", "sample_id", "is_bad"}
    if not required.issubset(features.columns):
        raise ValueError(f"Missing sample identity columns: {required - set(features.columns)}")
    if features.sample_id.duplicated().any() or features[["subject_id", "block_id", "channel"]].duplicated().any():
        raise ValueError("Duplicate channel samples")
    if any(c.startswith('curve__') for c in features):
        result=fit_curve_clusters(features,n_clusters=n_clusters,random_state=random_state)
    else:
        predictors = [c for c in features if c.startswith("beta_progress__")]
        table = features[["channel", "is_bad", *predictors]].copy()
        table["channel"] = features.sample_id.to_numpy()
        selected = select_beta_progress_feature_columns(table, correlation_threshold=correlation_threshold)
        weights = default_beta_progress_feature_weights(selected)
        result = cluster_electrodes(table, n_clusters=n_clusters, feature_columns=selected,
                                   feature_weights=weights, random_state=random_state)
    metadata = [c for c in features if not c.startswith(("beta_progress__",'curve__','relation__'))]
    assignments = features[metadata].copy().reset_index(drop=True)
    assignments["label"] = result.labels
    for i in range(3):
        assignments[f"pc{i + 1}"] = result.pca_scores[:, i]
    return result, assignments


def run_analysis(manifest, output_dir, *, n_clusters=2, random_state=50,
                 correlation_threshold=.95, progress_points=101,
                 duration_mad_threshold=2.5, min_trials_after_filter=3,
                 pre_post_window=None, baseline_window=(-1.3, -.3),
                 baseline_relative_floor=1e-3, baseline_edge_seconds=.3,
                 smoothing_sigma_seconds=.06, slope_method="savgol",
                 max_baseline_invalid_fraction=.25, display_sigma_seconds=.10,
                 curve_context_seconds=.75, curve_context_points=51, save_channel_figures=True):
    """Validate all inputs before extraction; fail explicitly, never skip a block."""
    validate_manifest(manifest)
    context=curve_context_seconds if pre_post_window is None else min(curve_context_seconds,pre_post_window)
    if not np.isfinite(context) or context<=0:
        raise ValueError('curve_context_seconds must be positive')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = {"n_clusters": n_clusters, "random_state": random_state,
              "correlation_threshold": correlation_threshold, "progress_points": progress_points,
              "duration_mad_threshold": duration_mad_threshold, "min_trials_after_filter": min_trials_after_filter,
              "pre_post_window": pre_post_window, "baseline_window": baseline_window,
              "baseline_relative_floor": baseline_relative_floor, "baseline_edge_seconds": baseline_edge_seconds,
              "smoothing_sigma_seconds": smoothing_sigma_seconds, "slope_method": slope_method,
              "max_baseline_invalid_fraction": max_baseline_invalid_fraction,
              "round": 1, "display_sigma_seconds": display_sigma_seconds,
              "display_audio_normalization": "Each block envelope divided by its speech peak; paired valid trials per channel",
              "baseline_mode": manifest.baseline_mode.iloc[0], "picks": "ecog",
              "weighting": "equal block/channel sample; no stability evaluation",
              "selected_blocks": manifest.to_dict("records")}
    config.update(classification_mode='curve_shape',curve_context_seconds=float(context),
                  curve_context_points=curve_context_points,save_channel_figures=save_channel_figures,
                  beta_amplitude_normalization='per-sample full pre/speech/post RMS; shared training scale applied afterwards',
                  curve_segment_budgets={'pre':.2,'speech':.6,'post':.2,'audio_relation':.25})
    config.update(curve_source='continuous Hilbert beta power; smooth before any trial sampling',
                  display_beta_units='dimensionless per-channel centered/RMS shape; no baseline Z',
                  time_alignment='speech duration normalized; pre/post real seconds')
    (output / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest.to_csv(output / "input_manifest.csv", index=False, encoding="utf-8-sig")
    tables, curve_parts, reports, durations = [], [], [], []
    try:
        for position, row in enumerate(manifest.to_dict("records"), 1):
            subject, block = row["subject_id"], row["block_id"]
            print(f"[{position}/{len(manifest)}] {subject}/{block}", flush=True)
            features, epochs = build_beta_progress_feature_table_from_files(
                row["fif_path"], row["onset_path"], row["wav_path"], offset_source=row["offset_path"],
                picks="ecog", progress_points=progress_points, duration_mad_threshold=duration_mad_threshold,
                min_trials_after_filter=min_trials_after_filter, pre_post_window=context,
                baseline_mode=row["baseline_mode"], baseline_fif=row["baseline_fif"] if row["baseline_mode"] == "file" else None,
                baseline_window=baseline_window, baseline_relative_floor=baseline_relative_floor,
                baseline_edge_seconds=baseline_edge_seconds, smoothing_sigma_seconds=smoothing_sigma_seconds,
                slope_method=slope_method, max_baseline_invalid_fraction=max_baseline_invalid_fraction)
            features = features.copy()
            features.insert(0, "subject_id", subject)
            features.insert(1, "block_id", block)
            features.insert(2, "sample_id", subject + ":" + block + ":" + features.channel.astype(str))
            features["n_block_trials"] = len(epochs.events)
            reasons = []
            for sample in features.itertuples():
                reason = []
                if sample.channel in epochs.bad_channels:
                    reason.append("raw_bad")
                if sample.baseline_invalid_fraction > max_baseline_invalid_fraction:
                    reason.append("invalid_trial_fraction")
                if sample.valid_trial_count < 2:
                    reason.append("insufficient_valid_trials")
                reasons.append(";".join(reason) or ("quality_excluded" if sample.is_bad else ""))
            features["exclusion_reason"] = reasons
            block_output = output / "blocks" / subject / block
            block_output.mkdir(parents=True)
            records = {"electrode_features": features, "valid_trials": epochs.events,
                       "dropped_trials": epochs.dropped_events, "baseline_quality": epochs.baseline_diagnostics,
                       "baseline_reference": epochs.baseline_reference, "baseline_segments_report": epochs.baseline_segments}
            for name, record in records.items():
                if record is not None:
                    record.to_csv(block_output / (name + ".csv"), index=False, encoding="utf-8-sig")
            means = summarize_native_files(row['fif_path'],row['wav_path'],epochs,
                                           context_seconds=context,context_points=curve_context_points,
                                           sigma_seconds=display_sigma_seconds)
            features['curve_valid_trial_count']=means['curve_valid_trial_count']
            insufficient=features.curve_valid_trial_count<2
            features.loc[insufficient,'is_bad']=True
            features.loc[insufficient,'exclusion_reason']=features.loc[insufficient,'exclusion_reason'].apply(
                lambda reason:(reason+';' if reason else '')+'insufficient_native_curve_trials')
            curve_features=build_curve_features(means)
            invalid_curve=~np.isfinite(curve_features.filter(like='curve__').to_numpy()).all(axis=1)
            features.loc[invalid_curve,'is_bad']=True
            features.loc[invalid_curve,'exclusion_reason']=features.loc[invalid_curve,'exclusion_reason'].apply(
                lambda reason: (reason+';' if reason else '')+'invalid_curve_context')
            features=pd.concat([features.reset_index(drop=True),curve_features],axis=1)
            if tuple(features.channel) != tuple(epochs.channel_names):
                raise ValueError("Feature and curve channel order mismatch")
            curve_parts.append(means)
            durations.extend(epochs.events.duration.to_list())
            tables.append(features)
            reports.append({"subject_id": subject, "block_id": block, "n_channels": len(features),
                            "n_included_channels": int((~features.is_bad).sum()),
                            "n_valid_trials": len(epochs.events), "n_dropped_trials": len(epochs.dropped_events)})
            del epochs
        merged = pd.concat(tables, ignore_index=True)
        merged.to_csv(output / "all_channel_features.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(reports).to_csv(output / "block_summary.csv", index=False, encoding="utf-8-sig")
        clustering, assignments = fit_global(merged, n_clusters=n_clusters, random_state=random_state,
                                             correlation_threshold=correlation_threshold)
        axes={'pre_times','post_times','progress_percent'}
        cache = {key: np.concatenate([part[key] for part in curve_parts]) for key in curve_parts[0] if key not in axes}
        cache.update({key:curve_parts[0][key] for key in axes})
        cache['display_speech_seconds']=np.asarray(np.median(durations))
        cache.update(sample_id=merged.sample_id.to_numpy(dtype=str), progress_percent=np.linspace(0,100,progress_points))
        save_fitted_round(merged, clustering, assignments, cache, output, n_clusters=n_clusters, random_state=random_state)
        if save_channel_figures:
            save_complete_plots(assignments,cache,output,save_channels=True)
        (output / "run_status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
        return GlobalAnalysisResult(merged, assignments, clustering, output)
    except Exception as error:
        (output / "run_status.json").write_text(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), encoding="utf-8")
        raise
