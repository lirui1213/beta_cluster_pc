"""Persist extraction and fitted rounds without overwriting earlier reviews."""
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from .review import pca_review_table
from .plots import save_round_plots, summarize_relationship
from signal_processing import finite_mean


def ensure_extraction_config(current, extracted):
    """Reject stale notebook data before exporting it as a new configuration."""
    current = json.loads(json.dumps(current))
    extracted = json.loads(json.dumps(extracted))
    changed = [key for key in sorted(set(current) | set(extracted)) if current.get(key) != extracted.get(key)]
    if changed:
        raise ValueError("Extraction settings changed; rerun extraction and round one: " + ", ".join(changed))


def ensure_first_round(first_extraction, current_extraction):
    if first_extraction is not current_extraction:
        raise ValueError("New extraction requires rerunning round one before PCA review and round two.")


def save_extraction(extraction, output_dir, config):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    extraction.features.to_csv(output / "fourband_electrode_features.csv", index=False, encoding="utf-8-sig")
    for band, epochs in extraction.epochs.items():
        folder = output / "bands" / band
        folder.mkdir(parents=True)
        for name, table in (("baseline_quality", epochs.baseline_diagnostics),
                            ("baseline_reference", epochs.baseline_reference),
                            ("baseline_segments_report", epochs.baseline_segments),
                            ("valid_trials", epochs.events), ("dropped_trials", epochs.dropped_events)):
            if table is not None:
                table.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")
    return output


def save_round(extraction, clustering, output_dir, *, n_clusters, audit=None,
               sigma_seconds=.10, display_sfreq=100, slope_method="savgol", round_config=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(round_config or {})
    config.update(n_clusters=int(n_clusters), display_sigma_seconds=float(sigma_seconds),
                  display_sfreq=float(display_sfreq), stability_evaluated=False)
    (output / "round_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        if tuple(extraction.features.channel) != clustering.channels:
            raise ValueError("Extraction must align with clustering before export.")
        assignments = clustering.to_frame()
        reasons = extraction.features.get("exclusion_reason", pd.Series("", index=extraction.features.index)).to_list()
        if audit is not None:
            if tuple(audit.channel) != clustering.channels:
                raise ValueError("Review audit must align with electrodes.")
            audit.to_csv(output / "review_audit.csv", index=False, encoding="utf-8-sig")
            for i, row in enumerate(audit.itertuples()):
                extra = [name for name in ("excluded_cluster", "excluded_manual", "excluded_distance") if getattr(row, name)]
                reasons[i] = ";".join(filter(None, [reasons[i], *extra]))
        assignments["exclusion_reason"] = reasons
        assignments.to_csv(output / "cluster_assignments.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"feature": clustering.feature_columns, "weight": clustering.feature_weights}).to_csv(
            output / "selected_feature_weights.csv", index=False, encoding="utf-8-sig")
        pca_review_table(clustering).to_csv(output / "pca_distances.csv", index=False, encoding="utf-8-sig")
        coupling = [c for c in extraction.features if "__" in c and any(t in c for t in ("audio", "slope_product", "same_sign"))]
        names = {}
        for column in coupling:
            band, suffix = column.split("__", 1)
            method = "feature_trialwise" if "trialwise" in suffix or suffix.startswith("duration_") else "feature_mean_curve"
            names[column] = f"{band}__{method}__{suffix}"
        relationships = pd.concat([assignments[["channel", "label"]],
            extraction.features[coupling].rename(columns=names).reset_index(drop=True)], axis=1)
        for band, epochs in extraction.epochs.items():
            if epochs.channel_names != clustering.channels:
                raise ValueError(f"{band} relationship channels must align with clustering.")
            paired = summarize_relationship(epochs, sigma_seconds=sigma_seconds, slope_method=slope_method)
            relationships[f"{band}__display_paired_trial_slope_product_mean"] = finite_mean(paired["interaction"], axis=-1)
            relationships[f"{band}__display_paired_trial_count"] = paired["valid_trial_count"]
        relationships.to_csv(output / "electrode_audio_relationship.csv", index=False, encoding="utf-8-sig")
        relationships.loc[relationships.label >= 0].drop(columns="channel").groupby("label").mean().to_csv(
            output / "cluster_audio_relationship.csv", encoding="utf-8-sig")
        (output / "relationship_methods.json").write_text(json.dumps(dict(
            feature_mean_curve="Derivatives/products of trial-average curves at extraction smoothing; audio retains extraction units.",
            feature_trialwise="Trialwise extraction descriptors; see original feature names.",
            display_paired_trial="Mean product of matched within-trial derivatives at display smoothing; audio divided by global retained-speech peak.",
            slope_axis="Speech percentage point, not seconds.", display_sigma_seconds=sigma_seconds,
            display_slope_method=slope_method), indent=2), encoding="utf-8")
        save_round_plots(extraction.epochs, clustering, output, sigma_seconds=sigma_seconds,
                         display_sfreq=display_sfreq, slope_method=slope_method)
        labels, counts = np.unique(clustering.labels[clustering.clean_mask], return_counts=True)
        silhouette = None
        if 1 < len(labels) < int(clustering.clean_mask.sum()):
            silhouette = float(silhouette_score(clustering.weighted_clean_features, clustering.labels[clustering.clean_mask]))
        summary = dict(n_electrodes=len(assignments), n_included=int(clustering.clean_mask.sum()),
            n_excluded=int((~clustering.clean_mask).sum()), requested_k=int(n_clusters), actual_k=len(labels),
            cluster_sizes={str(int(k)): int(n) for k, n in zip(labels, counts, strict=True)},
            silhouette=silhouette, stability_evaluated=False, bands=extraction.bands,
            band_weight_budgets={band: float(sum(w for c, w in zip(clustering.feature_columns, clustering.feature_weights)
                if c.startswith(band + "__"))) for band in extraction.bands})
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        joblib.dump(clustering, output / "joint_model.joblib")
        (output / "status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
        if len(labels) != n_clusters:
            warnings.warn("Fewer distinct clusters than requested; inspect summary.json.", RuntimeWarning)
        return assignments
    except Exception as error:
        (output / "status.json").write_text(json.dumps({"status": "failed", "error": str(error)}), encoding="utf-8")
        raise
