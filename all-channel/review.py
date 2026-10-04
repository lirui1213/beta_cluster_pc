"""Explicit first-round exclusions followed by independently refitted clustering."""
from pathlib import Path
import json
import numpy as np

from pipeline import fit_global, GlobalAnalysisResult, save_fitted_round


def pca_review_table(features, first):
    from auto_prune import review_coordinates
    if tuple(features.sample_id.astype(str)) != tuple(first.channels):
        raise ValueError("Features must align with the first-round sample order")
    audit = features[[c for c in ("sample_id", "subject_id", "block_id", "channel", "is_bad", "exclusion_reason") if c in features]].copy().reset_index(drop=True)
    audit["first_label"] = first.labels
    for i in range(3):
        audit[f"first_pc{i + 1}"] = first.pca_scores[:, i]
    distance = np.full(len(features), np.nan)
    coordinates = review_coordinates(first)
    for label in np.unique(first.labels):
        if label < 0:
            continue
        mask = first.labels == label
        points = coordinates[mask, :3]
        center = points.mean(axis=0)
        distance[mask] = np.linalg.norm(points - center, axis=1)
    audit["pca_cluster_distance"] = distance
    return audit


def refit_reviewed(features, first, *, second_n_clusters=2, bad_clusters=(), manual_bad_samples=(),
                   pca_distance_threshold=None, random_state=50, correlation_threshold=.95, auto_bad_clusters=()):
    audit = pca_review_table(features, first)
    unknown_clusters = (set(bad_clusters) | set(auto_bad_clusters)) - (set(first.labels) - {-1})
    unknown_samples = set(manual_bad_samples) - set(features.sample_id)
    if unknown_clusters or unknown_samples:
        raise ValueError(f"Unknown review selection: clusters={unknown_clusters}, sample_ids={unknown_samples}")
    if pca_distance_threshold is not None and (not np.isfinite(pca_distance_threshold) or pca_distance_threshold <= 0):
        raise ValueError("pca_distance_threshold must be positive or None")
    audit["excluded_cluster"] = np.isin(first.labels, list(bad_clusters))
    audit["excluded_auto_cluster"] = np.isin(first.labels, list(auto_bad_clusters))
    audit["excluded_manual"] = audit.sample_id.isin(manual_bad_samples)
    audit["excluded_distance"] = False if pca_distance_threshold is None else audit.pca_cluster_distance > pca_distance_threshold
    # Original quality mask and all review decisions are additive.
    combined = features.is_bad.to_numpy(bool) | (first.labels < 0) | audit[["excluded_auto_cluster", "excluded_cluster", "excluded_manual", "excluded_distance"]].any(axis=1).to_numpy()
    reviewed = features.copy().reset_index(drop=True)
    reviewed["is_bad"] = combined
    previous = reviewed.get("exclusion_reason", np.full(len(reviewed), ""))
    reasons = []
    for index, old in enumerate(previous):
        parts = [str(old)] if isinstance(old, str) and old else []
        for name, flag in [("auto_small_high_variance_cluster", "excluded_auto_cluster"), ("review_cluster", "excluded_cluster"), ("manual_sample", "excluded_manual"), ("pca_distance", "excluded_distance")]:
            if audit[flag].iloc[index]:
                parts.append(name)
        if combined[index] and not parts:
            parts.append("quality_excluded")
        reasons.append(";".join(parts))
    reviewed["exclusion_reason"] = reasons
    result, assignments = fit_global(reviewed, n_clusters=second_n_clusters, random_state=random_state,
                                     correlation_threshold=correlation_threshold)
    audit["excluded_final"] = combined
    audit["second_label"] = result.labels
    return reviewed, result, assignments, audit


def run_second_round(first_analysis, output_dir, *, second_n_clusters=2, bad_clusters=(),
                     manual_bad_samples=(), pca_distance_threshold=None, random_state=50, correlation_threshold=.95,
                     automatic_pruning=None):
    """Reuse extracted data; second-round changes never overwrite first outputs."""
    auto_clusters = ()
    if automatic_pruning is not None:
        if tuple(automatic_pruning.sample_report.sample_id) != tuple(first_analysis.features.sample_id):
            raise ValueError("Automatic screening must align with first-round sample_ids")
        if not np.array_equal(automatic_pruning.sample_report.first_label.to_numpy(), first_analysis.clustering.labels):
            raise ValueError("Automatic screening must come from the current first-round clustering")
        auto_clusters = automatic_pruning.bad_clusters
    reviewed, clustering, assignments, audit = refit_reviewed(
        first_analysis.features, first_analysis.clustering, second_n_clusters=second_n_clusters,
        bad_clusters=bad_clusters, manual_bad_samples=manual_bad_samples,
        pca_distance_threshold=pca_distance_threshold, random_state=random_state,
        correlation_threshold=correlation_threshold, auto_bad_clusters=auto_clusters)
    with np.load(first_analysis.output_dir / "mean_speech_curves.npz") as source:
        cache = {key: source[key] for key in source.files}
    if tuple(cache["sample_id"]) != tuple(reviewed.sample_id):
        raise ValueError("Cached curves must align with feature sample_ids")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = {"round": 2, "first_round_dir": str(first_analysis.output_dir.resolve()),
              "second_n_clusters": second_n_clusters, "bad_clusters": [int(x) for x in bad_clusters],
              "manual_bad_samples": list(manual_bad_samples), "pca_distance_threshold": pca_distance_threshold,
              "pca_distance_rule": "Euclidean distance to assigned cluster centroid in first 3 first-round PCs",
              "random_state": random_state, "correlation_threshold": correlation_threshold}
    config["automatic_pruning"] = automatic_pruning.policy if automatic_pruning is not None else {"enabled": False}
    if any(c.startswith('curve__') for c in clustering.feature_columns):
        config.update(classification_mode='curve_shape',distance='weighted Euclidean',
                      beta_centering='subtract each sample pre-onset mean; divide full curve by its own RMS',
                      shared_beta_scale=float(clustering.scaler.beta_scale_))
    (output / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        audit.to_csv(output / "review_audit.csv", index=False, encoding="utf-8-sig")
        if automatic_pruning is not None:
            automatic_pruning.cluster_report.to_csv(output / "auto_cluster_screening.csv", index=False, encoding="utf-8-sig")
            automatic_pruning.sample_report.to_csv(output / "auto_sample_screening.csv", index=False, encoding="utf-8-sig")
        reviewed.to_csv(output / "all_channel_features.csv", index=False, encoding="utf-8-sig")
        save_fitted_round(reviewed, clustering, assignments, cache, output, n_clusters=second_n_clusters, random_state=random_state)
        (output / "run_status.json").write_text(json.dumps({"status": "complete", "round": 2}), encoding="utf-8")
    except Exception as error:
        (output / "run_status.json").write_text(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), encoding="utf-8")
        raise
    return GlobalAnalysisResult(reviewed, assignments, clustering, output)
