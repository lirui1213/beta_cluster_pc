"""Auditable first-round PCA review and independently refitted manual-K results."""
import numpy as np
import pandas as pd

from .analysis import fit_joint


def pca_review_table(first):
    scores = np.asarray(first.pca_scores, dtype=float).copy()
    included = np.asarray(first.labels) >= 0
    if scores.shape != (len(first.channels), 3) or not included.any():
        raise ValueError("First-round PCA must align with included electrodes.")
    for i in range(3):
        if np.isnan(scores[included, i]).all():
            scores[included, i] = 0
    if not np.isfinite(scores[included]).all():
        raise ValueError("Included PCA coordinates must be finite.")
    table = pd.DataFrame({"channel": first.channels, "first_label": first.labels,
                          "pc1": scores[:, 0], "pc2": scores[:, 1], "pc3": scores[:, 2]})
    global_distance = np.full(len(included), np.nan)
    global_distance[included] = np.linalg.norm(scores[included] - np.median(scores[included], axis=0), axis=1)
    cluster_distance = np.full(len(included), np.nan)
    for label in np.unique(first.labels[included]):
        mask = first.labels == label
        cluster_distance[mask] = np.linalg.norm(scores[mask] - scores[mask].mean(axis=0), axis=1)
    table["pca_global_distance"] = global_distance
    table["pca_cluster_distance"] = cluster_distance
    return table


def refit_reviewed(features, first, *, n_clusters=4, bad_clusters=(), manual_bad_channels=(),
                   pca_distance_threshold=None, distance_reference="global", **fit_kwargs):
    if tuple(features.channel.astype(str)) != first.channels:
        raise ValueError("Feature rows do not align with first-round electrodes.")
    if distance_reference not in {"global", "cluster"}:
        raise ValueError("distance_reference must be global or cluster.")
    if pca_distance_threshold is not None and (not np.isfinite(pca_distance_threshold) or pca_distance_threshold <= 0):
        raise ValueError("PCA distance threshold must be positive or None.")
    unknown_clusters = set(bad_clusters) - set(first.labels[first.labels >= 0])
    unknown_channels = set(manual_bad_channels) - set(first.channels)
    if unknown_clusters or unknown_channels:
        raise ValueError(f"Unknown review selection: clusters={unknown_clusters}, channels={unknown_channels}")
    audit = pca_review_table(first)
    audit["excluded_initial"] = ~first.clean_mask
    audit["excluded_cluster"] = np.isin(first.labels, list(bad_clusters))
    audit["excluded_manual"] = audit.channel.isin(manual_bad_channels)
    distance_column = "pca_global_distance" if distance_reference == "global" else "pca_cluster_distance"
    audit["distance_reference"] = distance_reference
    audit["distance_threshold"] = pca_distance_threshold
    audit["excluded_distance"] = False if pca_distance_threshold is None else audit[distance_column] > pca_distance_threshold
    mask = audit[["excluded_initial", "excluded_cluster", "excluded_manual", "excluded_distance"]].any(axis=1).to_numpy()
    result = fit_joint(features, n_clusters=n_clusters, bad_mask=mask, **fit_kwargs)
    audit["excluded_final"] = ~result.clean_mask
    audit["second_label"] = result.labels
    return result, audit
