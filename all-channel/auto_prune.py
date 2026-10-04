"""One-pass small AND high-PCA-variance screening before manual review."""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass
class AutomaticPruning:
    bad_clusters: tuple[int, ...]
    bad_mask: np.ndarray
    cluster_report: pd.DataFrame
    sample_report: pd.DataFrame
    policy: dict


def review_coordinates(first):
    """Pad entirely unestimated PCs with zero for eligible samples only."""
    scores = np.asarray(first.pca_scores, dtype=float).copy()
    eligible = np.asarray(first.labels) >= 0
    if scores.shape != (len(first.labels), 3) or not eligible.any():
        raise ValueError("First-round PCA must align with eligible samples")
    for component in range(3):
        column = scores[eligible, component]
        if np.isnan(column).all():
            scores[eligible, component] = 0
        elif not np.isfinite(column).all():
            raise ValueError("Partially invalid eligible PCA coordinates")
    return scores


def screen_small_high_variance_clusters(features, first, *, enabled=True, small_fraction=.05, variance_ratio=3):
    """Use first-round PCs, remove a cluster only when BOTH tests pass.

    Variance is trace(sample covariance) in the first three PCs (ddof=1).
    Size is relative to first-round quality-eligible samples. Reference is
    the median variance of non-small clusters with at least two members.
    Singletons and absent references never result in automatic exclusion.
    """
    if not np.isfinite(small_fraction) or not 0 < small_fraction <= 1:
        raise ValueError("small_fraction must lie in (0, 1]")
    if not np.isfinite(variance_ratio) or variance_ratio <= 0:
        raise ValueError("variance_ratio must be finite and positive")
    if tuple(features.sample_id.astype(str)) != tuple(first.channels):
        raise ValueError("Features must align with first-round samples")
    scores = review_coordinates(first)
    labels = np.asarray(first.labels)
    eligible = labels >= 0
    if not eligible.any() or not np.isfinite(scores[eligible]).all():
        raise ValueError("First-round eligible PCA coordinates must be finite")
    total = int(eligible.sum())
    rows = []
    for label in sorted(set(labels[eligible])):
        points = scores[labels == label, :3]
        count = len(points)
        variance = float(((points - points.mean(axis=0)) ** 2).sum() / (count - 1)) if count > 1 else np.nan
        rows.append({"label": int(label), "n_samples": count, "size_fraction": count / total,
                     "pca_variance": variance, "variance_estimable": count > 1,
                     "is_small": count / total <= small_fraction})
    report = pd.DataFrame(rows)
    references = report.loc[~report.is_small & report.variance_estimable, "pca_variance"]
    reference = float(references.median()) if len(references) else np.nan
    numerical_floor = np.finfo(float).eps * max(1., float(np.var(scores[eligible, :3], axis=0).sum()))
    threshold = max(reference * variance_ratio, numerical_floor) if np.isfinite(reference) else np.nan
    report["reference_variance"] = reference
    report["variance_threshold"] = threshold
    report["is_high_variance"] = report.pca_variance > threshold
    report["auto_excluded"] = bool(enabled) & report.is_small & report.is_high_variance
    reasons = []
    for row in report.itertuples():
        if not enabled:
            reason = "disabled"
        elif not np.isfinite(reference):
            reason = "no_reference_cluster"
        elif not row.variance_estimable:
            reason = "singleton_variance_undefined"
        elif row.auto_excluded:
            reason = "small_and_high_variance"
        elif not row.is_small:
            reason = "not_small"
        else:
            reason = "variance_not_high"
        reasons.append(reason)
    report["auto_reason"] = reasons
    bad_clusters = tuple(report.loc[report.auto_excluded, "label"].astype(int))
    mask = np.isin(labels, bad_clusters)
    metadata = [c for c in ("sample_id", "subject_id", "block_id", "channel") if c in features]
    sample_report = features[metadata].copy().reset_index(drop=True)
    sample_report["first_label"] = labels
    sample_report["auto_excluded"] = mask
    sample_report["auto_reason"] = ["small_and_high_variance" if value else "" for value in mask]
    policy = {"enabled": bool(enabled), "small_fraction": float(small_fraction),
              "variance_ratio": float(variance_ratio), "reference_variance": reference if np.isfinite(reference) else None,
              "variance_threshold": threshold if np.isfinite(threshold) else None,
              "reference_cluster_count": len(references), "eligible_sample_count": total,
              "variance_definition": "sum of sample variances in first 3 first-round PCs (ddof=1)",
              "rule": "size_fraction <= small_fraction AND pca_variance > variance_threshold",
              "passes": 1, "bad_clusters": list(bad_clusters)}
    return AutomaticPruning(bad_clusters, mask, report, sample_report, policy)


def make_popup_result(first, bad_mask):
    """Preserve PCA and label numbers; excluded points are displayed as Bad."""
    bad_mask = np.asarray(bad_mask, dtype=bool)
    if bad_mask.shape != first.labels.shape:
        raise ValueError("Automatic mask must align with first-round samples")
    labels = first.labels.copy()
    labels[bad_mask] = -1
    return replace(first, labels=labels, clean_mask=first.clean_mask & ~bad_mask,
                   pca_scores=review_coordinates(first))


def show_pruned_review_window(first, automatic, *, initial_second_k=2):
    from visualization import BadChannelReviewWindow
    view = make_popup_result(first, automatic.bad_mask)
    title = f"Global PCA review | automatically excluded: {int(automatic.bad_mask.sum())} samples"
    return BadChannelReviewWindow(view, initial_second_k=initial_second_k, title=title).show()
