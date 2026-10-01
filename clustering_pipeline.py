"""Leakage-safe first- and second-round K-means clustering for ECoG features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ClusteringResult:
    """A fitted clustering result aligned to every input electrode row."""

    labels: IntArray
    pca_scores: FloatArray
    clean_mask: NDArray[np.bool_]
    channels: tuple[str, ...]
    feature_columns: tuple[str, ...]
    imputer: SimpleImputer
    scaler: StandardScaler
    kmeans: KMeans
    pca: PCA | None
    standardized_clean_features: FloatArray
    feature_weights: tuple[float, ...]
    weighted_clean_features: FloatArray

    def to_frame(self) -> pd.DataFrame:
        """Return channels, bad status, labels, and three plotting PCs."""

        return pd.DataFrame(
            {
                "channel": self.channels,
                "is_bad": ~self.clean_mask,
                "label": self.labels,
                "pc1": self.pca_scores[:, 0],
                "pc2": self.pca_scores[:, 1],
                "pc3": self.pca_scores[:, 2],
            }
        )


@dataclass(frozen=True)
class TwoRoundClusteringResult:
    """First result, reviewed bad mask, and independently refitted second result."""

    first_round: ClusteringResult
    combined_bad_mask: NDArray[np.bool_]
    second_round: ClusteringResult


def infer_feature_columns(
    feature_table: pd.DataFrame,
    *,
    excluded: Iterable[str] = ("channel", "is_bad", "label", "pc1", "pc2", "pc3"),
) -> tuple[str, ...]:
    """Select numeric feature columns while excluding metadata columns."""

    excluded_set = set(excluded)
    columns = tuple(
        str(column)
        for column in feature_table.columns
        if column not in excluded_set and pd.api.types.is_numeric_dtype(feature_table[column])
    )
    if not columns:
        raise ValueError("No numeric feature columns were found for clustering.")
    return columns


def _validate_bad_mask(mask: NDArray[np.bool_], row_count: int) -> NDArray[np.bool_]:
    result = np.asarray(mask, dtype=bool)
    if result.shape != (row_count,):
        raise ValueError(f"bad_mask must have shape ({row_count},), got {result.shape}.")
    return result


def _fit_pca_for_plotting(features: FloatArray, clean_mask: NDArray[np.bool_]) -> tuple[FloatArray, PCA | None]:
    scores = np.full((len(clean_mask), 3), np.nan, dtype=float)
    component_count = min(3, features.shape[0], features.shape[1])
    if component_count == 0:
        return scores, None
    pca = PCA(n_components=component_count)
    transformed = pca.fit_transform(features)
    scores[clean_mask, :component_count] = transformed
    return scores, pca


def _resolve_feature_weights(
    feature_weights: Mapping[str, float] | Sequence[float] | None,
    columns: Sequence[str],
) -> FloatArray:
    if feature_weights is None:
        return np.ones(len(columns), dtype=float)
    if isinstance(feature_weights, Mapping):
        weights = np.asarray([feature_weights.get(column, 1.0) for column in columns], dtype=float)
    else:
        weights = np.asarray(feature_weights, dtype=float)
        if weights.shape != (len(columns),):
            raise ValueError(
                f"feature_weights must have shape ({len(columns)},), got {weights.shape}."
            )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("feature_weights must be finite non-negative values.")
    if not np.any(weights > 0):
        raise ValueError("At least one feature weight must be positive.")
    return weights


def default_beta_progress_feature_weights(
    feature_columns: Sequence[str],
    *,
    slope_audio_weight: float = 3.0,
    beta_slope_weight: float = 1.8,
    energy_weight: float = 0.35,
    default_weight: float = 1.0,
) -> dict[str, float]:
    """Return interpretable default weights for beta-progress features.

    Slope/audio interaction columns are emphasized, beta-slope-only columns are
    mildly emphasized, and energy-level columns are down-weighted.
    """

    energy_tokens = (
        "energy_mean",
        "energy_std",
        "energy_auc",
        "energy_peak",
        "energy_range",
        "pre_mean",
        "post_mean",
        "post_minus_pre_mean",
    )
    interaction_tokens = (
        "audio_slope",
        "slope_product",
        "slope_same_sign",
        "slope_crossing",
        "trialwise_slope_audio",
        "duration_slope_audio",
    )
    weights: dict[str, float] = {}
    for column in feature_columns:
        if any(token in column for token in interaction_tokens):
            weights[str(column)] = slope_audio_weight
        elif "beta_slope" in column:
            weights[str(column)] = beta_slope_weight
        elif any(token in column for token in energy_tokens):
            weights[str(column)] = energy_weight
        else:
            weights[str(column)] = default_weight
    return weights


def cluster_electrodes(
    feature_table: pd.DataFrame,
    *,
    n_clusters: int = 5,
    bad_mask: NDArray[np.bool_] | None = None,
    feature_columns: Sequence[str] | None = None,
    feature_weights: Mapping[str, float] | Sequence[float] | None = None,
    random_state: int = 50,
) -> ClusteringResult:
    """Impute, scale, cluster, and compute plotting PCA using clean rows only.

    Every estimator is newly constructed and fitted inside this function. This
    property is important for the second review round: passing an updated bad
    mask cannot reuse statistics learned from newly excluded electrodes.
    """

    row_count = len(feature_table)
    if row_count == 0:
        raise ValueError("feature_table must contain at least one electrode.")
    if "channel" in feature_table:
        channels = tuple(feature_table["channel"].astype(str))
    else:
        channels = tuple(str(index) for index in feature_table.index)

    initial_bad = (
        feature_table["is_bad"].fillna(False).to_numpy(dtype=bool)
        if "is_bad" in feature_table
        else np.zeros(row_count, dtype=bool)
    )
    supplied_bad = (
        np.zeros(row_count, dtype=bool)
        if bad_mask is None
        else _validate_bad_mask(bad_mask, row_count)
    )
    clean_mask = ~(initial_bad | supplied_bad)
    clean_count = int(np.count_nonzero(clean_mask))
    if not 2 <= n_clusters <= clean_count:
        raise ValueError(
            f"n_clusters must be between 2 and the {clean_count} clean electrodes; "
            f"got {n_clusters}."
        )

    requested_columns = tuple(feature_columns or infer_feature_columns(feature_table))
    missing = sorted(set(requested_columns) - set(feature_table.columns))
    if missing:
        raise ValueError(f"Unknown feature columns: {missing}")
    raw_clean = feature_table.loc[clean_mask, list(requested_columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    usable = ~np.all(np.isnan(raw_clean), axis=0)
    if not np.any(usable):
        raise ValueError("Every requested feature is missing among clean electrodes.")
    if not np.all(usable):
        all_missing = [
            requested_columns[index] for index in np.flatnonzero(~usable)
        ]
        warnings.warn(
            "Dropping features that are entirely missing among clean electrodes: "
            f"{all_missing}",
            RuntimeWarning,
            stacklevel=2,
        )
        raw_clean = raw_clean[:, usable]
    columns = tuple(
        column for column, keep in zip(requested_columns, usable, strict=True) if keep
    )
    weights = _resolve_feature_weights(feature_weights, columns)

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(raw_clean)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(imputed)
    weighted = standardized * np.sqrt(weights)[None, :]
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=50,
        random_state=random_state,
        algorithm="lloyd",
        max_iter=300,
        tol=1e-4,
    )
    clean_labels = kmeans.fit_predict(weighted).astype(np.int64)
    labels = np.full(row_count, -1, dtype=np.int64)
    labels[clean_mask] = clean_labels
    pca_scores, pca = _fit_pca_for_plotting(weighted, clean_mask)

    if not np.all(labels[~clean_mask] == -1):
        raise AssertionError("All excluded electrodes must retain label -1.")
    if not np.all(np.isnan(pca_scores[~clean_mask])):
        raise AssertionError("All excluded electrodes must have NaN PCA coordinates.")
    return ClusteringResult(
        labels=labels,
        pca_scores=pca_scores,
        clean_mask=clean_mask,
        channels=channels,
        feature_columns=columns,
        imputer=imputer,
        scaler=scaler,
        kmeans=kmeans,
        pca=pca,
        standardized_clean_features=standardized,
        feature_weights=tuple(float(value) for value in weights),
        weighted_clean_features=weighted,
    )


def combine_bad_channels(
    channels: Sequence[str],
    initial_bads: Iterable[str] | NDArray[np.bool_],
    first_round_labels: Sequence[int],
    *,
    bad_clusters: Iterable[int] = (),
    manual_bads: Iterable[str] = (),
) -> NDArray[np.bool_]:
    """Return ``initial | selected clusters | manual channels`` as a boolean mask."""

    channel_array = np.asarray(channels, dtype=str)
    labels = np.asarray(first_round_labels, dtype=int)
    if labels.shape != channel_array.shape:
        raise ValueError("first_round_labels must align one-to-one with channels.")

    initial_array = np.asarray(initial_bads)
    if initial_array.dtype == bool:
        initial_mask = _validate_bad_mask(initial_array, len(channel_array))
    else:
        initial_names = {str(value) for value in initial_bads}
        initial_mask = np.isin(channel_array, list(initial_names))
    cluster_values = np.asarray(list(bad_clusters), dtype=int)
    cluster_mask = np.isin(labels, cluster_values) if cluster_values.size else np.zeros_like(labels, bool)
    manual_names = {str(value) for value in manual_bads}
    unknown = sorted(manual_names - set(channel_array))
    if unknown:
        raise ValueError(f"Manual bad channels are not present in the feature table: {unknown}")
    manual_mask = np.isin(channel_array, list(manual_names))
    return initial_mask | cluster_mask | manual_mask


def refit_after_bad_channel_review(
    feature_table: pd.DataFrame,
    combined_bad_mask: NDArray[np.bool_],
    *,
    n_clusters: int,
    feature_columns: Sequence[str] | None = None,
    feature_weights: Mapping[str, float] | Sequence[float] | None = None,
    random_state: int = 50,
) -> ClusteringResult:
    """Fit a fully independent second-round pipeline after bad-channel review."""

    result = cluster_electrodes(
        feature_table,
        n_clusters=n_clusters,
        bad_mask=combined_bad_mask,
        feature_columns=feature_columns,
        feature_weights=feature_weights,
        random_state=random_state,
    )
    combined = _validate_bad_mask(combined_bad_mask, len(feature_table))
    if not np.all(result.labels[combined] == -1):
        raise AssertionError("Second-round bad electrodes must all have label -1.")
    return result


def run_two_round_clustering(
    feature_table: pd.DataFrame,
    *,
    first_n_clusters: int = 5,
    second_n_clusters: int = 5,
    bad_clusters: Iterable[int] = (),
    manual_bads: Iterable[str] = (),
    feature_columns: Sequence[str] | None = None,
    feature_weights: Mapping[str, float] | Sequence[float] | None = None,
    random_state: int = 50,
) -> TwoRoundClusteringResult:
    """Run initial clustering, merge review decisions, and independently refit."""

    if not 2 <= second_n_clusters <= 10:
        raise ValueError("Second-round n_clusters must be between 2 and 10.")
    first = cluster_electrodes(
        feature_table,
        n_clusters=first_n_clusters,
        feature_columns=feature_columns,
        feature_weights=feature_weights,
        random_state=random_state,
    )
    initial_mask = (
        feature_table["is_bad"].fillna(False).to_numpy(bool)
        if "is_bad" in feature_table
        else np.zeros(len(feature_table), dtype=bool)
    )
    combined = combine_bad_channels(
        first.channels,
        initial_mask,
        first.labels,
        bad_clusters=bad_clusters,
        manual_bads=manual_bads,
    )
    second = refit_after_bad_channel_review(
        feature_table,
        combined,
        n_clusters=second_n_clusters,
        feature_columns=first.feature_columns,
        feature_weights=dict(zip(first.feature_columns, first.feature_weights, strict=True)),
        random_state=random_state,
    )
    return TwoRoundClusteringResult(first, combined, second)


__all__ = [
    "ClusteringResult",
    "TwoRoundClusteringResult",
    "cluster_electrodes",
    "combine_bad_channels",
    "default_beta_progress_feature_weights",
    "infer_feature_columns",
    "refit_after_bad_channel_review",
    "run_two_round_clustering",
]
