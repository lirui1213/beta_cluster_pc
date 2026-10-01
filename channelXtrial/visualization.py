"""Visualizations for channel x trial beta Type 1/Type 2 analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray

if TYPE_CHECKING:
    from audio_cross_3band_utils import BetaProgressEpochs


FloatArray = NDArray[np.float64]
TYPE_COLORS: dict[str, str] = {
    "Bad": "white",
    "Uncertain": "lightgray",
    "Type1": "#1f77b4",
    "Type2": "#ff7f0e",
}
TYPE_CODES: dict[str, int] = {"Bad": -1, "Uncertain": 0, "Type1": 1, "Type2": 2}


def _sorted_channels(
    classified_table: pd.DataFrame,
    stability_table: pd.DataFrame | None,
) -> list[str]:
    if stability_table is not None and not stability_table.empty:
        return stability_table["channel"].astype(str).tolist()
    summary = (
        classified_table.groupby("channel")["trial_type"]
        .agg(lambda values: values.value_counts().index[0])
        .reset_index(name="dominant_type")
    )
    return summary.sort_values(["dominant_type", "channel"])["channel"].astype(str).tolist()


def plot_channel_trial_heatmap(
    classified_table: pd.DataFrame,
    *,
    stability_table: pd.DataFrame | None = None,
    figure: Figure | None = None,
) -> tuple[Figure, Axes]:
    """Plot channel x valid-trial Type labels."""

    required = {"channel", "trial_index", "trial_type"}
    missing = sorted(required - set(classified_table.columns))
    if missing:
        raise ValueError(f"Missing required classified columns: {missing}")
    table = classified_table.copy()
    table["channel"] = table["channel"].astype(str)
    table["type_code"] = table["trial_type"].map(TYPE_CODES).fillna(0).astype(int)
    channels = _sorted_channels(table, stability_table)
    trial_indices = sorted(table["trial_index"].astype(int).unique().tolist())
    matrix = (
        table.pivot_table(
            index="channel",
            columns="trial_index",
            values="type_code",
            aggfunc="first",
        )
        .reindex(index=channels, columns=trial_indices)
        .fillna(0)
        .to_numpy(dtype=float)
    )
    if figure is None:
        height = max(4.0, 0.13 * len(channels))
        figure, axis = plt.subplots(figsize=(12, height), constrained_layout=True)
    else:
        figure.clear()
        axis = figure.add_subplot(111)

    cmap = ListedColormap(
        [TYPE_COLORS["Bad"], TYPE_COLORS["Uncertain"], TYPE_COLORS["Type1"], TYPE_COLORS["Type2"]]
    )
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    axis.set_xlabel("Valid trial index in block")
    axis.set_ylabel("Channel")
    axis.set_title("Channel x Trial beta pattern type")
    axis.set_xticks(np.arange(len(trial_indices)))
    axis.set_xticklabels(trial_indices, rotation=90, fontsize=7)
    if len(channels) <= 40:
        axis.set_yticks(np.arange(len(channels)))
        axis.set_yticklabels(channels, fontsize=7)
    else:
        tick_count = min(32, len(channels))
        ticks = np.linspace(0, len(channels) - 1, tick_count, dtype=int)
        axis.set_yticks(ticks)
        axis.set_yticklabels([channels[index] for index in ticks], fontsize=7)
    axis.legend(
        handles=[
            Patch(facecolor=TYPE_COLORS["Type1"], label="Type 1"),
            Patch(facecolor=TYPE_COLORS["Type2"], label="Type 2"),
            Patch(facecolor=TYPE_COLORS["Uncertain"], label="Uncertain"),
            Patch(facecolor=TYPE_COLORS["Bad"], edgecolor="black", label="Bad"),
        ],
        loc="upper right",
        framealpha=0.95,
    )
    return figure, axis


def plot_channel_type_sequence(
    classified_table: pd.DataFrame,
    channel: str,
    *,
    figure: Figure | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot trial-wise type, scores, confidence, and morphology for one channel."""

    subset = classified_table[classified_table["channel"].astype(str) == str(channel)].copy()
    if subset.empty:
        raise ValueError(f"Unknown channel: {channel}")
    subset = subset.sort_values("trial_index")
    x = subset["trial_index"].to_numpy(dtype=int)
    if figure is None:
        figure, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    else:
        figure.clear()
        axes = np.asarray([figure.add_subplot(4, 1, index + 1) for index in range(4)])

    colors = [TYPE_COLORS.get(value, "gray") for value in subset["trial_type"].astype(str)]
    axes[0].scatter(x, subset["type_code"], c=colors, s=42, edgecolor="black", linewidth=0.4)
    axes[0].set_yticks([-1, 0, 1, 2])
    axes[0].set_yticklabels(["Bad", "Uncertain", "Type1", "Type2"])
    axes[0].set_ylabel("Trial type")
    axes[0].set_title(f"{channel}: trial-wise beta pattern")

    axes[1].plot(x, subset["type1_score"], color=TYPE_COLORS["Type1"], label="Type1 score")
    axes[1].plot(x, subset["type2_score"], color=TYPE_COLORS["Type2"], label="Type2 score")
    axes[1].axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    axes[1].set_ylabel("Score")
    axes[1].legend(loc="best")

    axes[2].plot(x, subset["confidence"], color="black", label="Confidence")
    axes[2].set_ylabel("Confidence")
    axes[2].legend(loc="best")

    axes[3].plot(x, subset["mid_speech_rebound"], color=TYPE_COLORS["Type1"], label="Mid-speech rebound")
    axes[3].plot(
        x,
        subset["speech_low_maintenance"],
        color=TYPE_COLORS["Type2"],
        label="Low maintenance",
    )
    axes[3].set_ylabel("Morphology")
    axes[3].set_xlabel("Valid trial index in block")
    axes[3].legend(loc="best")
    return figure, np.asarray(axes, dtype=object)


def plot_block_type_proportions(
    classified_table: pd.DataFrame,
    *,
    figure: Figure | None = None,
) -> tuple[Figure, Axes]:
    """Plot Type 1, Type 2, and uncertain channel proportions across trials."""

    table = classified_table[~classified_table["trial_type"].eq("Bad")].copy()
    counts = (
        table.groupby(["trial_index", "trial_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Type1", "Type2", "Uncertain"], fill_value=0)
        .sort_index()
    )
    proportions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    x = proportions.index.to_numpy(dtype=int)
    if figure is None:
        figure, axis = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
    else:
        figure.clear()
        axis = figure.add_subplot(111)
    axis.stackplot(
        x,
        proportions["Type1"],
        proportions["Type2"],
        proportions["Uncertain"],
        colors=[TYPE_COLORS["Type1"], TYPE_COLORS["Type2"], TYPE_COLORS["Uncertain"]],
        labels=["Type 1", "Type 2", "Uncertain"],
        alpha=0.85,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Valid trial index in block")
    axis.set_ylabel("Channel proportion")
    axis.set_title("Block-level beta pattern proportions")
    axis.legend(loc="upper right")
    return figure, axis


def _concat_trial_curve(
    epochs: "BetaProgressEpochs",
    trial_index: int,
    channel_index: int,
) -> tuple[FloatArray, FloatArray]:
    duration = float(epochs.events.iloc[trial_index]["duration"])
    speech_seconds = epochs.progress_percent / 100.0 * duration
    x_axis = np.concatenate(
        [
            epochs.pre_times[:-1],
            speech_seconds,
            duration + epochs.post_times[1:],
        ]
    )
    values = np.concatenate(
        [
            epochs.pre[trial_index, channel_index, :-1],
            epochs.speech[trial_index, channel_index],
            epochs.post[trial_index, channel_index, 1:],
        ]
    )
    return x_axis * 1000.0, values


def plot_channel_trial_curves(
    epochs: "BetaProgressEpochs",
    classified_table: pd.DataFrame,
    channel: str,
    *,
    max_trials: int = 24,
    trial_indices: Sequence[int] | None = None,
    figure: Figure | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Show single-trial beta curves for one channel, colored by Type."""

    if channel not in epochs.channel_names:
        raise ValueError(f"Unknown channel: {channel}")
    channel_index = epochs.channel_names.index(channel)
    subset = classified_table[classified_table["channel"].astype(str) == str(channel)].copy()
    if trial_indices is not None:
        wanted = [int(value) for value in trial_indices]
        subset = subset[subset["trial_index"].isin(wanted)]
    subset = subset.sort_values("trial_index").head(max_trials)
    if subset.empty:
        raise ValueError(f"No classified trials are available for channel: {channel}")

    column_count = min(6, len(subset))
    row_count = int(np.ceil(len(subset) / column_count))
    if figure is None:
        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(3.1 * column_count, 2.4 * row_count),
            sharey=True,
            constrained_layout=True,
            squeeze=False,
        )
    else:
        figure.clear()
        axes = np.asarray(
            [
                [
                    figure.add_subplot(row_count, column_count, row * column_count + column + 1)
                    for column in range(column_count)
                ]
                for row in range(row_count)
            ],
            dtype=object,
        )

    flat_axes = axes.ravel()
    for axis, (_, row) in zip(flat_axes, subset.iterrows(), strict=False):
        trial_index = int(row["trial_index"])
        x_ms, values = _concat_trial_curve(epochs, trial_index, channel_index)
        duration_ms = float(epochs.events.iloc[trial_index]["duration"]) * 1000.0
        trial_type = str(row["trial_type"])
        axis.plot(x_ms, values, color=TYPE_COLORS.get(trial_type, "gray"), linewidth=1.4)
        axis.axvline(0.0, color="black", linestyle="--", linewidth=0.7)
        axis.axvline(duration_ms, color="black", linestyle="--", linewidth=0.7)
        axis.axhline(0.0, color="black", linewidth=0.5, alpha=0.35)
        axis.set_title(f"Trial {trial_index}: {trial_type}", fontsize=9)
        axis.set_xlabel("ms")
    for unused in flat_axes[len(subset):]:
        unused.set_visible(False)
    axes[0, 0].set_ylabel("Beta Z")
    figure.suptitle(f"{channel}: single-trial beta curves", fontsize=12)
    return figure, axes


__all__ = [
    "TYPE_CODES",
    "TYPE_COLORS",
    "plot_block_type_proportions",
    "plot_channel_trial_curves",
    "plot_channel_trial_heatmap",
    "plot_channel_type_sequence",
]
