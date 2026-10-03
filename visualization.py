"""Visualization, bad-channel review UI, and FreeSurfer surface mapping."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d

if TYPE_CHECKING:
    from audio_cross_3band_utils import BandPowerEpochs, BetaProgressEpochs
    from clustering_pipeline import ClusteringResult


FloatArray = NDArray[np.float64]
BASE_CLUSTER_COLORS: tuple[str, ...] = ("red", "blue", "green", "purple", "gray")
BAD_CHANNEL_COLOR = "white"
_EXTRA_CLUSTER_COLORS: tuple[str, ...] = ("orange", "cyan", "magenta", "gold", "brown")


@dataclass(frozen=True)
class BadChannelReview:
    """Selections returned by the Tk review window."""

    bad_clusters: tuple[int, ...]
    manual_bads: tuple[str, ...]
    n_clusters: int


def gaussian_smooth_nan(
    values: ArrayLike,
    sfreq: float,
    *,
    sigma_seconds: float = 0.060,
    axis: int = -1,
) -> FloatArray:
    """Gaussian-smooth while preventing NaNs from contaminating neighboring data.

    ``sigma_seconds`` is the Gaussian standard deviation, not the full kernel
    width. The normalized-convolution result is intended only for display.
    """

    array = np.asarray(values, dtype=float)
    if sfreq <= 0 or sigma_seconds < 0:
        raise ValueError("sfreq must be positive and sigma_seconds non-negative.")
    if sigma_seconds == 0:
        return array.copy()
    sigma_samples = sigma_seconds * sfreq
    finite = np.isfinite(array)
    numerator = gaussian_filter1d(
        np.where(finite, array, 0.0), sigma=sigma_samples, axis=axis, mode="nearest"
    )
    denominator = gaussian_filter1d(
        finite.astype(float), sigma=sigma_samples, axis=axis, mode="nearest"
    )
    result = np.full_like(numerator, np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 1e-12)
    return result


def cluster_color(label: int) -> str:
    """Return a stable display color; label ``-1`` is always white."""

    if label == -1:
        return BAD_CHANNEL_COLOR
    if label < -1:
        raise ValueError(f"Cluster labels must be -1 or non-negative, got {label}.")
    palette = BASE_CLUSTER_COLORS + _EXTRA_CLUSTER_COLORS
    return palette[label % len(palette)]


def plot_pca_clusters(
    pca_scores: ArrayLike,
    labels: ArrayLike,
    *,
    channel_names: Sequence[str] | None = None,
    annotate: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot three PCA components; rows with NaN coordinates are omitted."""

    scores = np.asarray(pca_scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    if scores.ndim != 2 or scores.shape[1] != 3 or scores.shape[0] != len(label_array):
        raise ValueError("pca_scores must have shape (electrodes, 3) and align with labels.")
    if channel_names is not None and len(channel_names) != len(labels):
        raise ValueError("channel_names must align with labels.")
    if ax is None:
        figure = plt.figure(figsize=(8, 6), constrained_layout=True)
        ax = figure.add_subplot(111, projection="3d")
    else:
        figure = ax.figure

    for label in np.unique(label_array):
        mask = label_array == label
        finite = mask & np.all(np.isfinite(scores), axis=1)
        if not np.any(finite):
            continue
        ax.scatter(
            scores[finite, 0],
            scores[finite, 1],
            scores[finite, 2],
            color=cluster_color(int(label)),
            edgecolor="black",
            linewidth=0.5,
            s=42,
            label="Bad" if label == -1 else f"Cluster {label}",
        )
        if annotate and channel_names is not None:
            for index in np.flatnonzero(finite):
                ax.text(*scores[index], str(channel_names[index]), fontsize=7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(loc="best")
    return figure, ax


def _mean_sem_by_electrode(data: FloatArray) -> tuple[FloatArray, FloatArray, int]:
    """Average trials first, then return mean and SEM across electrodes."""

    electrode_curves = np.nanmean(data, axis=0)
    count = electrode_curves.shape[0]
    mean = np.nanmean(electrode_curves, axis=0)
    if count <= 1:
        sem = np.zeros_like(mean)
    else:
        sem = np.nanstd(electrode_curves, axis=0, ddof=1) / np.sqrt(count)
    return mean, sem, count


def plot_cluster_dynamics(
    epochs: "BandPowerEpochs",
    labels: ArrayLike,
    *,
    alignment: str = "onset",
    sigma_seconds: float = 0.060,
    include_bad: bool = False,
    figure: Figure | None = None,
) -> tuple[Figure, NDArray[Any]]:
    """Plot each cluster's electrode-level mean and SEM for all frequency bands.

    Smoothing is applied after summary statistics are calculated and never
    mutates the epoch arrays used for feature engineering.
    """

    label_array = np.asarray(labels, dtype=int)
    if label_array.shape != (len(epochs.channel_names),):
        raise ValueError("labels must align one-to-one with epoch channels.")
    if alignment == "onset":
        band_data = epochs.onset
        times = epochs.onset_times
    elif alignment == "offset":
        if epochs.offset_times is None or not epochs.offset:
            raise ValueError("Offset epochs were not extracted.")
        band_data = epochs.offset
        times = epochs.offset_times
    else:
        raise ValueError("alignment must be 'onset' or 'offset'.")

    bands = list(band_data)
    if figure is None:
        figure, axes = plt.subplots(
            len(bands), 1, figsize=(10, max(3.0, 2.8 * len(bands))), sharex=True,
            constrained_layout=True, squeeze=False,
        )
    else:
        figure.clear()
        axes = np.asarray(
            [figure.add_subplot(len(bands), 1, index + 1) for index in range(len(bands))],
            dtype=object,
        ).reshape(-1, 1)

    shown_labels = [int(value) for value in np.unique(label_array) if include_bad or value != -1]
    for band_index, band in enumerate(bands):
        axis: Axes = axes[band_index, 0]
        values = np.asarray(band_data[band], dtype=float)
        if values.shape[1] != len(label_array):
            raise ValueError(f"Band {band!r} does not align with labels.")
        for label in shown_labels:
            selected = label_array == label
            if not np.any(selected):
                continue
            mean, sem, count = _mean_sem_by_electrode(values[:, selected, :])
            smooth_mean = gaussian_smooth_nan(mean, epochs.sfreq, sigma_seconds=sigma_seconds)
            smooth_sem = gaussian_smooth_nan(sem, epochs.sfreq, sigma_seconds=sigma_seconds)
            color = cluster_color(label)
            display = "Bad" if label == -1 else f"Cluster {label} (n={count})"
            axis.plot(times, smooth_mean, color=color, linewidth=1.8, label=display)
            axis.fill_between(
                times,
                smooth_mean - smooth_sem,
                smooth_mean + smooth_sem,
                color=color,
                alpha=0.18,
                linewidth=0,
            )
        axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
        axis.set_ylabel(f"{band}\nZ power")
        axis.legend(loc="upper right", fontsize=8, ncols=2)
    axes[-1, 0].set_xlabel(f"Time from speech {alignment} (s)")
    return figure, axes


def plot_beta_progress_cluster_dynamics(
    epochs: "BetaProgressEpochs",
    labels: ArrayLike,
    *,
    include_bad: bool = False,
    sigma_seconds: float = 0.0,
    figure: Figure | None = None,
) -> tuple[Figure, NDArray[Any]]:
    """Plot beta dynamics before onset, during speech progress, and after offset.

    The display emphasizes the signals used by the beta-progress classifier:
    beta power, beta power slope, and beta-slope/audio-slope interaction. The
    speech interval is shown as 0-100 percent progress, while pre-onset and
    post-offset context windows use real seconds.
    """

    label_array = np.asarray(labels, dtype=int)
    if label_array.shape != (len(epochs.channel_names),):
        raise ValueError("labels must align one-to-one with epoch channels.")
    if figure is None:
        figure, axes = plt.subplots(
            3,
            3,
            figsize=(14, 8.5),
            constrained_layout=True,
            squeeze=False,
        )
    else:
        figure.clear()
        axes = np.asarray(
            [
                [figure.add_subplot(3, 3, row * 3 + column + 1) for column in range(3)]
                for row in range(3)
            ],
            dtype=object,
        )

    shown_labels = [int(value) for value in np.unique(label_array) if include_bad or value != -1]
    median_duration = _median_event_duration(epochs)
    panels = (
        ("Pre-onset", epochs.pre_times, epochs.pre, epochs.audio_pre, "Time from onset (s)", 1.0),
        (
            "Speech",
            epochs.progress_percent,
            epochs.speech,
            epochs.audio_speech,
            "Speech progress (%)",
            median_duration / 100.0,
        ),
        ("Post-offset", epochs.post_times, epochs.post, epochs.audio_post, "Time from offset (s)", 1.0),
    )
    column_limits = ((float(epochs.pre_times[0]), 0.0), (0.0, 100.0), (0.0, float(epochs.post_times[-1])))

    for column, (title, axis_values, beta_values, audio_values, xlabel, seconds_per_axis_unit) in enumerate(panels):
        display_sfreq = _display_sfreq(axis_values, seconds_per_axis_unit=seconds_per_axis_unit)
        audio_mean = gaussian_smooth_nan(
            np.nanmean(audio_values, axis=0),
            display_sfreq,
            sigma_seconds=sigma_seconds,
        )
        audio_slope = np.gradient(audio_mean, axis_values)
        audio_display = audio_mean.copy()
        audio_display -= np.nanmean(audio_display)
        audio_peak = np.nanmax(np.abs(audio_display)) if np.isfinite(audio_display).any() else np.nan
        if np.isfinite(audio_peak) and audio_peak > 0:
            audio_display /= audio_peak

        axes[0, column].set_title(title)
        axes[0, column].plot(
            axis_values,
            audio_display,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
            label="Audio envelope, scaled" if column == 0 else None,
        )
        axes[1, column].plot(
            axis_values,
            audio_slope,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
            label="Audio slope" if column == 0 else None,
        )

        for label in shown_labels:
            selected = label_array == label
            if not np.any(selected):
                continue
            mean, sem, count = _mean_sem_by_electrode(beta_values[:, selected, :])
            mean = gaussian_smooth_nan(mean, display_sfreq, sigma_seconds=sigma_seconds)
            sem = gaussian_smooth_nan(sem, display_sfreq, sigma_seconds=sigma_seconds)
            beta_slope = np.gradient(mean, axis_values)
            interaction = beta_slope * audio_slope
            color = cluster_color(label)
            display = "Bad" if label == -1 else f"Cluster {label} (n={count})"
            axes[0, column].plot(axis_values, mean, color=color, linewidth=1.8, label=display)
            axes[0, column].fill_between(
                axis_values,
                mean - sem,
                mean + sem,
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            axes[1, column].plot(axis_values, beta_slope, color=color, linewidth=1.5, label=display)
            axes[2, column].plot(axis_values, interaction, color=color, linewidth=1.5, label=display)

        for row in range(3):
            axes[row, column].axhline(0.0, color="black", linewidth=0.5, alpha=0.35)
            axes[row, column].set_xlim(*column_limits[column])
            axes[row, column].set_xlabel(xlabel)

    axes[0, 0].set_ylabel("Beta Z power")
    axes[1, 0].set_ylabel("Beta slope")
    axes[2, 0].set_ylabel("Beta slope x audio slope")
    axes[0, 1].axvline(0.0, color="black", linewidth=0.7, linestyle=":")
    axes[0, 1].axvline(100.0, color="black", linewidth=0.7, linestyle=":")
    axes[1, 1].axvline(0.0, color="black", linewidth=0.7, linestyle=":")
    axes[1, 1].axvline(100.0, color="black", linewidth=0.7, linestyle=":")
    axes[2, 1].axvline(0.0, color="black", linewidth=0.7, linestyle=":")
    axes[2, 1].axvline(100.0, color="black", linewidth=0.7, linestyle=":")
    axes[0, 2].text(
        0.98,
        0.92,
        f"Context window = {epochs.pre_post_window:.3f} s",
        transform=axes[0, 2].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[1, 0].legend(loc="best", fontsize=8)
    return figure, np.asarray(axes, dtype=object)


def _default_nonoverlapping_trials(events: pd.DataFrame) -> tuple[int, int]:
    ordered = events.reset_index(drop=True)
    onset = ordered["onset"].to_numpy(float)
    offset = ordered["offset"].to_numpy(float)
    for first in range(len(ordered)):
        for second in range(first + 1, len(ordered)):
            if offset[first] <= onset[second] or offset[second] <= onset[first]:
                return first, second
    raise ValueError("Could not find two non-overlapping trials.")


def plot_two_nonoverlapping_beta_trials(
    epochs: "BetaProgressEpochs",
    *,
    trial_indices: Sequence[int] | None = None,
    channel_name: str | None = None,
    figure: Figure | None = None,
) -> tuple[Figure, NDArray[Any]]:
    """Plot two non-overlapping trials on one percent-progress figure."""

    if trial_indices is None:
        first, second = _default_nonoverlapping_trials(epochs.events)
    else:
        if len(trial_indices) != 2:
            raise ValueError("trial_indices must contain exactly two trial indices.")
        first, second = (int(trial_indices[0]), int(trial_indices[1]))
    if not (0 <= first < len(epochs.events) and 0 <= second < len(epochs.events)):
        raise IndexError("trial index out of range.")
    first_row = epochs.events.iloc[first]
    second_row = epochs.events.iloc[second]
    if not (
        float(first_row["offset"]) <= float(second_row["onset"])
        or float(second_row["offset"]) <= float(first_row["onset"])
    ):
        raise ValueError("Selected trials overlap in absolute time.")

    if channel_name is None:
        good = np.asarray([name not in epochs.bad_channels for name in epochs.channel_names], dtype=bool)
        if not np.any(good):
            good = np.ones(len(epochs.channel_names), dtype=bool)
        beta = np.nanmean(epochs.speech[:, good, :], axis=1)
        beta_label = "Mean good electrodes"
    else:
        if channel_name not in epochs.channel_names:
            raise ValueError(f"Unknown channel: {channel_name}")
        channel_index = epochs.channel_names.index(channel_name)
        beta = epochs.speech[:, channel_index, :]
        beta_label = channel_name

    if figure is None:
        figure, axes = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True, constrained_layout=True)
    else:
        figure.clear()
        axes = np.asarray([figure.add_subplot(2, 1, index + 1) for index in range(2)])

    colors = ("tab:blue", "tab:orange")
    for color, trial_index in zip(colors, (first, second), strict=True):
        row = epochs.events.iloc[trial_index]
        label = (
            f"Trial {int(row.get('original_trial_id', trial_index))}: "
            f"{float(row['duration']):.2f}s"
        )
        axes[0].plot(epochs.progress_percent, beta[trial_index], color=color, linewidth=1.7, label=label)
        axes[1].plot(
            epochs.progress_percent,
            epochs.audio_speech[trial_index],
            color=color,
            linewidth=1.7,
            label=label,
        )

    axes[0].set_ylabel(f"Beta Z power\n{beta_label}")
    axes[1].set_ylabel("Audio envelope")
    axes[1].set_xlabel("Speech progress (%)")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    return figure, np.asarray(axes, dtype=object)


def _median_event_duration(epochs: "BetaProgressEpochs") -> float:
    median_duration = float(np.nanmedian(epochs.events["duration"].to_numpy(float)))
    if not np.isfinite(median_duration) or median_duration <= 0:
        raise ValueError("Retained events must have a finite positive median duration.")
    return median_duration


def _display_sfreq(axis_values: ArrayLike, *, seconds_per_axis_unit: float = 1.0) -> float:
    axis = np.asarray(axis_values, dtype=float)
    steps = np.diff(axis)
    finite_steps = np.abs(steps[np.isfinite(steps) & (np.abs(steps) > 0)])
    if finite_steps.size == 0 or not np.isfinite(seconds_per_axis_unit) or seconds_per_axis_unit <= 0:
        return 1.0
    return 1.0 / (float(np.nanmedian(finite_steps)) * seconds_per_axis_unit)


def _concat_beta_progress_display(
    epochs: "BetaProgressEpochs",
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """Concatenate pre, normalized speech, and post on a display-time axis."""

    median_duration = _median_event_duration(epochs)
    speech_seconds = epochs.progress_percent / 100.0 * median_duration
    x_axis = np.concatenate(
        [
            epochs.pre_times[:-1],
            speech_seconds,
            median_duration + epochs.post_times[1:],
        ]
    )
    beta = np.concatenate(
        [
            epochs.pre[:, :, :-1],
            epochs.speech,
            epochs.post[:, :, 1:],
        ],
        axis=2,
    )
    audio = np.concatenate(
        [
            epochs.audio_pre[:, :-1],
            epochs.audio_speech,
            epochs.audio_post[:, 1:],
        ],
        axis=1,
    )
    return x_axis, beta, audio, median_duration


def plot_beta_cluster_grid(
    epochs: "BetaProgressEpochs",
    labels: ArrayLike,
    *,
    include_bad: bool = False,
    max_columns: int = 5,
    sigma_seconds: float = 0.0,
    figure: Figure | None = None,
) -> tuple[Figure, NDArray[Any]]:
    """Plot one beta-only cluster panel per cluster with audio overlay.

    The beta signal is averaged across trials first and then across electrodes
    in the cluster. Shading shows SEM across electrodes. The speech interval is
    duration-normalized before averaging and displayed using the retained
    trials' median duration so onset, offset, pre-onset, and post-offset context
    are visible in one continuous axis.
    """

    label_array = np.asarray(labels, dtype=int)
    if label_array.shape != (len(epochs.channel_names),):
        raise ValueError("labels must align one-to-one with epoch channels.")
    if max_columns < 1:
        raise ValueError("max_columns must be positive.")

    shown_labels = [int(value) for value in np.unique(label_array) if include_bad or value != -1]
    if not shown_labels:
        raise ValueError("No clusters are available to plot.")
    column_count = min(max_columns, len(shown_labels))
    row_count = int(np.ceil(len(shown_labels) / column_count))
    if figure is None:
        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(4.2 * column_count, 3.4 * row_count),
            sharex=True,
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

    x_axis, beta_values, audio_values, median_duration = _concat_beta_progress_display(epochs)
    display_sfreq = _display_sfreq(x_axis)
    audio_mean = gaussian_smooth_nan(
        np.nanmean(audio_values, axis=0),
        display_sfreq,
        sigma_seconds=sigma_seconds,
    )
    x_ms = x_axis * 1000.0
    onset_ms = 0.0
    offset_ms = median_duration * 1000.0

    flat_axes = axes.ravel()
    for axis_index, label in enumerate(shown_labels):
        axis: Axes = flat_axes[axis_index]
        selected = label_array == label
        mean, sem, count = _mean_sem_by_electrode(beta_values[:, selected, :])
        mean = gaussian_smooth_nan(mean, display_sfreq, sigma_seconds=sigma_seconds)
        sem = gaussian_smooth_nan(sem, display_sfreq, sigma_seconds=sigma_seconds)
        color = cluster_color(label)
        title = "Bad" if label == -1 else f"Cluster {label}"
        axis.plot(x_ms, mean, color=color, linewidth=2.2)
        axis.fill_between(x_ms, mean - sem, mean + sem, color=color, alpha=0.22, linewidth=0)
        axis.axvline(onset_ms, color="black", linewidth=0.8, linestyle="--")
        axis.axvline(offset_ms, color="black", linewidth=0.8, linestyle="--")
        axis.axhline(0.0, color="black", linewidth=0.5, alpha=0.35)
        axis.axvspan(x_ms[0], onset_ms, color="gray", alpha=0.08, linewidth=0)
        axis.axvspan(offset_ms, x_ms[-1], color="gray", alpha=0.08, linewidth=0)
        axis.set_title(f"{title}\n(n={count})", fontsize=11, fontweight="bold")
        axis.set_ylabel("Beta\nECoG Z")
        axis.set_xlabel("Time from speech onset (ms)")

        audio_axis = axis.twinx()
        audio_axis.plot(x_ms, audio_mean, color="orange", linewidth=1.4, alpha=0.85)
        audio_axis.set_ylim(-0.05, 1.05)
        audio_axis.tick_params(axis="y", colors="orange", labelsize=7)
        if axis_index % column_count != column_count - 1:
            audio_axis.set_yticklabels([])
        else:
            audio_axis.set_ylabel("Audio envelope", color="orange")

        axis.text(
            0.02,
            0.94,
            f"pre/post {epochs.pre_post_window:.2f}s",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "none",
            },
        )

    for unused_axis in flat_axes[len(shown_labels):]:
        unused_axis.set_visible(False)
    return figure, axes


class BadChannelReviewWindow:
    """Matplotlib/Tk window for cluster-level and manual bad-channel review.

    The window is constructed only when :meth:`show` is called, which keeps the
    module safe in headless batch environments. Closing the window without
    pressing Apply returns ``None``.
    """

    def __init__(
        self,
        result: "ClusteringResult",
        *,
        title: str = "ECoG bad-channel review",
        initial_second_k: int = 5,
    ) -> None:
        if not 2 <= initial_second_k <= 10:
            raise ValueError("initial_second_k must be between 2 and 10.")
        self.result = result
        self.title = title
        self.initial_second_k = initial_second_k
        self.selection: BadChannelReview | None = None

    def show(self) -> BadChannelReview | None:
        """Open the modal review UI and return its accepted selection."""

        try:
            import tkinter as tk
            from tkinter import messagebox, ttk
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise ImportError("The review window requires Tk and Matplotlib TkAgg.") from exc

        root = tk.Tk()
        root.title(self.title)
        root.geometry("980x720")
        container = ttk.Frame(root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        figure = Figure(figsize=(7.2, 5.3), constrained_layout=True)
        axis = figure.add_subplot(111, projection="3d")
        plot_pca_clusters(
            self.result.pca_scores,
            self.result.labels,
            channel_names=self.result.channels,
            ax=axis,
        )
        canvas = FigureCanvasTkAgg(figure, master=container)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, rowspan=5, sticky="nsew", padx=(0, 12))

        cluster_frame = ttk.LabelFrame(container, text="Mark anomalous clusters", padding=8)
        cluster_frame.grid(row=0, column=1, sticky="new")
        variables: dict[int, Any] = {}
        for row, label in enumerate(sorted(set(self.result.labels) - {-1})):
            variable = tk.BooleanVar(value=False)
            variables[int(label)] = variable
            count = int(np.count_nonzero(self.result.labels == label))
            ttk.Checkbutton(
                cluster_frame,
                text=f"Cluster {label} ({count} electrodes)",
                variable=variable,
            ).grid(row=row, column=0, sticky="w")

        ttk.Label(container, text="Manual bad channels (comma-separated)").grid(
            row=1, column=1, sticky="sw", pady=(12, 2)
        )
        manual_entry = ttk.Entry(container, width=34)
        manual_entry.grid(row=2, column=1, sticky="new")
        k_frame = ttk.Frame(container)
        k_frame.grid(row=3, column=1, sticky="new", pady=(12, 0))
        ttk.Label(k_frame, text="Second-round K").pack(side=tk.LEFT)
        k_variable = tk.IntVar(value=self.initial_second_k)
        ttk.Spinbox(k_frame, from_=2, to=10, textvariable=k_variable, width=5).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        def accept() -> None:
            raw_names = manual_entry.get().replace(";", ",").split(",")
            manual = tuple(dict.fromkeys(name.strip() for name in raw_names if name.strip()))
            unknown = sorted(set(manual) - set(self.result.channels))
            if unknown:
                messagebox.showerror("Unknown channels", ", ".join(unknown), parent=root)
                return
            try:
                k_value = int(k_variable.get())
            except (TypeError, ValueError):
                messagebox.showerror("Invalid K", "K must be an integer from 2 to 10.", parent=root)
                return
            if not 2 <= k_value <= 10:
                messagebox.showerror("Invalid K", "K must be from 2 to 10.", parent=root)
                return
            clusters = tuple(label for label, variable in variables.items() if variable.get())
            self.selection = BadChannelReview(clusters, manual, k_value)
            root.destroy()

        ttk.Button(container, text="Apply", command=accept).grid(
            row=4, column=1, sticky="se", pady=(16, 0)
        )
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        root.transient()
        root.grab_set()
        root.mainloop()
        plt.close(figure)
        return self.selection


def scanner_ras_to_tkras(scanner_ras: ArrayLike, reference_mri: str | Path) -> FloatArray:
    """Transform Scanner RAS coordinates with ``vox2ras_tkr @ inv(vox2ras)``."""

    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Coordinate transforms require nibabel.") from exc
    coordinates = np.asarray(scanner_ras, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("scanner_ras must have shape (electrodes, 3).")
    image = nib.load(str(reference_mri))
    header = image.header
    if not hasattr(header, "get_vox2ras") or not hasattr(header, "get_vox2ras_tkr"):
        raise ValueError("reference_mri must be a FreeSurfer MGH/MGZ image.")
    transform = np.asarray(header.get_vox2ras_tkr()) @ np.linalg.inv(
        np.asarray(header.get_vox2ras())
    )
    homogeneous = np.column_stack([coordinates, np.ones(len(coordinates))])
    return (transform @ homogeneous.T).T[:, :3]


def load_electrode_coordinates(
    csv_path: str | Path,
    *,
    channel_column: str | None = None,
    coordinate_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load Scanner RAS electrode coordinates into channel/x/y/z columns."""

    frame = pd.read_csv(csv_path)
    def canonical(value: str) -> str:
        return "".join(character for character in value.lower() if character.isalnum())

    lower = {canonical(str(column)): str(column) for column in frame.columns}
    channel_values: pd.Series | None = None
    if channel_column is None:
        channel_column = next(
            (lower[name] for name in ("channel", "name", "electrode", "label") if name in lower),
            None,
        )
        if channel_column is None and "gridid" in lower:
            grid_ids = pd.to_numeric(frame[lower["gridid"]], errors="raise")
            if grid_ids.isna().any():
                raise ValueError("grid id contains missing values; cannot infer ECoG channels.")
            channel_values = grid_ids.astype(int).map(lambda value: f"ECoG_{value}")
    if channel_values is None:
        if channel_column is None or channel_column not in frame:
            raise ValueError("Could not identify the electrode channel-name column.")
        channel_values = frame[channel_column].astype(str)
    if channel_values.isna().any():
        raise ValueError("Could not identify the electrode channel-name column.")
    if coordinate_columns is None:
        candidate_sets = (
            ("registeredlocationx", "registeredlocationy", "registeredlocationz"),
            ("x", "y", "z"),
            ("scannerrasx", "scannerrasy", "scannerrasz"),
            ("rasx", "rasy", "rasz"),
            ("originallocationx", "originallocationy", "originallocationz"),
        )
        coordinate_columns = next(
            ((lower[x], lower[y], lower[z]) for x, y, z in candidate_sets if all(v in lower for v in (x, y, z))),
            None,
        )
    if coordinate_columns is None or len(coordinate_columns) != 3:
        raise ValueError("Could not identify three Scanner RAS coordinate columns.")
    result = pd.DataFrame(
        {
            "channel": channel_values.astype(str),
            "x": pd.to_numeric(frame[coordinate_columns[0]], errors="raise"),
            "y": pd.to_numeric(frame[coordinate_columns[1]], errors="raise"),
            "z": pd.to_numeric(frame[coordinate_columns[2]], errors="raise"),
        }
    )
    if result["channel"].duplicated().any():
        duplicates = result.loc[result["channel"].duplicated(), "channel"].tolist()
        raise ValueError(f"Duplicate electrode channels in coordinate CSV: {duplicates}")
    return result


def _triangular_polydata(vertices: FloatArray, triangles: NDArray[np.integer[Any]]) -> Any:
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Brain rendering requires pyvista.") from exc
    triangles = np.asarray(triangles, dtype=np.int64)
    faces = np.column_stack([np.full(len(triangles), 3, dtype=np.int64), triangles]).ravel()
    return pv.PolyData(np.asarray(vertices, dtype=float), faces)


def _load_pial_mesh(path: Path) -> Any:
    try:
        from nibabel.freesurfer import read_geometry
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("FreeSurfer surface loading requires nibabel.") from exc
    vertices, triangles = read_geometry(str(path))
    return _triangular_polydata(vertices, triangles)


def _load_tumor_mesh(
    tumor_path: str | Path,
    reference_mri: str | Path,
    *,
    threshold: float = 0.5,
) -> Any:
    try:
        import nibabel as nib
        from skimage.measure import marching_cubes
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Tumor rendering requires nibabel and scikit-image.") from exc
    image = nib.load(str(tumor_path))
    volume = np.asarray(image.get_fdata(), dtype=float)
    if not np.any(np.isfinite(volume)) or np.nanmax(volume) <= threshold:
        raise ValueError("Tumor mask contains no voxels above the requested threshold.")
    vertices_voxel, triangles, _, _ = marching_cubes(volume, level=threshold)
    scanner_vertices = nib.affines.apply_affine(image.affine, vertices_voxel)
    tkras_vertices = scanner_ras_to_tkras(scanner_vertices, reference_mri)
    return _triangular_polydata(tkras_vertices, triangles)


def show_pyvista_window(
    plotter: Any,
    *,
    pv_module: Any,
    camera_position: str = "xy",
    screenshot: str | Path | None = None,
) -> Any:
    """Open an interactive desktop PyVista window, even from a notebook."""

    plotter.camera_position = camera_position
    plotter.reset_camera()
    plotter.add_axes()
    try:
        plotter.enable_trackball_style()
    except AttributeError:
        pass

    try:
        pv_module.global_theme.notebook = False
    except Exception:
        pass
    try:
        pv_module.set_jupyter_backend(None)
    except Exception:
        try:
            pv_module.set_jupyter_backend("none")
        except Exception:
            pass

    try:
        result = plotter.show(notebook=False, interactive=True, auto_close=False)
    except TypeError:
        try:
            result = plotter.show(notebook=False)
        except TypeError:
            result = plotter.show()
    if screenshot is not None:
        try:
            plotter.screenshot(str(screenshot), return_img=False)
        except RuntimeError as exc:
            warnings.warn(f"PyVista window opened, but screenshot was not saved: {exc}")
    return result


def _labels_for_electrodes(
    electrodes: pd.DataFrame,
    labels: Mapping[str, int] | Sequence[int],
    channel_order: Sequence[str] | None,
) -> NDArray[np.int64]:
    """Align mapping- or sequence-based labels to coordinate-table rows."""

    if isinstance(labels, Mapping):
        label_by_name = {str(key): int(value) for key, value in labels.items()}
        missing = sorted(set(electrodes["channel"]) - set(label_by_name))
        if missing:
            raise ValueError(f"No cluster label supplied for electrodes: {missing}")
    else:
        label_values = np.asarray(labels, dtype=int)
        order = tuple(channel_order or electrodes["channel"].tolist())
        if len(order) != len(label_values):
            raise ValueError("channel_order and labels must have equal length.")
        label_by_name = dict(zip(map(str, order), map(int, label_values), strict=True))
    return np.asarray(
        [label_by_name[name] for name in electrodes["channel"]], dtype=np.int64
    )


def _set_actor_visibility(actor: Any, visible: bool) -> None:
    if hasattr(actor, "SetVisibility"):
        actor.SetVisibility(bool(visible))
    elif hasattr(actor, "visibility"):
        actor.visibility = bool(visible)


def _set_actor_color(actor: Any, color: Sequence[float]) -> None:
    prop = actor.GetProperty() if hasattr(actor, "GetProperty") else getattr(actor, "prop", None)
    if prop is None:
        return
    if hasattr(prop, "SetColor"):
        prop.SetColor(float(color[0]), float(color[1]), float(color[2]))
    elif hasattr(prop, "color"):
        prop.color = tuple(map(float, color[:3]))


def render_selectable_electrode_class(
    electrode_csv: str | Path,
    labels: Mapping[str, int] | Sequence[int],
    *,
    subjects_dir: str | Path | None = None,
    subject: str | None = None,
    channel_order: Sequence[str] | None = None,
    reference_mri: str | Path | None = None,
    hemispheres: Sequence[str] = ("lh", "rh"),
    tumor_nifti: str | Path | None = None,
    tumor_threshold: float = 0.5,
    selected_label: int | None = None,
    selected_color: str | Sequence[float] = "red",
    electrode_radius: float = 1.6,
    window_size: tuple[int, int] = (1200, 900),
    camera_position: str = "xy",
    brain_color: str = "lightgray",
    brain_opacity: float = 0.38,
    brain_specular: float = 0.08,
    tumor_color: str = "yellow",
    tumor_opacity: float = 0.30,
    background_color: str = "white",
    slider_x: float = 0.02,
    slider_top_y: float = 0.93,
    slider_length: float = 0.24,
    slider_y_gap: float = 0.055,
    slider_width: float = 0.018,
    slider_tube_width: float = 0.005,
    slider_title_height: float = 0.020,
    slider_text_font_size: int = 8,
    show: bool = True,
) -> Any:
    """Render one selectable electrode class with PyVista popup controls."""

    try:
        import matplotlib.colors as mcolors
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Brain rendering requires pyvista and matplotlib.") from exc

    surface_requested = subjects_dir is not None or subject is not None
    if (subjects_dir is None) != (subject is None):
        raise ValueError("subjects_dir and subject must be supplied together.")
    root = None if not surface_requested else Path(subjects_dir) / str(subject)
    if reference_mri is None and root is not None:
        reference_mri = root / "mri" / "orig.mgz"
    reference_path = None if reference_mri is None else Path(reference_mri)
    if reference_path is not None and not reference_path.is_file():
        raise FileNotFoundError(f"FreeSurfer reference MRI does not exist: {reference_path}")
    if tumor_nifti is not None and reference_path is None:
        raise ValueError("tumor_nifti requires reference_mri or FreeSurfer subject data.")

    electrodes = load_electrode_coordinates(electrode_csv)
    aligned_labels = _labels_for_electrodes(electrodes, labels, channel_order)
    unique_labels = tuple(int(label) for label in sorted(set(aligned_labels)))
    if not unique_labels:
        raise ValueError("No electrode labels are available to display.")
    if selected_label is None:
        selected_label = next((label for label in unique_labels if label >= 0), unique_labels[0])
    if selected_label not in unique_labels:
        raise ValueError(f"selected_label {selected_label} is not present in labels: {unique_labels}")

    scanner_coordinates = electrodes[["x", "y", "z"]].to_numpy(dtype=float)
    display_coordinates = (
        scanner_coordinates
        if reference_path is None
        else scanner_ras_to_tkras(scanner_coordinates, reference_path)
    )
    selected_rgb = list(mcolors.to_rgb(selected_color))

    try:
        pv.global_theme.notebook = False
    except Exception:
        pass
    plotter = pv.Plotter(off_screen=not show, notebook=False, window_size=list(window_size))
    if root is not None:
        for hemisphere in hemispheres:
            if hemisphere not in {"lh", "rh"}:
                raise ValueError("hemispheres may contain only 'lh' and 'rh'.")
            pial_path = root / "surf" / f"{hemisphere}.pial"
            if not pial_path.is_file():
                raise FileNotFoundError(f"Pial surface does not exist: {pial_path}")
            plotter.add_mesh(
                _load_pial_mesh(pial_path),
                color=brain_color,
                opacity=brain_opacity,
                smooth_shading=True,
                specular=brain_specular,
            )
    if tumor_nifti is not None:
        assert reference_path is not None
        plotter.add_mesh(
            _load_tumor_mesh(tumor_nifti, reference_path, threshold=tumor_threshold),
            color=tumor_color,
            opacity=tumor_opacity,
            smooth_shading=True,
        )

    actors_by_label: dict[int, list[Any]] = {label: [] for label in unique_labels}
    for coordinate, label in zip(display_coordinates, aligned_labels, strict=True):
        actor = plotter.add_mesh(
            pv.Sphere(
                radius=electrode_radius,
                center=coordinate,
                theta_resolution=24,
                phi_resolution=24,
            ),
            color=selected_rgb if int(label) == selected_label else cluster_color(int(label)),
            smooth_shading=True,
        )
        actors_by_label[int(label)].append(actor)

    state = {"label": int(selected_label), "color": selected_rgb}

    def apply_selection() -> None:
        for label, actors in actors_by_label.items():
            visible = label == state["label"]
            for actor in actors:
                _set_actor_visibility(actor, visible)
                if visible:
                    _set_actor_color(actor, state["color"])
        if hasattr(plotter, "render"):
            plotter.render()

    def set_label(slider_value: float) -> None:
        index = int(round(float(slider_value)))
        index = min(max(index, 0), len(unique_labels) - 1)
        state["label"] = unique_labels[index]
        apply_selection()

    def set_color_component(index: int, value: float) -> None:
        state["color"][index] = float(value)
        apply_selection()

    selected_index = unique_labels.index(int(selected_label))
    plotter.add_slider_widget(
        set_label,
        [0, len(unique_labels) - 1],
        value=selected_index,
        title="Cluster",
        pointa=(slider_x, slider_top_y),
        pointb=(slider_x + slider_length, slider_top_y),
        slider_width=slider_width,
        tube_width=slider_tube_width,
        title_height=slider_title_height,
        style="modern",
    )
    for color_index, (title, y_position) in enumerate(
        (
            ("Red", slider_top_y - slider_y_gap),
            ("Green", slider_top_y - slider_y_gap * 2),
            ("Blue", slider_top_y - slider_y_gap * 3),
        )
    ):
        plotter.add_slider_widget(
            lambda value, index=color_index: set_color_component(index, value),
            [0.0, 1.0],
            value=state["color"][color_index],
            title=title,
            pointa=(slider_x, y_position),
            pointb=(slider_x + slider_length, y_position),
            slider_width=slider_width,
            tube_width=slider_tube_width,
            title_height=slider_title_height,
            style="modern",
        )
    if hasattr(plotter, "add_text"):
        labels_text = ", ".join(map(str, unique_labels))
        plotter.add_text(
            f"Only selected cluster is shown. Labels: {labels_text}",
            position="upper_left",
            font_size=slider_text_font_size,
            color="black",
        )

    plotter.set_background(background_color)
    plotter.camera_position = camera_position
    apply_selection()
    if show:
        show_pyvista_window(plotter, pv_module=pv, camera_position=camera_position)
    return plotter


def render_electrode_clusters(
    electrode_csv: str | Path,
    labels: Mapping[str, int] | Sequence[int],
    *,
    subjects_dir: str | Path | None = None,
    subject: str | None = None,
    channel_order: Sequence[str] | None = None,
    reference_mri: str | Path | None = None,
    hemispheres: Sequence[str] = ("lh", "rh"),
    tumor_nifti: str | Path | None = None,
    tumor_threshold: float = 0.5,
    point_size: float = 14.0,
    window_size: tuple[int, int] = (1200, 900),
    camera_position: str = "xy",
    brain_color: str = "lightgray",
    brain_opacity: float = 0.38,
    brain_specular: float = 0.08,
    tumor_color: str = "yellow",
    tumor_opacity: float = 0.30,
    background_color: str = "white",
    show: bool = True,
    screenshot: str | Path | None = None,
) -> Any:
    """Render clustered electrodes, optionally together with pial surfaces.

    With no ``subjects_dir`` and ``subject``, electrodes are displayed directly
    in Scanner RAS coordinates and no anatomical files are required. Supplying
    both arguments enables FreeSurfer tkRAS transformation and pial surfaces.
    ``reference_mri`` may be supplied alone to transform coordinates without
    drawing pial surfaces. Labels equal to ``-1`` are always white.
    """

    try:
        import matplotlib.colors as mcolors
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Brain rendering requires pyvista and matplotlib.") from exc

    surface_requested = subjects_dir is not None or subject is not None
    if (subjects_dir is None) != (subject is None):
        raise ValueError("subjects_dir and subject must be supplied together.")
    root = None if not surface_requested else Path(subjects_dir) / str(subject)
    if reference_mri is None and root is not None:
        reference_mri = root / "mri" / "orig.mgz"
    reference_path = None if reference_mri is None else Path(reference_mri)
    if reference_path is not None and not reference_path.is_file():
        raise FileNotFoundError(f"FreeSurfer reference MRI does not exist: {reference_path}")
    if tumor_nifti is not None and reference_path is None:
        raise ValueError("tumor_nifti requires reference_mri or FreeSurfer subject data.")

    electrodes = load_electrode_coordinates(electrode_csv)
    aligned_labels = _labels_for_electrodes(electrodes, labels, channel_order)

    scanner_coordinates = electrodes[["x", "y", "z"]].to_numpy(dtype=float)
    display_coordinates = (
        scanner_coordinates
        if reference_path is None
        else scanner_ras_to_tkras(scanner_coordinates, reference_path)
    )
    rgb = np.asarray(
        [np.asarray(mcolors.to_rgb(cluster_color(int(label)))) * 255 for label in aligned_labels],
        dtype=np.uint8,
    )

    try:
        pv.global_theme.notebook = False
    except Exception:
        pass
    plotter = pv.Plotter(off_screen=not show, notebook=False, window_size=list(window_size))
    if root is not None:
        for hemisphere in hemispheres:
            if hemisphere not in {"lh", "rh"}:
                raise ValueError("hemispheres may contain only 'lh' and 'rh'.")
            pial_path = root / "surf" / f"{hemisphere}.pial"
            if not pial_path.is_file():
                raise FileNotFoundError(f"Pial surface does not exist: {pial_path}")
            plotter.add_mesh(
                _load_pial_mesh(pial_path),
                color=brain_color,
                opacity=brain_opacity,
                smooth_shading=True,
                specular=brain_specular,
            )
    if tumor_nifti is not None:
        assert reference_path is not None
        plotter.add_mesh(
            _load_tumor_mesh(tumor_nifti, reference_path, threshold=tumor_threshold),
            color=tumor_color,
            opacity=tumor_opacity,
            smooth_shading=True,
        )
    cloud = pv.PolyData(display_coordinates)
    cloud["cluster_rgb"] = rgb
    plotter.add_points(
        cloud,
        scalars="cluster_rgb",
        rgb=True,
        render_points_as_spheres=True,
        point_size=point_size,
    )
    plotter.set_background(background_color)
    plotter.camera_position = camera_position
    if screenshot is not None and not show:
        plotter.screenshot(str(screenshot), return_img=False)
    if show:
        show_pyvista_window(
            plotter,
            pv_module=pv,
            camera_position=camera_position,
            screenshot=screenshot,
        )
    return plotter


def render_brain_clusters(
    subjects_dir: str | Path,
    subject: str,
    electrode_csv: str | Path,
    labels: Mapping[str, int] | Sequence[int],
    **kwargs: Any,
) -> Any:
    """Compatibility wrapper that explicitly requests FreeSurfer surfaces."""

    return render_electrode_clusters(
        electrode_csv,
        labels,
        subjects_dir=subjects_dir,
        subject=subject,
        **kwargs,
    )


__all__ = [
    "BAD_CHANNEL_COLOR",
    "BASE_CLUSTER_COLORS",
    "BadChannelReview",
    "BadChannelReviewWindow",
    "cluster_color",
    "gaussian_smooth_nan",
    "load_electrode_coordinates",
    "plot_beta_cluster_grid",
    "plot_beta_progress_cluster_dynamics",
    "plot_cluster_dynamics",
    "plot_pca_clusters",
    "plot_two_nonoverlapping_beta_trials",
    "render_brain_clusters",
    "render_electrode_clusters",
    "render_selectable_electrode_class",
    "scanner_ras_to_tkras",
    "show_pyvista_window",
]
