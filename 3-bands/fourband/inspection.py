"""Bounded speech-progress inspection conditional on fitted electrode labels.

This module consumes extracted epochs only. Audio and continuous real-time
traces belong to the existing notebook displays and never gate these summaries.
"""
from dataclasses import dataclass
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from signal_processing import finite_mean, gaussian_smooth_nan
from visualization import cluster_color

__all__ = ["InspectionData", "prepare_inspection", "plot_inspection", "save_inspection"]

_AXIS_LABEL = "Duration-normalized speech progress (%)"
_CI_DESCRIPTION = (
    "95% pointwise trial bootstrap CI conditional on fixed electrodes and fitted "
    "labels; common original-trial draws with replacement, no refitting. "
    "Unavailable draws never change the electrode population. "
    "Not significance or cluster stability."
)


@dataclass(frozen=True)
class InspectionData:
    """Copies of included speech arrays, shaped [trial, channel, progress].

    ``channels`` and ``labels`` refer only to clean, nonnegative fitted labels.
    ``trial_ids`` keeps extraction row order, including manually masked trials.
    ``result`` is the original fitted object, retained for caller provenance.
    """

    speech: dict
    channels: tuple
    labels: np.ndarray
    progress_percent: np.ndarray
    trial_ids: tuple
    durations: np.ndarray
    events: pd.DataFrame
    trial_exclusions: tuple
    bands: dict
    result: object
    sigma_seconds: float


def prepare_inspection(extraction, result, sigma_seconds=.10):
    """Smooth each trial in physical seconds without filtering raw or mutating inputs."""
    if not np.isfinite(sigma_seconds) or sigma_seconds < 0:
        raise ValueError("sigma_seconds must be finite and nonnegative.")
    channels = tuple(result.channels)
    labels = np.asarray(result.labels)
    clean = np.asarray(result.clean_mask, dtype=bool)
    if (len(set(channels)) != len(channels) or labels.shape != (len(channels),)
            or clean.shape != labels.shape or not np.isfinite(labels).all()
            or np.any(labels != np.round(labels))):
        raise ValueError("Result channels, integer labels and clean_mask must align.")
    if not extraction.epochs or set(extraction.epochs) != set(extraction.bands):
        raise ValueError("Inspection bands must align with extracted epochs.")
    reference = next(iter(extraction.epochs.values()))
    progress = np.asarray(reference.progress_percent, dtype=float)
    if (progress.ndim != 1 or len(progress) < 2 or not np.isfinite(progress).all()
            or not np.all(np.diff(progress) > 0)
            or not np.allclose(np.diff(progress), np.diff(progress)[0])
            or not np.isclose(progress[0], 0) or not np.isclose(progress[-1], 100)):
        raise ValueError("Speech progress must be uniform and span 0 to 100 percent.")
    events = reference.events
    if not {"original_trial_id", "duration"} <= set(events):
        raise ValueError("Events need original_trial_id and duration for inspection.")
    ids = events.original_trial_id.to_numpy(dtype=float)
    durations = events.duration.to_numpy(dtype=float)
    if (not np.isfinite(ids).all() or np.any(ids != np.round(ids))
            or len(np.unique(ids)) != len(ids) or not np.isfinite(durations).all()
            or np.any(durations <= 0)):
        raise ValueError("Original trial IDs must be unique integers and durations positive.")
    for band, epoch in extraction.epochs.items():
        if (tuple(epoch.channel_names) != channels
                or np.shape(epoch.speech) != (len(events), len(channels), len(progress))
                or not np.array_equal(epoch.progress_percent, progress)
                or not epoch.events.equals(events)):
            raise ValueError(f"{band} trial/channel/progress alignment differs from result or extraction.")
    included = clean & (labels >= 0)
    selected_channels = tuple(name for name, keep in zip(channels, included) if keep)
    trial_ids = tuple(int(value) for value in ids)
    exclusions = tuple((str(channel), int(trial)) for channel, trial
                       in getattr(extraction, "trial_exclusions", ()))
    excluded_pairs = set(exclusions)
    manual = np.array([[(channel, trial) in excluded_pairs for channel in selected_channels]
                       for trial in trial_ids], dtype=bool).reshape(len(events), len(selected_channels))
    speech = {}
    for band, epoch in extraction.epochs.items():
        values = np.asarray(epoch.speech, dtype=float)[:, included, :].copy()
        values[~np.isfinite(values)] = np.nan
        values[manual] = np.nan
        for row, duration in enumerate(durations):
            if len(selected_channels):
                rate = 100. / (duration * np.diff(progress)[0])
                values[row] = gaussian_smooth_nan(values[row], rate, sigma_seconds=sigma_seconds)
        speech[band] = values
    return InspectionData(speech, selected_channels, labels[included].astype(int, copy=True),
        progress.copy(), trial_ids, durations.copy(), events.copy(deep=True), exclusions,
        dict(extraction.bands), result, float(sigma_seconds))


def _validate_bootstrap(n_bootstrap):
    if (isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(n_bootstrap, Integral)
            or n_bootstrap < 0):
        raise ValueError("n_bootstrap must be a nonnegative integer (0 disables CI).")


def _mean_sem(curves):
    count = np.isfinite(curves).sum(axis=0)
    mean = finite_mean(curves)
    square = np.where(np.isfinite(curves), curves - mean, 0.) ** 2
    variance = np.full_like(mean, np.nan)
    np.divide(square.sum(axis=0), count * (count - 1), out=variance, where=count > 1)
    return mean, np.sqrt(variance), count


def _trial_ci(values, n_bootstrap, random_state):
    """Resample trial multiplicities; keep duplicate draws and fixed electrodes.

    Scratch space is one trial/channel/progress block plus bootstrap/progress,
    rather than bootstrap/trial/channel/progress. Identical seeds across bands
    and clusters give common trial draws without storing a draw matrix.
    """
    n_trials, n_channels, n_points = values.shape
    low, high = np.full(n_points, np.nan), np.full(n_points, np.nan)
    supported = np.zeros(n_points, dtype=int)
    if not n_bootstrap or not n_trials or not n_channels:
        return low, high, supported
    valid = np.isfinite(values)
    finite = np.where(valid, values, 0.)
    samples = np.full((n_bootstrap, n_points), np.nan)
    rng = np.random.default_rng(random_state)
    for replicate in range(n_bootstrap):
        weights = np.bincount(rng.integers(0, n_trials, n_trials), minlength=n_trials)
        counts = np.einsum("t,tcp->cp", weights, valid)
        totals = np.einsum("t,tcp->cp", weights, finite)
        means = np.full_like(totals, np.nan)
        np.divide(totals, counts, out=means, where=counts > 0)
        available = (counts > 0).all(axis=0)
        samples[replicate, available] = means[:, available].mean(axis=0)
    supported = np.isfinite(samples).sum(axis=0)
    for point in np.flatnonzero(supported):
        low[point], high[point] = np.quantile(samples[np.isfinite(samples[:, point]), point], [.025, .975])
    return low, high, supported


def _summarize_band(data, band, n_bootstrap, random_state):
    values = data.speech[band]
    means = finite_mean(values)
    counts = np.isfinite(values).sum(axis=0)
    clusters = {}
    for label in np.unique(data.labels):
        mask = data.labels == label
        mean, sem, n_electrodes = _mean_sem(means[mask])
        low, high, supported = _trial_ci(values[:, mask, :], n_bootstrap, random_state)
        clusters[int(label)] = dict(mean=mean, sem=sem, n_electrodes=n_electrodes,
                                    low=low, high=high, supported=supported)
    return means, counts, clusters


def _annotate(axis, message):
    axis.text(.5, .5, message, ha="center", va="center", transform=axis.transAxes, fontsize=10)


def _row_ticks(n_rows):
    return np.unique(np.linspace(0, n_rows - 1, min(20, n_rows), dtype=int))


def _heatmap(axis, values, row_labels, progress, title):
    axis.set_title(title, fontsize=10)
    if not values.shape[0]:
        _annotate(axis, "No included data available")
        return
    step = progress[1] - progress[0]
    image = axis.imshow(np.ma.masked_invalid(values), aspect="auto", interpolation="nearest",
        cmap="RdBu_r", origin="upper",
        extent=(progress[0] - step / 2, progress[-1] + step / 2, len(values) - .5, -.5))
    ticks = _row_ticks(len(row_labels))
    axis.set_yticks(ticks, [row_labels[i] for i in ticks], fontsize=8)
    if not np.isfinite(values).any():
        _annotate(axis, "No finite data available (masked trials/baselines)")
    else:
        axis.figure.colorbar(image, ax=axis, pad=.01, label="Power Z")


def _plot_prepared(data, band, channel, label, n_bootstrap, summary, figure,
                   max_overlay_channels, overlay_channels):
    if band not in data.speech:
        raise ValueError(f"Unknown inspection band: {band}")
    if (isinstance(max_overlay_channels, (bool, np.bool_))
            or not isinstance(max_overlay_channels, Integral) or not 0 <= max_overlay_channels <= 40):
        raise ValueError("max_overlay_channels must be an integer from 0 to 40.")
    if channel is not None and channel not in data.result.channels:
        raise ValueError(f"Unknown result channel: {channel}")
    if overlay_channels is not None:
        overlay_channels = tuple(overlay_channels)
        if (len(set(overlay_channels)) != len(overlay_channels)
                or any(name not in data.channels for name in overlay_channels)
                or len(overlay_channels) > max_overlay_channels):
            raise ValueError("overlay_channels must be unique included channels within the overlay limit.")
    fig = plt.figure(figsize=(13, 10), layout="constrained") if figure is None else figure
    if figure is not None:
        fig.clear()
        fig.set_layout_engine("constrained")
    axes = fig.subplots(3, 1, gridspec_kw={"height_ratios": [1, 1, 1.1]})
    for axis in axes:
        axis.set_xlabel(_AXIS_LABEL)
        axis.set_xlim(0, 100)
    fig.suptitle(f"{band} | fixed fitted labels | smoothing {data.sigma_seconds:g} s", fontsize=12)
    excluded = channel is not None and channel not in data.channels
    if excluded or label == -1:
        for axis in axes:
            _annotate(axis, f"No included cluster data: channel {channel or '(none)'} is excluded")
        return fig, axes
    indices = np.flatnonzero(np.ones(len(data.channels), bool) if label is None else data.labels == label)
    if channel is not None and label is not None and data.labels[data.channels.index(channel)] != label:
        for axis in axes:
            _annotate(axis, f"No data: channel {channel} is outside cluster {label}")
        return fig, axes
    means, _, clusters = summary
    selected_names = tuple(data.channels[i] for i in indices)
    overlay = (selected_names[:max_overlay_channels] if overlay_channels is None
               else tuple(name for name in overlay_channels if name in selected_names))
    for name in overlay:
        index = data.channels.index(name)
        line, = axes[0].plot(data.progress_percent, means[index], linewidth=.65, alpha=.35,
                            color=cluster_color(int(data.labels[index])), label=name)
        line.set_gid("electrode")
    for cluster in np.unique(data.labels[indices]):
        statistics = clusters[int(cluster)]
        color = cluster_color(int(cluster))
        line, = axes[0].plot(data.progress_percent, statistics["mean"], color=color,
                            linewidth=2, label=f"Cluster {cluster} mean")
        line.set_gid("cluster_mean")
        axes[0].fill_between(data.progress_percent, statistics["mean"] - statistics["sem"],
            statistics["mean"] + statistics["sem"], color=color, alpha=.18,
            label=f"Cluster {cluster} electrode SEM")
        if n_bootstrap:
            axes[0].fill_between(data.progress_percent, statistics["low"], statistics["high"],
                facecolor="none", edgecolor=color, hatch="///", alpha=.35,
                label=f"Cluster {cluster} 95% pointwise trial CI (fixed labels)")
    axes[0].set_ylabel("Power Z")
    axes[0].set_title(f"Electrode means | overlays {len(overlay)}/{len(indices)} | means use all included electrodes",
                      fontsize=10)
    if not len(indices) or not np.isfinite(means[indices]).any():
        _annotate(axes[0], "No finite included electrode means available")
    if len(indices):
        from matplotlib.patches import Patch
        handles, names = axes[0].get_legend_handles_labels()
        keep = [i for i, name in enumerate(names) if name.endswith(" mean")]
        compact_handles = [handles[i] for i in keep]
        compact_names = [names[i] for i in keep]
        compact_handles.append(Patch(facecolor="gray", alpha=.18))
        compact_names.append("electrode SEM (n >= 2)")
        if n_bootstrap:
            compact_handles.append(Patch(facecolor="none", edgecolor="gray", hatch="///"))
            compact_names.append("95% pointwise trial CI (fixed labels)")
        axes[0].legend(compact_handles, compact_names, fontsize=7, loc="upper right",
            ncol=min(3, max(1, int(np.ceil(len(compact_names) / 4)))))
    _heatmap(axes[1], means[indices],
        [f"{data.channels[i]} | cluster {data.labels[i]}" for i in indices], data.progress_percent,
        "Electrode-by-time | trial-mean Power Z")
    axes[1].set_ylabel("Actual electrode")
    if channel is None and len(indices):
        valid = [i for i in indices if np.isfinite(means[i]).any()]
        channel = data.channels[valid[0] if valid else indices[0]]
    if channel is None:
        _annotate(axes[2], "No included channel available for trial inspection")
    else:
        index = data.channels.index(channel)
        manual = {trial for name, trial in data.trial_exclusions if name == channel}
        row_labels = [f"{trial} [manual excluded]" if trial in manual else str(trial) for trial in data.trial_ids]
        _heatmap(axes[2], data.speech[band][:, index, :], row_labels, data.progress_percent,
                 f"{channel} trial-by-time | manual exclusions marked; other NaNs remain masked")
        for row, trial in enumerate(data.trial_ids):
            if trial in manual:
                axes[2].axhline(row, color="black", linestyle="--", linewidth=.7)
                axes[2].text(99, row, f"{trial} manual", ha="right", va="center", fontsize=7,
                             bbox=dict(facecolor="white", edgecolor="none", pad=1))
    axes[2].set_ylabel("Original trial ID")
    return fig, axes


def plot_inspection(extraction, result, band, channel=None, label=None, sigma_seconds=.10,
                    n_bootstrap=100, random_state=50, *, figure=None,
                    max_overlay_channels=40, overlay_channels=None):
    """Return ``(Figure, axes[3])``; optionally clear/reuse an explorer Figure.

    Thin overlays are the first <=40 included channels in result order unless
    explicitly selected. Means/SEM and the electrode heatmap use all selected
    included electrodes. An excluded channel returns unavailable annotations.
    ``n_bootstrap=0`` disables CI. No controls refit labels or change exclusions.
    """
    _validate_bootstrap(n_bootstrap)
    data = prepare_inspection(extraction, result, sigma_seconds)
    if band not in data.speech:
        raise ValueError(f"Unknown inspection band: {band}")
    summary = _summarize_band(data, band, n_bootstrap, random_state)
    return _plot_prepared(data, band, channel, label, n_bootstrap, summary, figure,
                          max_overlay_channels, overlay_channels)


def save_inspection(extraction, result, output_dir, sigma_seconds=.10,
                    n_bootstrap=100, random_state=50):
    """Save one figure per band, two CSVs and JSON provenance; return their tables.

    The default selected trial channel is the first included channel with finite
    data in that band. Export count is independent of electrode/cluster count.
    CSV counts are pointwise, and failed bootstrap draws have explicit coverage.
    """
    _validate_bootstrap(n_bootstrap)
    data = prepare_inspection(extraction, result, sigma_seconds)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    electrode_records, cluster_records, files, selected = [], [], [], {}
    for band in data.bands:
        summary = _summarize_band(data, band, n_bootstrap, random_state)
        means, counts, clusters = summary
        finite_channels = np.flatnonzero(np.isfinite(means).any(axis=-1))
        selected[band] = (data.channels[finite_channels[0]] if len(finite_channels)
                          else data.channels[0] if data.channels else None)
        fig, _ = _plot_prepared(data, band, selected[band], None, n_bootstrap, summary, None, 40, None)
        try:
            path = output / f"{band}_inspection.png"
            fig.savefig(path, dpi=130)
            files.append(path)
        finally:
            plt.close(fig)
        for row, channel in enumerate(data.channels):
            for point, progress in enumerate(data.progress_percent):
                electrode_records.append(dict(band=band, channel=channel, label=int(data.labels[row]),
                    progress_percent=float(progress), mean_power_z=float(means[row, point]),
                    n_trials=int(counts[row, point])))
        for label, statistics in clusters.items():
            for point, progress in enumerate(data.progress_percent):
                cluster_records.append(dict(band=band, label=label, progress_percent=float(progress),
                    mean_power_z=float(statistics["mean"][point]), electrode_sem=float(statistics["sem"][point]),
                    n_electrodes=int(statistics["n_electrodes"][point]),
                    trial_ci_low=float(statistics["low"][point]), trial_ci_high=float(statistics["high"][point]),
                    n_bootstrap_supported=int(statistics["supported"][point])))
    electrode_summary = pd.DataFrame(electrode_records, columns=["band", "channel", "label", "progress_percent",
                                                                "mean_power_z", "n_trials"])
    cluster_summary = pd.DataFrame(cluster_records, columns=["band", "label", "progress_percent", "mean_power_z",
        "electrode_sem", "n_electrodes", "trial_ci_low", "trial_ci_high", "n_bootstrap_supported"])
    metadata = dict(channels=list(data.channels), labels=data.labels.tolist(),
        result_channels=list(result.channels), result_labels=np.asarray(result.labels).tolist(),
        result_clean_mask=np.asarray(result.clean_mask, dtype=bool).tolist(),
        trial_ids=list(data.trial_ids), trial_durations_seconds=data.durations.tolist(),
        trial_exclusions=[list(pair) for pair in data.trial_exclusions], bands=data.bands,
        sigma_seconds=data.sigma_seconds, n_bootstrap=int(n_bootstrap),
        random_state=None if random_state is None else int(random_state),
        time_axis=_AXIS_LABEL, trial_ci_description=_CI_DESCRIPTION,
        electrode_sem_description="SEM across included electrode trial means; not significance or trial CI.",
        overlay_limit=40, overlay_selection="First included channels in fitted result order",
        selected_trial_channels=selected, audio="Display-only in other notebook plots; unused here.")
    for table, filename in ((electrode_summary, "inspection_electrodes.csv"),
                            (cluster_summary, "inspection_clusters.csv")):
        path = output / filename
        table.to_csv(path, index=False, encoding="utf-8-sig")
        files.append(path)
    path = output / "inspection_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
    files.append(path)
    return dict(electrode_summary=electrode_summary, cluster_summary=cluster_summary,
                metadata=metadata, files=tuple(files))
