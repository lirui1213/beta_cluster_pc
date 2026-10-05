"""Within-cluster displays preserve trial identity, masks and fixed labels."""
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))
from fourband.analysis import DEFAULT_BANDS, FourBandExtraction


def inspection():
    return importlib.import_module("fourband.inspection")


@pytest.fixture
def sample():
    channels = ("E9", "E2", "E7")
    progress = np.linspace(0, 100, 101)
    events = pd.DataFrame({"original_trial_id": [17, 4, 32], "duration": [1., 2., 1.]})
    speech = np.zeros((3, 3, 101))
    speech[:, :, 50] = 1
    speech[:, 1] = 10000
    speech[2, 0, 30:35] = np.nan
    epochs = {band: SimpleNamespace(speech=speech.copy(), channel_names=channels,
              progress_percent=progress.copy(), events=events.copy(),
              audio_speech=np.full((3, 101), np.nan)) for band in DEFAULT_BANDS}
    extraction = FourBandExtraction(pd.DataFrame(), epochs, dict(DEFAULT_BANDS),
                                    trial_exclusions=(("E7", 4),))
    result = SimpleNamespace(channels=channels, labels=np.array([2, 2, 5]),
                             clean_mask=np.array([True, False, True]))
    return extraction, result


def test_prepare_masks_only_included_channels_retains_ids_and_does_not_mutate(sample):
    extraction, result = sample
    before = {b: e.speech.copy() for b, e in extraction.epochs.items()}
    data = inspection().prepare_inspection(extraction, result, sigma_seconds=0)
    assert data.channels == ("E9", "E7")
    assert data.trial_ids == (17, 4, 32)
    assert data.trial_exclusions == (("E7", 4),)
    np.testing.assert_array_equal(data.labels, [2, 5])
    for band, values in data.speech.items():
        assert values.shape == (3, 2, 101)
        np.testing.assert_allclose(values[:, 0], before[band][:, 0], equal_nan=True)
        assert np.isnan(values[1, 1]).all()
        np.testing.assert_array_equal(extraction.epochs[band].speech, before[band])
    np.testing.assert_array_equal(result.labels, [2, 2, 5])
    np.testing.assert_array_equal(result.clean_mask, [True, False, True])


def test_smoothing_uses_each_trials_physical_duration_and_keeps_nan(sample):
    extraction, result = sample
    data = inspection().prepare_inspection(extraction, result, sigma_seconds=.1)
    impulse = np.zeros(101)
    impulse[50] = 1
    np.testing.assert_allclose(data.speech["theta"][0, 0], gaussian_filter1d(impulse, 10))
    np.testing.assert_allclose(data.speech["theta"][1, 0], gaussian_filter1d(impulse, 5))
    assert np.isnan(data.speech["theta"][2, 0, 30:35]).all()
    assert np.isnan(data.speech["theta"][1, 1]).all()


@pytest.mark.parametrize("sigma", [-1, np.nan, np.inf])
def test_invalid_smoothing_fails(sample, sigma):
    with pytest.raises(ValueError, match="sigma"):
        inspection().prepare_inspection(*sample, sigma_seconds=sigma)


@pytest.mark.parametrize("change", ["channels", "ids", "duration", "progress"])
def test_misaligned_or_invalid_epochs_fail_before_plotting(sample, change):
    extraction, result = sample
    epoch = extraction.epochs["beta1"]
    if change == "channels":
        epoch.channel_names = tuple(reversed(epoch.channel_names))
    elif change == "ids":
        epoch.events.loc[0, "original_trial_id"] = 99
    elif change == "duration":
        epoch.events.loc[0, "duration"] = 0
    else:
        epoch.progress_percent[40] = epoch.progress_percent[39]
    with pytest.raises(ValueError):
        inspection().prepare_inspection(extraction, result)


def test_plot_reuses_figure_and_marks_manual_exclusions_on_original_ids(sample):
    fig = plt.figure()
    try:
        actual, axes = inspection().plot_inspection(*sample, "theta", channel="E7",
                                                   figure=fig, n_bootstrap=5)
        assert actual is fig
        assert len(axes) == 3
        fig.canvas.draw()
        ticks = [tick.get_text() for tick in axes[2].get_yticklabels()]
        assert any("4" in tick and "manual" in tick.lower() for tick in ticks)
        assert any("17" in tick for tick in ticks)
        displayed = np.asarray(axes[2].images[0].get_array())
        assert np.isnan(displayed[1]).all()
        assert all("Duration-normalized" in axis.get_xlabel() for axis in axes)
        assert {line.get_label() for line in axes[0].lines if line.get_gid() == "electrode"} == {"E9", "E7"}
        legends = " ".join(t.get_text() for t in axes[0].get_legend().get_texts())
        assert "electrode SEM" in legends and "pointwise trial" in legends
    finally:
        plt.close(fig)


def test_excluded_result_channel_returns_explicit_unavailable_in_existing_figure(sample):
    fig = plt.figure()
    fig.add_subplot().plot([0, 1])
    try:
        actual, axes = inspection().plot_inspection(*sample, "theta", channel="E2", label=-1,
                                                   n_bootstrap=0, figure=fig)
        assert actual is fig and len(fig.axes) == 3
        fig.canvas.draw()
        assert all(any("excluded" in t.get_text().lower() for t in axis.texts) for axis in axes)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("all_masked", [False, True])
def test_empty_or_all_nan_panels_annotate_and_draw(sample, all_masked):
    extraction, result = sample
    if all_masked:
        for epoch in extraction.epochs.values():
            epoch.speech[:] = np.nan
    else:
        result.clean_mask[:] = False
    fig, axes = inspection().plot_inspection(extraction, result, "theta", n_bootstrap=3)
    try:
        fig.canvas.draw()
        assert all(any("No" in t.get_text() for t in axis.texts) for axis in axes)
    finally:
        plt.close(fig)


def test_overlay_is_bounded_deterministic_and_heatmap_keeps_all_channels(sample):
    extraction, result = sample
    channels = tuple(f"E{i}" for i in range(65))
    result.channels, result.labels, result.clean_mask = channels, np.zeros(65, int), np.ones(65, bool)
    extraction.trial_exclusions = ()
    for epoch in extraction.epochs.values():
        epoch.channel_names = channels
        epoch.speech = np.broadcast_to(np.arange(65)[None, :, None], (3, 65, 101)).copy()
    fig, axes = inspection().plot_inspection(extraction, result, "theta", n_bootstrap=0)
    try:
        curves = [line for line in axes[0].lines if line.get_gid() == "electrode"]
        assert len(curves) == 40
        assert [line.get_label() for line in curves] == list(channels[:40])
        assert axes[1].images[0].get_array().shape == (65, 101)
        mean = next(line for line in axes[0].lines if line.get_gid() == "cluster_mean")
        np.testing.assert_allclose(mean.get_ydata(), 32)
    finally:
        plt.close(fig)
    fig, axes = inspection().plot_inspection(extraction, result, "theta", n_bootstrap=0,
                                           overlay_channels=("E64", "E3"))
    try:
        assert [line.get_label() for line in axes[0].lines if line.get_gid() == "electrode"] == ["E64", "E3"]
    finally:
        plt.close(fig)


def test_export_is_bounded_reproducible_and_conditional_on_fixed_labels(sample, tmp_path):
    existing_figures = plt.get_fignums()
    extraction, result = sample
    for epoch in extraction.epochs.values():
        epoch.speech[:, 0] = np.array([0., 1., 2.])[:, None]
        epoch.speech[:, 2] = 7.
    saved = inspection().save_inspection(extraction, result, tmp_path / "first",
                                         sigma_seconds=0, n_bootstrap=20, random_state=12)
    again = inspection().save_inspection(extraction, result, tmp_path / "second",
                                         sigma_seconds=0, n_bootstrap=20, random_state=12)
    assert len(list((tmp_path / "first").glob("*.png"))) == 4
    electrode = pd.read_csv(tmp_path / "first" / "inspection_electrodes.csv")
    clusters = pd.read_csv(tmp_path / "first" / "inspection_clusters.csv")
    assert set(electrode.channel) == {"E9", "E7"}
    assert set(electrode.loc[electrode.channel == "E7", "n_trials"]) == {2}
    assert set(electrode.loc[electrode.channel == "E7", "mean_power_z"]) == {7.}
    constant = clusters.loc[clusters.label == 5]
    np.testing.assert_allclose(constant[["mean_power_z", "trial_ci_low", "trial_ci_high"]], 7.)
    rng = np.random.default_rng(12)
    expected = np.quantile([np.mean(np.array([0., 1., 2.])[rng.integers(0, 3, 3)])
                            for _ in range(20)], [.025, .975])
    variable = clusters.loc[(clusters.band == "theta") & (clusters.label == 2)]
    np.testing.assert_allclose(variable.trial_ci_low, expected[0])
    np.testing.assert_allclose(variable.trial_ci_high, expected[1])
    pd.testing.assert_frame_equal(saved["cluster_summary"], again["cluster_summary"])
    metadata = json.loads((tmp_path / "first" / "inspection_metadata.json").read_text())
    assert metadata["trial_ids"] == [17, 4, 32]
    assert metadata["trial_exclusions"] == [["E7", 4]]
    assert metadata["n_bootstrap"] == 20
    assert "fixed" in metadata["trial_ci_description"]
    assert "significance" in metadata["trial_ci_description"]
    assert plt.get_fignums() == existing_figures


@pytest.mark.parametrize("n_bootstrap", [-1, 1.5, True])
def test_invalid_bootstrap_counts_fail(sample, n_bootstrap):
    with pytest.raises(ValueError, match="n_bootstrap"):
        inspection().plot_inspection(*sample, "theta", n_bootstrap=n_bootstrap)


def test_trial_ci_keeps_electrodes_fixed_and_reports_unsupported_draws(sample, tmp_path):
    extraction, result = sample
    result.labels[:] = 2
    for epoch in extraction.epochs.values():
        epoch.speech[:, 0] = np.array([0., 0., 9.])[:, None]
        epoch.speech[:, 2] = np.nan
        epoch.speech[0, 2] = 12.
    saved = inspection().save_inspection(extraction, result, tmp_path, sigma_seconds=0,
                                         n_bootstrap=15, random_state=4)
    summary = saved["cluster_summary"]
    np.testing.assert_allclose(summary.mean_power_z, 7.5)
    np.testing.assert_allclose(summary.electrode_sem, 4.5)
    rng = np.random.default_rng(4)
    samples = []
    for _ in range(15):
        draw = rng.integers(0, 3, 3)
        if 0 in draw:
            samples.append((np.array([0., 0., 9.])[draw].mean() + 12.) / 2)
    assert 0 < len(samples) < 15
    assert (summary.n_bootstrap_supported == len(samples)).all()
    expected = np.quantile(samples, [.025, .975])
    np.testing.assert_allclose(summary.trial_ci_low, expected[0])
    np.testing.assert_allclose(summary.trial_ci_high, expected[1])


def test_repeated_explorer_redraw_clears_axes_and_disables_ci(sample):
    fig = plt.figure()
    try:
        for channel in ("E9", "E7", "E9"):
            _, axes = inspection().plot_inspection(*sample, "theta", channel=channel,
                                                   figure=fig, n_bootstrap=0)
            assert len(fig.axes) == 5  # Three panels and two colorbars.
            legend = " ".join(t.get_text() for t in axes[0].get_legend().get_texts())
            assert "trial CI" not in legend
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("empty", ["channels", "trials"])
def test_unavailable_export_still_saves_bounded_figures_and_metadata(sample, tmp_path, empty):
    extraction, result = sample
    if empty == "channels":
        result.clean_mask[:] = False
    else:
        for epoch in extraction.epochs.values():
            epoch.events = epoch.events.iloc[:0].copy()
            epoch.speech = epoch.speech[:0].copy()
    saved = inspection().save_inspection(extraction, result, tmp_path, n_bootstrap=3)
    assert len(list(tmp_path.glob("*.png"))) == 4
    assert (tmp_path / "inspection_metadata.json").is_file()
    if empty == "channels":
        assert saved["electrode_summary"].empty and saved["cluster_summary"].empty
    else:
        assert (saved["electrode_summary"].n_trials == 0).all()
        assert (saved["cluster_summary"].n_bootstrap_supported == 0).all()


def test_unknown_channel_or_excess_overlay_selection_fails_without_leaking_figures(sample):
    existing_figures = plt.get_fignums()
    for kwargs in ({"channel": "unknown"}, {"max_overlay_channels": 41},
                   {"overlay_channels": ("E2",)}, {"overlay_channels": ("E9", "E9")}):
        with pytest.raises(ValueError):
            inspection().plot_inspection(*sample, "theta", n_bootstrap=0, **kwargs)
    assert plt.get_fignums() == existing_figures


def test_many_cluster_legend_stays_bounded_and_inside_top_panel(sample):
    extraction, result = sample
    names = tuple(f"E{i}" for i in range(10))
    result = SimpleNamespace(channels=names, labels=np.arange(10), clean_mask=np.ones(10, bool))
    for epoch in extraction.epochs.values():
        epoch.channel_names = names
        epoch.speech = np.tile(np.arange(10)[None, :, None], (3, 1, 101)).astype(float)
    extraction.trial_exclusions = ()
    fig, axes = inspection().plot_inspection(extraction, result, "theta", n_bootstrap=5)
    try:
        fig.canvas.draw()
        legend = axes[0].get_legend()
        assert len(legend.get_texts()) <= 12
        bounds = legend.get_window_extent(fig.canvas.get_renderer())
        assert bounds.y0 >= axes[0].get_window_extent().y0
    finally:
        plt.close(fig)
