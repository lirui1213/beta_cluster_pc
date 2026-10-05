"""Session power reuse must never reuse stale signal or baseline state."""
import copy
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))
from test_fourband_continuous import recording  # noqa: E402, F401


@pytest.fixture
def cache():
    module = importlib.import_module("fourband.continuous")
    assert hasattr(module, "ContinuousPowerCache"), "bounded power cache is missing"
    return module.ContinuousPowerCache()


@pytest.fixture
def filter_calls(monkeypatch):
    import mne
    original = mne.io.BaseRaw.filter
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(mne.io.BaseRaw, "filter", counted)
    return calls


def prepare(recording, **kwargs):
    return importlib.import_module("fourband.continuous").prepare_continuous_trials(
        *recording, **kwargs)


def test_reuses_power_across_display_pairs_baselines_and_review_masks(recording, cache, filter_calls):
    raw, events, audio, extraction = recording
    for epochs in extraction.epochs.values():
        epochs.events = events.copy()
        third = epochs.baseline_diagnostics.iloc[2:].copy()
        third["original_trial_id"] = 12
        epochs.baseline_diagnostics = pd.concat([epochs.baseline_diagnostics, third], ignore_index=True)
    original = prepare(recording, cache=cache, sigma_seconds=0)
    revised = copy.deepcopy(extraction)
    revised.trial_exclusions = (("E1", 11),)
    for epochs in revised.epochs.values():
        epochs.baseline_diagnostics["baseline_mean"] += 3
        epochs.baseline_diagnostics.loc[epochs.baseline_diagnostics.channel == "E2", "valid"] = False
    current = (raw, events, audio, revised)
    displayed = prepare(current, cache=cache, sigma_seconds=.15, display_sfreq=80)
    paired = prepare(current, cache=cache, trial_ids=(11, 12), sigma_seconds=0)
    same_interval = prepare(current, cache=cache, sigma_seconds=0)
    assert len(filter_calls) == 4
    assert cache.info()["hits"] == 3
    assert np.isnan(displayed.power["theta"][1]).all()
    assert not np.isnan(displayed.power["theta"][0]).all()
    assert paired.metadata["baseline_anchor_trial_id"] == 11
    uncached = prepare(current, trial_ids=(11, 12), sigma_seconds=0)
    for band in paired.power:
        np.testing.assert_allclose(paired.power[band], uncached.power[band], equal_nan=True)
    np.testing.assert_allclose(original.power["theta"][0] - 1.5, same_interval.power["theta"][0])
    assert paired.metadata["power_cache"]["status"] == "hit"


def test_mutable_raw_revision_outside_display_invalidates_full_record(recording, cache, filter_calls):
    first = prepare(recording, cache=cache)
    recording[0]._data[0, -100:] *= 8
    second = prepare(recording, cache=cache)
    assert len(filter_calls) == 8
    expected = prepare(recording)
    for band in second.power:
        np.testing.assert_allclose(second.power[band], expected.power[band], equal_nan=True)
    assert first.metadata["power_cache"]["status"] == "miss"
    assert second.metadata["power_cache"]["status"] == "miss"


@pytest.mark.parametrize("revision", ["bands", "filter", "sfreq", "raw_order", "channel_order"])
def test_processing_provenance_invalidates(recording, cache, filter_calls, revision):
    prepare(recording, cache=cache)
    raw, _, _, extraction = recording
    if revision == "bands":
        extraction.bands["theta"] = (5., 9.)
        extraction.epochs["theta"].beta_band = (5., 9.)
    elif revision == "filter":
        extraction.filter_kwargs = dict(method="iir", iir_params=dict(order=3, ftype="butter"))
    elif revision == "sfreq":
        with raw.info._unlock():
            raw.info["sfreq"] = 401.
        for epochs in extraction.epochs.values():
            epochs.sfreq = 401.
    elif revision == "raw_order":
        raw.reorder_channels(["E2", "E1"])
    else:
        for epochs in extraction.epochs.values():
            epochs.channel_names = ("E2", "E1")
            epochs.baseline_diagnostics = epochs.baseline_diagnostics.iloc[[1, 0, 3, 2]].reset_index(drop=True)
    changed = prepare(recording, cache=cache)
    assert len(filter_calls) == 8
    assert changed.metadata["power_cache"]["status"] == "miss"


def test_full_raw_revision_includes_channels_not_displayed(recording, cache, filter_calls):
    import mne
    raw, events, audio, extraction = recording
    raw.add_channels([mne.io.RawArray(np.ones((1, raw.n_times)),
        mne.create_info(["AUX"], raw.info["sfreq"], "misc"), verbose=False)])
    prepare(recording, cache=cache)
    raw._data[2, 123] = 10
    prepare(recording, cache=cache)
    assert len(filter_calls) == 8


def test_budget_fallback_is_bounded_correct_and_reported(recording, cache, filter_calls):
    small = type(cache)(max_bytes=1)
    limited = prepare(recording, cache=small)
    assert small.info()["bytes"] == 0
    assert small.info()["entries"] == 0
    assert small.info()["fallbacks"] == 1
    assert limited.metadata["power_cache"]["status"] == "fallback"
    assert "limit" in limited.metadata["power_cache"]["reason"]
    expected = prepare(recording)
    for band in limited.power:
        np.testing.assert_allclose(limited.power[band], expected.power[band])
    prepare(recording, cache=small)
    assert len(filter_calls) == 12


def test_lru_eviction_clear_and_returned_arrays_cannot_poison_cache(recording, cache, filter_calls):
    budget = 4 * 2 * 4000 * 8
    bounded = type(cache)(max_bytes=budget)
    first = prepare(recording, cache=bounded)
    first.power["theta"][:] = 999
    pristine = prepare(recording, cache=bounded)
    assert not np.any(pristine.power["theta"] == 999)
    recording[0]._data[0, 1200] += 3
    prepare(recording, cache=bounded)
    assert bounded.info()["bytes"] <= budget
    assert bounded.info()["entries"] == 1
    assert bounded.info()["evictions"] == 1
    bounded.clear()
    assert bounded.info()["bytes"] == bounded.info()["entries"] == 0
    prepare(recording, cache=bounded)
    assert len(filter_calls) == 12


def test_nonpreloaded_raw_reads_revision_in_bounded_chunks(recording, cache, tmp_path, monkeypatch):
    import mne
    raw, events, audio, extraction = recording
    raw = mne.io.RawArray(np.tile(raw.get_data(), (1, 20)), raw.info, verbose=False)
    path = tmp_path / "recording_raw.fif"
    raw.save(path, overwrite=True, verbose=False)
    lazy = mne.io.read_raw_fif(path, preload=False, verbose=False)
    reads = []
    get_data = mne.io.BaseRaw.get_data

    def bounded_read(self, *args, **kwargs):
        if self is lazy:
            assert kwargs.get("start") is not None and kwargs.get("stop") is not None
            reads.append(kwargs["stop"] - kwargs["start"])
        return get_data(self, *args, **kwargs)

    monkeypatch.setattr(mne.io.BaseRaw, "get_data", bounded_read)
    prepare((lazy, events, audio, extraction), cache=cache)
    cached = prepare((lazy, events, audio, extraction), cache=cache)
    assert len(reads) >= 4
    assert max(reads) * len(raw.ch_names) * 8 <= 1024 ** 2
    assert cache.info()["hits"] == 1
    expected = prepare((lazy, events, audio, extraction))
    for band in cached.power:
        np.testing.assert_allclose(cached.power[band], expected.power[band])


def test_load_accepts_explicit_cache(recording, cache, monkeypatch):
    module = importlib.import_module("fourband.continuous")
    raw, events, audio, extraction = recording
    monkeypatch.setattr(module, "load_ecog_fif", lambda *a, **k: raw)
    monkeypatch.setattr(module, "load_speech_events", lambda *a, **k: events)
    monkeypatch.setattr(module, "extract_audio_envelope", lambda *a, **k: audio)
    module.load_continuous_trials("raw", "on", "off", "wav", extraction, cache=cache)
    module.load_continuous_trials("raw", "on", "off", "wav", extraction, cache=cache)
    assert cache.info()["hits"] == 1


def test_confirmed_exclusions_are_metadata_only_for_continuous_signal(recording, cache):
    original = prepare(recording, cache=cache)
    recording[3].trial_exclusions = (("E1", 10), ("E2", 11))
    reviewed = prepare(recording, cache=cache)
    assert reviewed.metadata["trial_exclusions"] == [("E1", 10), ("E2", 11)]
    for band in reviewed.power:
        np.testing.assert_array_equal(reviewed.power[band], original.power[band])
    assert original.metadata["trial_exclusions"] == []


@pytest.mark.parametrize("reviewed", [False, True])
def test_export_marks_qc_per_cluster_event_without_masking_curves(recording, reviewed, monkeypatch, tmp_path):
    from types import SimpleNamespace
    module = importlib.import_module("fourband.continuous")
    if reviewed:
        recording[3].trial_exclusions = (("E1", 10), ("E2", 10), ("E2", 11))
    data = prepare(recording)
    result = SimpleNamespace(channels=("E1", "E2"), labels=np.array([0, 1]),
        clean_mask=np.ones(2, dtype=bool))
    subplots = module.plt.subplots
    figures = []

    def capture(*args, **kwargs):
        fig, axes = subplots(*args, **kwargs)
        figures.append(axes)
        return fig, axes

    monkeypatch.setattr(module.plt, "subplots", capture)
    table = module.save_continuous_trial_plot(data, result, tmp_path / "qc_plot")
    axes = figures[0]
    if reviewed:
        for row in (0, 1):
            assert any("Trial 10" in text.get_text() and "QC 1" in text.get_text()
                for text in axes[row, 0].texts)
            assert any(patch.get_hatch() for patch in axes[row, 0].patches)
        assert any("Trial 11" in text.get_text() and "QC 1" in text.get_text()
            for text in axes[1, 0].texts)
        assert not any("Trial 11" in text.get_text() and "QC" in text.get_text()
            for text in axes[0, 0].texts)
    else:
        assert not any("QC" in text.get_text() for axis in axes.flat for text in axis.texts)
        assert not any(patch.get_hatch() for axis in axes.flat for patch in axis.patches)
    for row, channel in enumerate(("E1", "E2")):
        np.testing.assert_array_equal(axes[row, 0].lines[0].get_ydata(), data.power["theta"][row])
        np.testing.assert_allclose(table.loc[(table.band == "theta") & (table.label == row), "mean_power"],
            data.power["theta"][row])
