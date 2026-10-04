"""Contracts for manual selection, pooled fitting and real file extraction."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "all-channel"))


def api():
    import inputs
    import pipeline
    return inputs, pipeline


def test_selection_resolves_only_requested_blocks_and_shared_baseline(tmp_path):
    inputs, _ = api()
    table = inputs.build_manifest(tmp_path, {"HS0005": ["B03", "B01"]})
    assert table.block_id.tolist() == ["B03", "B01"]
    assert Path(table.fif_path.iloc[0]) == tmp_path / "HS0005/blocks/B03/standardized/ECoG/ecog_TEST_clean_car-raw.fif"
    assert table.baseline_fif.nunique() == 1
    assert len(table) == 2


@pytest.mark.parametrize("selection", [{}, {"HS0005": []}, {"../outside": ["B01"]}, {"HS0005": ["../B01"]}, {"HS0005": ["B01", "B01"]}, {"HS0005": "B01"}])
def test_invalid_selection_fails_before_reading_data(tmp_path, selection):
    inputs, _ = api()
    with pytest.raises(ValueError):
        inputs.build_manifest(tmp_path, selection)


def test_preflight_reports_all_missing_files_and_local_baseline_needs_no_file(tmp_path):
    inputs, _ = api()
    table = inputs.build_manifest(tmp_path, {"HS0005": ["B01"]}, baseline_mode="pre_onset")
    report = inputs.inspect_manifest(table)
    assert not report.ready.any()
    assert "fif_path" in report.missing_files.iloc[0]
    for name in ["fif_path", "onset_path", "offset_path", "wav_path"]:
        path = Path(table[name].iloc[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    inputs.validate_manifest(table)


def test_pooled_fit_retains_identity_and_ignores_numeric_metadata_and_bad_outliers():
    _, pipeline = api()
    table = pd.DataFrame({
        "subject_id": ["A"] * 4 + ["B"] * 4,
        "block_id": ["B01"] * 8,
        "channel": [f"E{i}" for i in range(4)] * 2,
        "sample_id": [f"S{i}" for i in range(8)],
        "is_bad": [False] * 7 + [True],
        "valid_trial_count": [100, 2, 300, 4, 500, 6, 700, 0],
        "beta_progress__early_mean": [-3, -2.9, -3.1, 3, 2.9, 3.1, 3.2, 1e9],
    })
    result, assignments = pipeline.fit_global(table, n_clusters=2)
    assert result.feature_columns == ("beta_progress__early_mean",)
    assert assignments.channel.tolist() == table.channel.tolist()
    assert assignments.sample_id.is_unique
    assert assignments.label.iloc[-1] == -1
    assert assignments.pc1.iloc[-1] != assignments.pc1.iloc[-1]
    assert assignments.label.iloc[0] == assignments.label.iloc[1]
    assert assignments.label.iloc[0] != assignments.label.iloc[3]
    assert result.scaler.mean_[0] == pytest.approx(3.2 / 7)


def test_duplicate_samples_are_rejected():
    _, pipeline = api()
    frame = pd.DataFrame({"subject_id": ["A"] * 3, "block_id": ["B01"] * 3,
                          "channel": ["E0"] * 3, "sample_id": ["A:B01:E0"] * 3,
                          "is_bad": [False] * 3, "beta_progress__early_mean": [0, 1, 2]})
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        pipeline.fit_global(frame, n_clusters=2)


def make_block(root, block):
    import mne
    from scipy.io.wavfile import write
    base = root / "HS0005" / "blocks" / block / "standardized"
    for folder in ["ECoG", "events", "audio"]:
        (base / folder).mkdir(parents=True)
    time = np.arange(2400) / 100
    rng = np.random.default_rng(31 if block == "B01" else 32)
    data = []
    for channel in range(4):
        amp = 1 + .2 * np.sin(time * (channel + 1))
        for start in [3, 8, 13, 18]:
            amp += (-.5 if channel < 2 else .6) * np.exp(-((time - start - 1) / .5) ** 2)
        data.append(1e-5 * (amp * np.sin(2 * np.pi * (18 + channel) * time) + .03 * rng.normal(size=len(time))))
    raw = mne.io.RawArray(np.array(data), mne.create_info([f"E{i}" for i in range(4)], 100, "ecog"), verbose="ERROR")
    raw.info["bads"] = ["E3"]
    raw.save(base / "ECoG/ecog_TEST_clean_car-raw.fif", overwrite=True, verbose="ERROR")
    np.save(base / "events/onset.npy", [3., 8., 13., 18.])
    np.save(base / "events/offset.npy", [5., 10., 15., 20.])
    write(base / "audio/analog1.wav", 1000, (np.sin(np.arange(24000) / 1000 * 2 * np.pi * 70) * 10000).astype(np.int16))


@pytest.mark.parametrize("baseline_mode", ["pre_onset", "file"])
def test_two_blocks_real_file_pipeline_exports_all_rows_and_plots(tmp_path, baseline_mode):
    inputs, pipeline = api()
    root = tmp_path / "data"
    make_block(root, "B01")
    make_block(root, "B03")
    if baseline_mode == "file":
        import mne
        baseline = root / "HS0005/baseline/baseline.fif"
        baseline.parent.mkdir(parents=True)
        raw = mne.io.read_raw_fif(root / "HS0005/blocks/B01/standardized/ECoG/ecog_TEST_clean_car-raw.fif", preload=True, verbose="ERROR")
        raw.save(baseline, overwrite=True, verbose="ERROR")
    manifest = inputs.build_manifest(root, {"HS0005": ["B01", "B03"]}, baseline_mode=baseline_mode)
    output = tmp_path / "results"
    result = pipeline.run_analysis(manifest, output, n_clusters=2, pre_post_window=.5)
    labels = pd.read_csv(output / "global_assignments.csv")
    assert len(labels) == 8
    assert labels.sample_id.nunique() == 8
    assert (labels.loc[labels.channel == "E3", "label"] == -1).all()
    assert labels.loc[labels.channel == "E3", "exclusion_reason"].str.contains("raw_bad").all()
    assert len(result.features) == 8
    assert all(c.startswith(('curve__','relation__')) for c in result.clustering.feature_columns)
    assert (output / 'cluster_grid_beta_audio.png').stat().st_size > 1000
    figures = pd.read_csv(output / 'channel_figure_index.csv')
    assert figures.sample_id.tolist() == result.features.sample_id.tolist()
    assert (figures.loc[labels.label >= 0, 'status'] == 'saved').all()
    assert all(Path(p).stat().st_size > 1000 for p in figures.loc[figures.status == 'saved', 'image_path'])
    assert (output / "global_beta_curves.png").stat().st_size > 1000
    assert (output / "pca_clusters.png").stat().st_size > 1000
    assert (output / "blocks/HS0005/B03/baseline_quality.csv").is_file()
    assert (output / "run_config.json").is_file()
    with np.load(output / "mean_speech_curves.npz") as cache:
        assert cache["audio"].shape == cache["beta"].shape
        assert cache["interaction"].shape == (8, 101)
        assert cache['pre_beta'].shape == (8,51)
        assert cache['pre_times'][0] == pytest.approx(-.5)
        assert cache['post_times'][-1] == pytest.approx(.5)
        assert cache['display_speech_seconds'] == pytest.approx(2.)
        assert cache['full_beta'].shape == (8,201)
        finite=np.isfinite(cache['full_beta']).all(axis=1)
        np.testing.assert_allclose(np.sqrt(np.mean(cache['full_beta'][finite]**2,axis=1)),1)
        np.testing.assert_allclose(cache['full_beta'][finite,:51].mean(axis=1),0,atol=1e-10)
    assert (output / "beta_audio_dynamics.png").stat().st_size > 1000
    from review import run_second_round
    from auto_prune import screen_small_high_variance_clusters
    automatic = screen_small_high_variance_clusters(result.features, result.clustering)
    second = pipeline.GlobalAnalysisResult(result.features, result.assignments, result.clustering, result.output_dir)
    rerun = run_second_round(second, output / "round2", second_n_clusters=2,
                             manual_bad_samples=["HS0005:B01:E0"], automatic_pruning=automatic)
    assert rerun.assignments.loc[rerun.assignments.sample_id == "HS0005:B01:E0", "label"].iloc[0] == -1
    assert (output / "round2/beta_audio_dynamics.png").stat().st_size > 1000
    assert (output / 'round2/cluster_grid_beta_audio.png').stat().st_size > 1000
    assert all(c.startswith(('curve__','relation__')) for c in rerun.clustering.feature_columns)
    assert rerun.clustering.scaler is not result.clustering.scaler
    assert (output / "round2/review_audit.csv").is_file()
    assert (output / "round2/auto_cluster_screening.csv").is_file()
    assert len(pd.read_csv(output / "round2/auto_sample_screening.csv")) == 8
    assert len(pd.read_csv(output / "global_assignments.csv")) == 8
    with pytest.raises(FileExistsError):
        pipeline.run_analysis(manifest, output, n_clusters=2)
