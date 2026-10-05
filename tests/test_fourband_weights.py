"""Explicit group budgets cannot silently add audio or feature-count weight."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))
from test_fourband import feature_table
from fourband.analysis import DEFAULT_BANDS, fit_joint


def diverse_table():
    table = feature_table()
    rng = np.random.default_rng(34)
    for band in DEFAULT_BANDS:
        for suffix in ("middle_mean", "energy_mean", "energy_std", "power_slope_mean"):
            table[f"{band}__{suffix}"] = rng.normal(size=len(table))
    return table


def test_legacy_preset_is_exactly_existing_weights():
    old = fit_joint(diverse_table(), n_clusters=2)
    new = fit_joint(diverse_table(), n_clusters=2, weight_preset="legacy")
    assert old.feature_columns == new.feature_columns
    assert old.feature_weights == new.feature_weights
    np.testing.assert_array_equal(old.labels, new.labels)


@pytest.mark.parametrize("preset,expected", [("balanced", (1, 1, 1)),
    ("morphology", (3, 1, 1)), ("amplitude", (1, 3, 1))])
def test_each_group_receives_budget_not_number_of_features(preset, expected):
    from fourband.weights import feature_group
    result = fit_joint(diverse_table(), n_clusters=2, weight_preset=preset)
    for band in DEFAULT_BANDS:
        values = {g: sum(w for c, w in zip(result.feature_columns, result.feature_weights)
            if c.startswith(band + "__") and feature_group(c) == g)
            for g in ("morphology", "amplitude", "slope")}
        np.testing.assert_allclose(list(values.values()), np.asarray(expected) / sum(expected))


def test_zero_group_is_removed_before_redundancy_selection():
    from fourband.weights import feature_group
    table = diverse_table()
    for band in DEFAULT_BANDS:
        table[f"{band}__energy_mean"] = table[f"{band}__early_mean"]
    fit = fit_joint(table, n_clusters=2, weight_preset="custom",
        group_weights=dict(morphology=0, amplitude=1, slope=0))
    assert all(feature_group(c) == "amplitude" for c in fit.feature_columns)
    assert all("audio_slope" not in c for c in fit.feature_columns)
    for band in DEFAULT_BANDS:
        assert sum(w for c, w in zip(fit.feature_columns, fit.feature_weights)
            if c.startswith(band + "__")) == pytest.approx(1)


@pytest.mark.parametrize("kwargs", [dict(weight_preset="unknown"),
    dict(weight_preset="custom", group_weights=dict(morphology=0, amplitude=0, slope=0)),
    dict(weight_preset="custom", group_weights=dict(amplitude=-1)),
    dict(weight_preset="custom", group_weights=dict(slope=np.nan)),
    dict(weight_preset="balanced", group_weights=dict(slope=3)),
    dict(weight_preset="custom", group_weights=dict(audio=3))])
def test_invalid_weight_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        fit_joint(diverse_table(), n_clusters=2, **kwargs)


def test_missing_group_is_reported_and_band_budget_retained():
    result = fit_joint(feature_table(), n_clusters=2, weight_preset="balanced")
    from fourband.weights import weight_audit
    audit = weight_audit(result)
    assert (audit.loc[audit.group == "amplitude", "effective_budget"] == 0).all()
    assert (audit.loc[audit.group == "morphology", "effective_budget"] == 1).all()
    assert len(audit) == 12


def test_large_finite_custom_group_ratios_match_balanced():
    balanced = fit_joint(diverse_table(), n_clusters=2, weight_preset="balanced")
    huge = fit_joint(diverse_table(), n_clusters=2, weight_preset="custom",
        group_weights=dict(morphology=1e308, amplitude=1e308, slope=1e308))
    np.testing.assert_allclose(huge.feature_weights, balanced.feature_weights)
    np.testing.assert_array_equal(huge.labels, balanced.labels)


def test_export_records_effective_groups_and_confirmed_trial_mask(tmp_path, monkeypatch):
    import json
    from dataclasses import replace
    import pandas as pd
    from fourband import workflow
    from fourband.analysis import FourBandExtraction
    from test_beta_optimization import curve_epochs
    table = diverse_table()
    fitted = fit_joint(table, n_clusters=2, weight_preset="balanced")
    epoch = replace(curve_epochs(np.zeros((6, 16, 101))), channel_names=fitted.channels)
    extraction = FourBandExtraction(table, {b: epoch for b in DEFAULT_BANDS}, dict(DEFAULT_BANDS),
        feature_settings={"smoothing_sigma_seconds": .06}, trial_exclusions=(("E0", 2),))
    monkeypatch.setattr(workflow, "save_round_plots", lambda *a, **k: None)
    workflow.save_round(extraction, fitted, tmp_path / "fit", n_clusters=2,
        round_config=dict(weight_preset="balanced"))
    audit = pd.read_csv(tmp_path / "fit/feature_group_budgets.csv")
    assert len(audit) == 12
    settings = json.loads((tmp_path / "fit/round_config.json").read_text())
    assert settings["trial_exclusions"] == [["E0", 2]]
    assert settings["weight_preset"] == "balanced"
    assert (tmp_path / "fit/fitted_electrode_features.csv").is_file()
