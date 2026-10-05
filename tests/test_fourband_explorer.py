"""Navigation uses actual neighboring events and never modifies fitted labels."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))
from test_fourband_continuous import recording


def test_explorer_navigation_and_channel_selection_are_display_only(recording):
    from fourband.explorer import ExplorerState
    raw, events, audio, extraction = recording
    result = SimpleNamespace(channels=("E1", "E2"), labels=np.array([0, 1]), clean_mask=np.array([True, True]))
    state = ExplorerState(raw, events, audio, extraction, result)
    assert state.trial_ids == (10, 11)
    assert not state.navigate(-1)
    assert not state.navigate(1)
    state.select_channel("E2")
    assert state.channel == "E2" and state.label == 1
    np.testing.assert_array_equal(result.labels, [0, 1])
    with pytest.raises(ValueError, match="channel"):
        state.select_channel("missing")


def test_explorer_rejects_invalid_pair_or_band(recording):
    from fourband.explorer import ExplorerState
    raw, events, audio, extraction = recording
    result = SimpleNamespace(channels=("E1", "E2"), labels=np.array([0, 1]))
    state = ExplorerState(raw, events, audio, extraction, result)
    with pytest.raises(ValueError):
        state.select_pair((10, 12))
    with pytest.raises(ValueError):
        state.select_band("gamma")


def test_trial_flags_use_actual_ids_and_do_not_mask_raw_display(recording):
    from fourband.explorer import ExplorerState
    raw, events, audio, extraction = recording
    extraction.trial_exclusions = (("E1", 11),)
    result = SimpleNamespace(channels=("E1", "E2"), labels=np.array([0, 1]))
    state = ExplorerState(raw, events, audio, extraction, result)
    assert state.excluded_trial_ids == (11,)
    data = state.continuous()
    assert np.isfinite(data.power["theta"][0]).all()
