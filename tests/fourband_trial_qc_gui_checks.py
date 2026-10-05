"""Real Tk checks, isolated from Windows pytest fd capture."""
import gc
import numpy as np
import pytest
from test_fourband_trial_qc import qc_fixture, qc_module


@pytest.fixture
def hidden_tk(monkeypatch):
    import tkinter as tk
    gc.collect()
    factory = tk.Tk
    callback = {}
    def create():
        root = factory()
        root.withdraw()
        root.grab_set = lambda: None
        root.mainloop = lambda: callback["run"]()
        return root
    monkeypatch.setattr(tk, "Tk", create)
    yield callback
    callback.clear()
    gc.collect()


def popup_fixture():
    raw, extraction = qc_fixture()
    raw._data[0, 550] = .01
    report = qc_module().detect_trial_quality(raw, extraction)
    popup = qc_module("trial_review_window").TrialQualityReviewWindow(raw, extraction, report)
    return popup


def test_all_rows_unchecked_and_raw_waveform_draws(hidden_tk):
    popup = popup_fixture()
    def interact():
        assert len(popup.table.get_children()) == 8
        assert not any(v.get() for v in popup.pair_variables.values())
        popup.display_pair(("E1", 17))
        popup.canvas.draw()
        line = popup.axis.lines[0]
        mask = (popup.raw.times >= 4.5) & (popup.raw.times <= 6.5)
        np.testing.assert_allclose(line.get_ydata(), popup.raw.get_data()[0, mask] * 1e6)
        assert popup.axis.get_xlabel() == "Seconds from onset"
        assert "E1" in popup.axis.get_title() and "17" in popup.axis.get_title()
        popup.cancel()
    hidden_tk["run"] = interact
    assert popup.show() is None


def test_checkbox_manual_unflagged_pair_and_confirm(hidden_tk):
    popup = popup_fixture()
    def interact():
        popup.display_pair(("E2", 31))
        popup.pair_variables[("E2", 31)].set(True)
        assert popup.remaining_counts()["E2"] == 3
        popup.confirm()
    hidden_tk["run"] = interact
    assert popup.show() == (("E2", 31),)


def test_bulk_selection_is_reversible_and_cancel_discards_pending(hidden_tk):
    popup = popup_fixture()
    def interact():
        popup.set_pair(("E2", 31), True)
        popup.select_candidates()
        assert ("E1", 17) in popup.selected_pairs
        popup.undo_selection()
        assert popup.selected_pairs == {("E2", 31)}
        popup.clear_selection()
        assert not popup.selected_pairs
        popup.undo_selection()
        assert popup.selected_pairs == {("E2", 31)}
        popup.cancel()
        assert all(not v.trace_info() for v in popup.pair_variables.values())
    hidden_tk["run"] = interact
    assert popup.show() is None


def test_empty_confirmation_is_distinct_from_close(hidden_tk):
    popup = popup_fixture()
    hidden_tk["run"] = popup.confirm
    accepted = popup.show()
    assert isinstance(accepted, tuple) and accepted == ()


def test_existing_tk_owner_survives_and_pending_draw_is_cancelled():
    import tkinter as tk
    owner = tk.Tk()
    owner.withdraw()
    popup = popup_fixture()
    observed = []
    def interact():
        popup.canvas.draw()
        popup.canvas.draw_idle()
        timer = popup.canvas._idle_draw_id
        popup.cancel()
        observed.append(timer not in owner.tk.call("after", "info"))
    owner.after(100, interact)
    try:
        assert popup.show() is None
        assert owner.winfo_exists()
        assert observed == [True]
    finally:
        owner.destroy()
