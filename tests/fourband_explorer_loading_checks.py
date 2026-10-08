"""Native pending-computation UI checks, isolated from the test runner."""
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np

from test_fourband_continuous import recording


def pending_explorer():
    from fourband.explorer import FourBandExplorer
    raw, events, audio, extraction = recording.__wrapped__()
    result = SimpleNamespace(channels=("E1", "E2"), labels=np.array([0, 1]))
    job = Future()
    popup = FourBandExplorer(raw, events, audio, extraction, result,
                             continuous_submitter=lambda settings: job)
    popup.show(block=False)
    return popup, job


def test_pending_compute_has_axes_and_loading_text_without_fake_curves():
    popup, job = pending_explorer()
    try:
        popup.pump()
        assert not job.done() and not popup._render_valid
        assert len(popup.figure.axes) == 4, "Pending computation must not show a blank canvas."
        assert any("Computing" in text.get_text() for axis in popup.figure.axes for text in axis.texts)
        assert all(not axis.lines for axis in popup.figure.axes), "Do not fabricate ECoG while loading."
    finally:
        popup.close()


def test_status_remains_visible_at_minimum_window_height():
    popup, job = pending_explorer()
    try:
        popup.root.geometry("900x650")
        popup.pump()
        labels = [widget for widget in popup.root.winfo_children()
                  if widget.winfo_class() == "TLabel"]
        assert len(labels) == 1
        status = labels[0]
        assert status.winfo_y() + status.winfo_height() <= popup.root.winfo_height(), \
            "The growing canvas must not push loading/errors out of the window."
    finally:
        popup.close()


def test_channel_change_reuses_identical_pending_computation():
    from fourband.explorer import FourBandExplorer
    raw, events, audio, extraction = recording.__wrapped__()
    result = SimpleNamespace(channels=("E1", "E2"), labels=np.array([0, 1]))
    submitted = []
    def submit(settings):
        future = Future()
        future.set_running_or_notify_cancel()
        submitted.append((settings, future))
        return future
    popup = FourBandExplorer(raw, events, audio, extraction, result, continuous_submitter=submit)
    try:
        popup.show(block=False)
        first = popup._future
        popup.variables["channel"].set("E2")
        popup._changed("channel")
        assert len(submitted) == 1 and popup._future is first, \
            "Channel changes share power data and must not queue another full-record computation."
        popup.variables["sigma"].set("0.15")
        popup._changed("sigma")
        assert len(submitted) == 2, "Changed smoothing settings still need a new computation."
    finally:
        popup.close()


def test_loading_elapsed_updates_without_marking_fake_data_ready(monkeypatch):
    import fourband.explorer as module
    popup, job = pending_explorer()
    try:
        monkeypatch.setattr(module, "monotonic", lambda: popup._loading_started + 12.1)
        popup._update_loading()
        assert all("12 s" in text.get_text() for text in popup._loading_texts)
        assert not popup._render_valid and popup._future is job
    finally:
        popup.close()


def test_loading_failure_shows_error_instead_of_perpetual_computation():
    popup, job = pending_explorer()
    try:
        job.set_exception(ValueError("reference unavailable"))
        popup._finish_continuous(popup._generation, job)
        assert "reference unavailable" in popup.variables["status"].get()
        assert all(text.get_text() == "Display unavailable" for text in popup._loading_texts)
        assert not popup._render_valid and popup._future is None
    finally:
        popup.close()
