"""Real Tk/VTK linked-window smoke checks, run explicitly in a subprocess."""
from datetime import datetime
from pathlib import Path
import time
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from test_fourband_continuous import recording


def explorer_inputs():
    raw, events, audio, extraction = recording.__wrapped__()
    for epoch in extraction.epochs.values():
        epoch.events = events.copy()
        epoch.progress_percent = np.linspace(0, 100, 101)
        epoch.speech = np.stack([np.tile(np.sin(np.linspace(0, 4, 101)), (2, 1)) * (i + 1) for i in range(3)])
        epoch.baseline_diagnostics.loc[len(epoch.baseline_diagnostics)] = [12, "E1", 1., 2., True]
        epoch.baseline_diagnostics.loc[len(epoch.baseline_diagnostics)] = [12, "E2", 2., 4., True]
        epoch.audio_speech = np.ones((3, 101))
    extraction.trial_exclusions = (("E1", 11),)
    for epoch in extraction.epochs.values():
        epoch.speech[1, 0] = np.nan
    result = SimpleNamespace(channels=("E1", "E2"), labels=np.array([0, 1]), clean_mask=np.ones(2, bool))
    return raw, events, audio, extraction, result


def output_dir():
    path = Path(__file__).resolve().parents[1] / "3-bands/outputs/verification/gui_reliability" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path.mkdir(parents=True)
    return path


def wait_for_render(popup, timeout=10):
    deadline = time.monotonic() + timeout
    while popup.variables["status"].get() == "Rendering..." and time.monotonic() < deadline:
        popup.pump()
        time.sleep(.01)
    assert popup.variables["status"].get() != "Rendering..."


def test_slow_continuous_processing_keeps_tk_responsive(monkeypatch):
    import fourband.explorer as module
    release = threading.Event()
    started = threading.Event()
    original = module.prepare_continuous_trials
    main_thread = threading.get_ident()
    worker_threads = []

    def slow(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        started.set()
        release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "prepare_continuous_trials", slow)
    popup = module.FourBandExplorer(*explorer_inputs())
    try:
        before = time.monotonic()
        popup.show(block=False)
        assert time.monotonic() - before < 1, "Opening must not wait for full-record signal processing."
        assert started.wait(1)
        heartbeat = []
        popup.root.after(0, lambda: heartbeat.append(True))
        popup.pump()
        assert heartbeat and not release.is_set()
        assert worker_threads == [worker_threads[0]] and worker_threads[0] != main_thread
        release.set()
        wait_for_render(popup)
        assert len(popup.figure.axes) == 8
        assert "unavailable" not in popup.variables["status"].get().lower()
    finally:
        release.set()
        popup.close()


def test_worker_result_is_ignored_after_navigation_or_close(monkeypatch):
    from concurrent.futures import Future
    import fourband.explorer as module
    jobs = []

    def submit(request):
        future = Future()
        future.set_running_or_notify_cancel()
        jobs.append((request, future))
        return future

    monkeypatch.setattr(module, "_CONTINUOUS_WORKER", SimpleNamespace(submit=submit))
    popup = module.FourBandExplorer(*explorer_inputs())
    try:
        popup.show(block=False)
        old_request, old_future = jobs[0]
        old_generation = popup._generation
        popup.navigate(1)
        old_future.set_result(old_request())
        popup._finish_continuous(old_generation, old_future)
        assert popup.variables["status"].get() == "Rendering..."
        assert not popup.figure.axes
        request, future = jobs[1]
        future.set_result(request())
        wait_for_render(popup)
        assert "(11, 12)" in popup.variables["status"].get()
        popup.redraw()
        generation, pending = popup._generation, jobs[-1][1]
        popup.close()
        pending.set_result(None)
        popup._finish_continuous(generation, pending)
        assert popup.closed
    finally:
        popup.close()


def test_failed_new_render_cannot_export_previous_plot(tmp_path, monkeypatch):
    from concurrent.futures import Future
    import fourband.explorer as module
    popup = module.FourBandExplorer(*explorer_inputs(), output_dir=tmp_path)
    try:
        popup.show(block=False)
        wait_for_render(popup)
        assert len(popup.figure.axes) == 8
        failed = Future()
        failed.set_exception(ValueError("test preparation failure"))
        monkeypatch.setattr(module, "_CONTINUOUS_WORKER", SimpleNamespace(submit=lambda _: failed))
        popup.navigate(1)
        wait_for_render(popup)
        popup.save()
        assert not list(tmp_path.glob("*.json")), "Never save old curves with the new trial IDs."
    finally:
        popup.close()


def test_invalid_sigma_does_not_change_pending_pair():
    from fourband.explorer import FourBandExplorer
    popup = FourBandExplorer(*explorer_inputs())
    try:
        popup.show(block=False)
        popup.variables["sigma"].set("invalid")
        popup.variables["pair"].set(popup.pair_labels[1])
        popup._changed("pair")
        assert popup.state.trial_ids == (10, 11)
    finally:
        popup.close()


def test_native_pick_selection_does_not_enter_tk_before_pump(monkeypatch):
    from fourband.explorer import FourBandExplorer
    popup = FourBandExplorer(*explorer_inputs())
    try:
        popup.show(block=False)
        calls = []
        original = popup._sync
        monkeypatch.setattr(popup, "_sync", lambda: (calls.append("tk"), original()))
        popup.select_channel("E2")
        assert not calls, "Picker callback must only queue pure-Python state, not enter Tk."
        popup.pump()
        assert calls and popup.state.channel == "E2"
    finally:
        popup.close()


def test_real_explorer_pair_navigation_detail_and_surface_callback():
    from fourband.continuous import ContinuousPowerCache
    from fourband.explorer import FourBandExplorer
    popup = FourBandExplorer(*explorer_inputs(), cache=ContinuousPowerCache(), output_dir=output_dir())
    try:
        popup.show(block=False)
        popup.pump()
        wait_for_render(popup)
        assert popup.state.trial_ids == (10, 11)
        assert "unavailable" not in popup.variables["status"].get().lower()
        assert all(text.get_text() != "12" for axis in popup.figure.axes for text in axis.texts)
        popup.figure.savefig(popup.output_dir / "continuous.png", dpi=120)
        popup.navigate(1)
        popup.pump()
        wait_for_render(popup)
        assert popup.state.trial_ids == (11, 12)
        assert popup.state.cache.info()["hits"] >= 1
        popup.select_channel("E2")
        popup.pump()
        wait_for_render(popup)
        assert popup.variables["channel"].get() == "E2"
        popup.variables["view"].set("Cluster detail")
        popup.redraw()
        popup.pump()
        assert "unavailable" not in popup.variables["status"].get().lower()
        popup.canvas.draw()
        popup.figure.savefig(popup.output_dir / "detail.png", dpi=120)
        popup.root.geometry("900x650")
        popup.pump()
        popup.canvas.draw()
        popup.save()
        print(f"Explorer render checks: {popup.output_dir}")
    finally:
        popup.close()
    assert popup.closed


def test_native_surface_tick_services_tk_and_pick_selects_real_channel(tmp_path):
    pv = pytest.importorskip("pyvista")
    from visualization import render_electrode_clusters
    from fourband.explorer import FourBandExplorer
    import pandas as pd
    popup = FourBandExplorer(*explorer_inputs(), output_dir=output_dir())
    popup.show(block=False)
    csv = tmp_path / "electrodes.csv"
    pd.DataFrame(dict(channel=["E1", "E2"], x=[0., 10.], y=[0., 0.], z=[0., 0.])).to_csv(csv, index=False)
    plotter = render_electrode_clusters(csv, {"E1": 0, "E2": 1}, show=True,
        on_channel_pick=popup.select_channel, on_window_tick=popup.pump,
        interactive_update=True, window_size=(400, 300))
    try:
        popup.root.after(0, lambda: popup.variables["band"].set("beta2"))
        deadline = time.monotonic() + 2.
        while popup.variables["band"].get() != "beta2" and time.monotonic() < deadline:
            plotter.update(force_redraw=False)
            time.sleep(.01)
        assert popup.variables["band"].get() == "beta2"
        renderer = plotter.renderer
        renderer.SetWorldPoint(10., 0., 0., 1.)
        renderer.WorldToDisplay()
        x, y, _ = renderer.GetDisplayPoint()
        plotter.iren.picker.Pick(x, y, 0, renderer)
        popup.pump()
        assert popup.state.channel == "E2"
    finally:
        plotter.close()
        popup.close()
