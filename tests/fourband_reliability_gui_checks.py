"""Real Tk/VTK linked-window smoke checks, run explicitly in a subprocess."""
from datetime import datetime
from pathlib import Path
import time
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


def test_real_explorer_pair_navigation_detail_and_surface_callback():
    from fourband.continuous import ContinuousPowerCache
    from fourband.explorer import FourBandExplorer
    popup = FourBandExplorer(*explorer_inputs(), cache=ContinuousPowerCache(), output_dir=output_dir())
    try:
        popup.show(block=False)
        popup.pump()
        assert popup.state.trial_ids == (10, 11)
        assert "unavailable" not in popup.variables["status"].get().lower()
        assert all(text.get_text() != "12" for axis in popup.figure.axes for text in axis.texts)
        popup.figure.savefig(popup.output_dir / "continuous.png", dpi=120)
        popup.navigate(1)
        popup.pump()
        assert popup.state.trial_ids == (11, 12)
        assert popup.state.cache.info()["hits"] >= 1
        popup.select_channel("E2")
        popup.pump()
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
            plotter.iren.process_events()
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
