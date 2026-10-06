"""Native regression checks for the idle_draw callbacks seen in kernel logs."""
from dataclasses import replace
import gc
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))


def test_explorer_close_cancels_canvas_callbacks():
    from fourband.explorer import FourBandExplorer
    from fourband_reliability_gui_checks import explorer_inputs
    popup = FourBandExplorer(*explorer_inputs())
    try:
        popup.show(block=False)
        popup.canvas.draw()
        popup.canvas.draw_idle()
        timer = popup.canvas._idle_draw_id
        assert timer is not None
        popup.close()
        assert timer not in popup.root.tk.call("after", "info")
        assert popup.canvas._idle_draw_id is None
        assert popup.figure.canvas is not popup.canvas
        popup.root.tk.call("update", "idletasks")
    finally:
        popup.close()
        gc.collect()


def test_pca_review_close_cancels_canvas_callbacks(monkeypatch):
    import tkinter as tk
    from fourband.review_window import FourBandReviewWindow, ReviewSettings
    from test_fourband_review_window import first_model
    real_tk = tk.Tk
    popup = FourBandReviewWindow(first_model(), initial=ReviewSettings(n_clusters=2))

    def interact():
        popup.canvas.draw()
        popup.canvas.draw_idle()
        timer = popup.canvas._idle_draw_id
        assert timer is not None
        popup.cancel()
        assert timer not in popup.root.tk.call("after", "info")
        assert popup.canvas._idle_draw_id is None
        assert popup.figure.canvas is not popup.canvas
        popup.root.tk.call("update", "idletasks")

    def create():
        root = real_tk()
        root.withdraw()
        root.grab_set = lambda: None
        root.mainloop = interact
        return root

    monkeypatch.setattr(tk, "Tk", create)
    try:
        assert popup.show() is None
    finally:
        popup.cancel()
        gc.collect()


def test_round_exports_do_not_create_tk_windows_even_with_tkagg_backend(tmp_path, monkeypatch):
    import tkinter as tk
    from matplotlib import pyplot as plt
    from fourband.analysis import DEFAULT_BANDS, fit_joint
    from fourband.plots import save_round_plots
    from test_fourband import feature_table
    from test_beta_optimization import curve_epochs
    original_backend = plt.get_backend()
    plt.switch_backend("TkAgg")
    created = []
    real_tk = tk.Tk

    def create(*args, **kwargs):
        root = real_tk(*args, **kwargs)
        root.withdraw()
        created.append(root)
        return root

    monkeypatch.setattr(tk, "Tk", create)
    result = fit_joint(feature_table(), n_clusters=3)
    reference = replace(curve_epochs(np.tile(np.sin(np.linspace(0, 6, 101)), (6, 16, 1))),
                        channel_names=result.channels)
    try:
        save_round_plots({band: reference for band in DEFAULT_BANDS}, result, tmp_path)
        assert (tmp_path / "fourband_cluster_grid.png").stat().st_size > 1000
        assert (tmp_path / "pca_clusters.png").stat().st_size > 1000
        assert not created, "Static exports must not create Tk figures behind the notebook."
    finally:
        plt.close("all")
        plt.switch_backend(original_backend)
        gc.collect()
