"""Explicit subprocess checks for real Tk; not collected by the parent suite."""
import gc
import pytest
from test_fourband_review_window import module, first_model


@pytest.fixture
def hidden_tk(monkeypatch):
    import tkinter as tk
    gc.collect()
    real_tk = tk.Tk
    callback = {}
    def create():
        root = real_tk()
        root.withdraw()
        root.grab_set = lambda: None
        root.mainloop = lambda: callback["run"]()
        return root
    monkeypatch.setattr(tk, "Tk", create)
    yield callback
    callback.clear()
    gc.collect()


def test_real_popup_controls_apply_threshold_reference_manual_cluster_and_k(hidden_tk):
    popup = module().FourBandReviewWindow(first_model(), initial=module().ReviewSettings(n_clusters=2))
    def interact():
        popup.cluster_variables[2].set(True)
        popup.variables["manual"].set("E0")
        popup.variables["distance_enabled"].set(True)
        popup.variables["distance_threshold"].set("3")
        popup.variables["distance_reference"].set("global")
        popup.variables["n_clusters"].set("2")
        popup.update_preview()
        assert popup.preview_audit.excluded_final.sum() == 3
        assert popup.widgets["apply"].instate(["!disabled"])
        assert int(float(popup.widgets["n_clusters"].cget("to"))) == 3
        popup.apply()
    hidden_tk["run"] = interact
    settings = popup.show()
    assert settings.n_clusters == 2 and settings.bad_clusters == (2,)
    assert settings.manual_bad_channels == ("E0",) and settings.pca_distance_threshold == 3


def test_real_popup_disables_apply_for_invalid_k_then_cancels(hidden_tk):
    popup = module().FourBandReviewWindow(first_model(), initial=module().ReviewSettings(n_clusters=2))
    def interact():
        popup.variables["n_clusters"].set("2.5")
        popup.update_preview()
        assert popup.widgets["apply"].instate(["disabled"])
        popup.cancel()
    hidden_tk["run"] = interact
    assert popup.show() is None


def test_direct_preview_consumes_its_pending_timer_before_cancel(hidden_tk):
    popup = module().FourBandReviewWindow(first_model(), initial=module().ReviewSettings(n_clusters=2))
    def interact():
        popup.variables["manual"].set("E0")
        timer = popup._pending_preview
        popup.update_preview()
        assert timer not in popup.root.tk.call("after", "info")
        popup.cancel()
    hidden_tk["run"] = interact
    assert popup.show() is None


def test_real_popup_pca_axis_labels_fit_inside_canvas(hidden_tk):
    popup = module().FourBandReviewWindow(first_model(), initial=module().ReviewSettings(n_clusters=2))
    def interact():
        popup.canvas.draw()
        renderer = popup.canvas.get_renderer()
        width, height = popup.figure.bbox.width, popup.figure.bbox.height
        for label in (popup.axis.xaxis.label, popup.axis.yaxis.label, popup.axis.zaxis.label):
            bounds = label.get_window_extent(renderer)
            assert 0 <= bounds.x0 <= bounds.x1 <= width
            assert 0 <= bounds.y0 <= bounds.y1 <= height
        popup.cancel()
    hidden_tk["run"] = interact
    assert popup.show() is None


def test_popup_removes_variable_callbacks_when_closed(hidden_tk):
    popup = module().FourBandReviewWindow(first_model(), initial=module().ReviewSettings(n_clusters=2))
    def interact():
        manual = popup.variables["manual"]
        assert manual.trace_info()
        popup.cancel()
        assert not manual.trace_info()
    hidden_tk["run"] = interact
    assert popup.show() is None
