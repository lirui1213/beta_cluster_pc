"""Exercise native VTK actor/point identity without requiring Qt."""
import numpy as np
import pandas as pd
import pytest

import visualization


RENDERERS = (visualization.render_electrode_clusters,
             visualization.render_selectable_electrode_class)


@pytest.fixture
def surface(tmp_path):
    pv = pytest.importorskip("pyvista")
    path = tmp_path / "electrodes.csv"
    pd.DataFrame(dict(channel=["Grid_A09", "Strip-B2", "ECoG_101"],
        x=[-10., 0., 10.], y=[0.] * 3, z=[0.] * 3)).to_csv(path, index=False)
    return pv, path


def pick(plotter, coordinate):
    renderer = plotter.renderer
    renderer.SetWorldPoint(*coordinate, 1.)
    renderer.WorldToDisplay()
    x, y, _ = renderer.GetDisplayPoint()
    plotter.iren.picker.Pick(x, y, 0, renderer)


def make_plotter(renderer, path, callback, **kwargs):
    return renderer(path, [3, -1, 3],
        channel_order=["Strip-B2", "ECoG_101", "Grid_A09"],
        on_channel_pick=callback, show=False, show_channel_labels=True,
        window_size=(400, 300), **kwargs)


@pytest.mark.parametrize("renderer", RENDERERS)
def test_native_pick_returns_actual_csv_channel_preserving_scene(renderer, surface):
    _, path = surface
    selected = []
    plotter = make_plotter(renderer, path, selected.append)
    try:
        plotter.show(auto_close=False, interactive=False)
        camera = tuple(plotter.camera_position)
        actors = tuple(plotter.renderer.actors.values())
        pick(plotter, [-10., 0., 0.])
        pick(plotter, [0., 0., 0.])
        assert selected == ["Grid_A09", "Strip-B2"]
        assert tuple(plotter.camera_position) == camera
        assert tuple(plotter.renderer.actors.values()) == actors
    finally:
        plotter.close()


def test_class_visibility_and_sphere_local_point_ids_use_actor_identity(surface):
    _, path = surface
    selected = []
    plotter = make_plotter(RENDERERS[1], path, selected.append)
    try:
        plotter.show(auto_close=False, interactive=False)
        pick(plotter, [10., 0., 0.])
        assert selected == []
        # Slider indexes sorted actual labels (-1, 3), not CSV row numbers.
        plotter.slider_widgets[0].GetRepresentation().SetValue(0.)
        plotter.slider_widgets[0].InvokeEvent("EndInteractionEvent")
        plotter.render()
        pick(plotter, [10., 0., 0.])
        pick(plotter, [-10., 0., 0.])
        assert selected == ["ECoG_101"]
    finally:
        plotter.close()


@pytest.mark.parametrize("renderer", RENDERERS)
def test_brain_background_and_invalid_point_are_ignored(renderer, surface):
    pv, path = surface
    selected = []
    plotter = make_plotter(renderer, path, selected.append)
    try:
        brain = plotter.add_mesh(pv.Sphere(center=(0., 8., 0.), radius=2.))
        plotter.show(auto_close=False, interactive=False)
        pick(plotter, [0., 8., 0.])
        plotter.iren.picker.Pick(0, 0, 0, plotter.renderer)
        assert selected == []
        assert brain.GetVisibility()
    finally:
        plotter.close()


def test_callback_exception_is_reported_and_close_disconnects(surface):
    _, path = surface

    def broken(channel):
        raise RuntimeError("explorer closed")

    plotter = make_plotter(RENDERERS[0], path, broken)
    plotter.show(auto_close=False, interactive=False)
    picker = plotter.iren.picker
    with pytest.warns(UserWarning, match="channel.*explorer closed"):
        pick(plotter, [-10., 0., 0.])
    plotter.close()
    # No user callback or VTK observers survive programmatic close.
    assert not picker.HasObserver("EndPickEvent")
    picker.InvokeEvent("EndPickEvent")
    plotter.close()


def test_native_window_exit_disconnects_callbacks_before_plotter_close(surface):
    _, path = surface
    selected = []
    plotter = make_plotter(RENDERERS[0], path, selected.append)
    try:
        plotter.show(auto_close=False, interactive=False)
        picker = plotter.iren.picker
        plotter.iren.interactor.InvokeEvent("ExitEvent")
        assert not picker.HasObserver("EndPickEvent")
        picker.InvokeEvent("EndPickEvent")
        assert selected == []
    finally:
        plotter.close()


@pytest.mark.parametrize("renderer", RENDERERS)
def test_opt_in_nonblocking_show_for_tk_event_pump(renderer, surface, monkeypatch):
    _, path = surface
    options = []

    def show(plotter, **kwargs):
        options.append(kwargs)

    monkeypatch.setattr(visualization, "show_pyvista_window", show)
    plotter = renderer(path, {"Grid_A09": 3, "Strip-B2": 3, "ECoG_101": -1},
        on_channel_pick=lambda _: None, show=True, interactive_update=True)
    try:
        assert options[0]["interactive_update"] is True
    finally:
        plotter.close()


def test_nonblocking_helper_uses_current_native_show_api():
    calls = []

    class Plotter:
        def reset_camera(self):
            pass

        def add_axes(self):
            pass

        def show(self, *, interactive, auto_close, interactive_update):
            calls.append((interactive, auto_close, interactive_update))

    visualization.show_pyvista_window(Plotter(), pv_module=object(), interactive_update=True)
    assert calls == [(True, False, True)]


@pytest.mark.parametrize("closes_inside_show", [False, True])
def test_window_tick_timer_hookup_and_cleanup(closes_inside_show):
    ticks, events = [], []

    class Interactor:
        def initialize(self):
            events.append("initialize")

        def add_observer(self, name, callback):
            assert name == "TimerEvent"
            self.callback = callback
            return 23

        def create_timer(self, duration, repeating=True):
            assert duration == 50 and repeating
            events.append("timer")
            return 19

        def GetTimerEventId(self):
            # Native Win32 TimerEvent passes its ID as event calldata instead.
            return 0

        def destroy_timer(self, timer_id):
            assert timer_id == 19
            events.append("destroy")

        def remove_observer(self, observer_id):
            assert observer_id == 23
            self.callback = None
            events.append("remove")

    class Plotter:
        def __init__(self):
            self.iren = Interactor()

        def reset_camera(self):
            pass

        def add_axes(self):
            pass

        def show(self, **kwargs):
            assert kwargs == dict(notebook=False, interactive=True, auto_close=False)
            self.iren.callback(self.iren, "TimerEvent")
            if closes_inside_show:
                self.close()

        def close(self):
            events.append("close")

    plotter = Plotter()
    visualization.show_pyvista_window(plotter, pv_module=object(),
        on_window_tick=lambda: ticks.append("pump"))
    assert ticks == ["pump"]
    assert events.count("destroy") == events.count("remove") == 1
    assert plotter.iren.callback is None
    plotter.close()
    assert events.count("destroy") == events.count("remove") == 1


@pytest.mark.parametrize("renderer", RENDERERS)
def test_renderer_forwards_opt_in_window_tick(renderer, surface, monkeypatch):
    _, path = surface
    captured = []
    pump = lambda: None
    monkeypatch.setattr(visualization, "show_pyvista_window",
        lambda plotter, **kwargs: captured.append(kwargs))
    plotter = renderer(path, {"Grid_A09": 0, "Strip-B2": 0, "ECoG_101": 1},
        on_window_tick=pump, show=True)
    try:
        assert captured[0]["on_window_tick"] is pump
    finally:
        plotter.close()


def test_unsupported_timer_warns_and_keeps_surface_window_usable():
    shown = []

    class Plotter:
        def reset_camera(self):
            pass

        def add_axes(self):
            pass

        def show(self, **kwargs):
            shown.append(kwargs)

    with pytest.warns(UserWarning, match="tick is unavailable"):
        visualization.show_pyvista_window(Plotter(), pv_module=object(), on_window_tick=lambda: None)
    assert shown == [dict(notebook=False, interactive=True, auto_close=False)]
