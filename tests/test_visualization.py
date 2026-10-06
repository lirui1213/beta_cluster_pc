from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import visualization
from visualization import load_electrode_coordinates
from visualization import plot_beta_cluster_grid
from visualization import render_electrode_clusters
from visualization import render_selectable_electrode_class
from visualization import show_pyvista_window


def test_load_electrode_coordinates_supports_recon_grid_export(tmp_path):
    csv_path = tmp_path / "electrodes.csv"
    pd.DataFrame(
        {
            "component name": ["", ""],
            "grid label": ["PMT", "PMT"],
            "grid id": [1, 2],
            "original location.x": [-1.0, -2.0],
            "original location.y": [1.0, 2.0],
            "original location.z": [3.0, 4.0],
            "registered location.x": [-10.0, -20.0],
            "registered location.y": [10.0, 20.0],
            "registered location.z": [30.0, 40.0],
        }
    ).to_csv(csv_path, index=False)

    coordinates = load_electrode_coordinates(csv_path)

    assert coordinates["channel"].tolist() == ["ECoG_1", "ECoG_2"]
    assert coordinates[["x", "y", "z"]].to_numpy().tolist() == [
        [-10.0, 10.0, 30.0],
        [-20.0, 20.0, 40.0],
    ]


def test_show_pyvista_window_forces_desktop_backend():
    class FakeTheme:
        notebook = True

    class FakePyVista:
        global_theme = FakeTheme()

        def __init__(self):
            self.backends = []

        def set_jupyter_backend(self, backend):
            self.backends.append(backend)

    class FakePlotter:
        def __init__(self):
            self.calls = []
            self.camera_position = None
            self.axes_added = False
            self.trackball_enabled = False
            self.camera_reset = False

        def add_axes(self):
            self.axes_added = True

        def enable_trackball_style(self):
            self.trackball_enabled = True

        def reset_camera(self):
            self.camera_reset = True

        def show(self, **kwargs):
            self.calls.append(kwargs)
            return "shown"

    pv_module = FakePyVista()
    plotter = FakePlotter()

    result = show_pyvista_window(plotter, pv_module=pv_module)

    assert result == "shown"
    assert pv_module.global_theme.notebook is False
    assert pv_module.backends == [None]
    assert plotter.camera_position == "xy"
    assert plotter.axes_added
    assert plotter.trackball_enabled
    assert plotter.camera_reset
    assert plotter.calls == [{"notebook": False, "interactive": True, "auto_close": False}]


def test_show_pyvista_window_saves_screenshot_after_show(tmp_path):
    class FakeTheme:
        notebook = True

    class FakePyVista:
        global_theme = FakeTheme()

        def set_jupyter_backend(self, _backend):
            pass

    class FakePlotter:
        def __init__(self):
            self.events = []
            self.shown = False

        def add_axes(self):
            pass

        def enable_trackball_style(self):
            pass

        def reset_camera(self):
            pass

        def show(self, **_kwargs):
            self.events.append("show")
            self.shown = True

        def screenshot(self, filename, return_img=False):
            if not self.shown:
                raise RuntimeError("Nothing to screenshot - call .show first")
            self.events.append((filename, return_img))

    plotter = FakePlotter()
    screenshot_path = tmp_path / "brain.png"

    show_pyvista_window(plotter, pv_module=FakePyVista(), screenshot=screenshot_path)

    assert plotter.events == ["show", (str(screenshot_path), False)]


def test_linked_surface_pumps_ui_outside_native_vtk_callbacks():
    events = []
    callbacks = {}

    class Interactor:
        def AddObserver(self, event, callback):
            callbacks[event] = callback
            return event
        def RemoveObserver(self, event):
            callbacks.pop(event, None)
        def TerminateApp(self):
            pass

    class Plotter:
        _closed = False
        inside_native = False
        def __init__(self):
            self.iren = SimpleNamespace(interactor=Interactor(), process_events=self.native_events)
        def reset_camera(self):
            pass
        def add_axes(self):
            pass
        def close(self):
            self._closed = True
        def show(self, **kwargs):
            assert kwargs.get("interactive_update") is True, "Linked UI must not enter the blocking VTK Start loop."
        def update(self, **kwargs):
            self.native_events()
            self.render()
        def render(self):
            assert "ExitEvent" in callbacks, "Never render after native ExitEvent finalized the window."
        def native_events(self):
            self.inside_native = True
            events.append("vtk")
            if events.count("vtk") == 3:
                callbacks["ExitEvent"](self.iren.interactor, "ExitEvent")
            self.inside_native = False

    plotter = Plotter()
    def pump():
        assert not plotter.inside_native
        events.append("tk")
    pv = SimpleNamespace(global_theme=SimpleNamespace(notebook=True), set_jupyter_backend=lambda _: None)
    show_pyvista_window(plotter, pv_module=pv, on_window_tick=pump)
    assert events[0:2] == ["vtk", "tk"]
    assert events.count("vtk") == 3 and events[-1] == "vtk"


def test_linked_native_close_drops_invalid_window_before_cached_screenshot(tmp_path):
    events = []
    class Window:
        current = True
        def IsCurrent(self):
            return self.current
    class Plotter:
        _closed = False
        def __init__(self):
            self.render_window = Window()
            interactor = SimpleNamespace(AddObserver=lambda *_: 1, RemoveObserver=lambda *_: None)
            self.iren = SimpleNamespace(interactor=interactor, process_events=self.native_close)
        def reset_camera(self):
            pass
        def add_axes(self):
            pass
        def show(self, **kwargs):
            events.append("show")
        def update(self, **kwargs):
            raise AssertionError("Original update would render the destroyed window")
        def native_close(self):
            events.append("native_close")
            self.render_window.current = False
        def render(self):
            raise AssertionError("Destroyed OpenGL window must not be rendered")
        def _clear_ren_win(self):
            events.append("drop_window")
            self.render_window = None
        def close(self):
            events.append("close")
            self._closed = True
        def screenshot(self, *args, **kwargs):
            assert self._closed and self.render_window is None
            events.append("cached_screenshot")
    pv = SimpleNamespace(global_theme=SimpleNamespace(notebook=True), set_jupyter_backend=lambda _: None)
    show_pyvista_window(Plotter(), pv_module=pv, on_window_tick=lambda: None,
                       screenshot=tmp_path / "brain.png")
    assert events == ["show", "native_close", "drop_window", "close", "cached_screenshot"]


def test_failed_nonblocking_show_restores_original_update():
    class Plotter:
        def reset_camera(self):
            pass
        def add_axes(self):
            pass
        def close(self):
            pass
        def update(self, **kwargs):
            pass
        def show(self, **kwargs):
            raise RuntimeError("native initialization failed")
    plotter = Plotter()
    original = plotter.update
    pv = SimpleNamespace(global_theme=SimpleNamespace(notebook=True), set_jupyter_backend=lambda _: None)
    with pytest.raises(RuntimeError, match="initialization failed"):
        show_pyvista_window(plotter, pv_module=pv, interactive_update=True, on_window_tick=lambda: None)
    assert plotter.update == original


def test_render_electrode_clusters_forwards_display_style(monkeypatch, tmp_path):
    class FakeCloud:
        def __init__(self, coordinates):
            self.coordinates = coordinates
            self.data = {}

        def __setitem__(self, key, value):
            self.data[key] = value

    class FakePlotter:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.meshes = []
            self.points = []
            self.background = None
            self.camera_position = None
            FakePlotter.instances.append(self)

        def add_mesh(self, mesh, **kwargs):
            self.meshes.append((mesh, kwargs))

        def add_points(self, cloud, **kwargs):
            self.points.append((cloud, kwargs))

        def set_background(self, color):
            self.background = color

        def screenshot(self, *_args, **_kwargs):
            pass

    class FakeTheme:
        notebook = True

    class FakePyVista:
        global_theme = FakeTheme()
        Plotter = FakePlotter
        PolyData = FakeCloud

    monkeypatch.setitem(sys.modules, "pyvista", FakePyVista)
    monkeypatch.setattr(visualization, "_load_pial_mesh", lambda _path: "brain")
    monkeypatch.setattr(visualization, "_load_tumor_mesh", lambda *_args, **_kwargs: "tumor")
    monkeypatch.setattr(
        visualization,
        "scanner_ras_to_tkras",
        lambda coordinates, _reference_mri: coordinates,
    )

    csv_path = tmp_path / "electrodes.csv"
    pd.DataFrame(
        {
            "channel": ["ECoG_1"],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
        }
    ).to_csv(csv_path, index=False)
    subject_root = tmp_path / "subject"
    (subject_root / "surf").mkdir(parents=True)
    (subject_root / "surf" / "lh.pial").write_text("", encoding="utf-8")
    reference_mri = subject_root / "mri" / "T1.mgz"
    reference_mri.parent.mkdir()
    reference_mri.write_text("", encoding="utf-8")

    render_electrode_clusters(
        csv_path,
        {"ECoG_1": 0},
        subjects_dir=tmp_path,
        subject="subject",
        reference_mri=reference_mri,
        hemispheres=("lh",),
        tumor_nifti=tmp_path / "tumor.nii",
        point_size=22.0,
        window_size=(500, 400),
        brain_color="ivory",
        brain_opacity=0.7,
        brain_specular=0.4,
        tumor_color="crimson",
        tumor_opacity=0.8,
        background_color="navy",
        show=False,
    )

    plotter = FakePlotter.instances[-1]
    assert plotter.kwargs["window_size"] == [500, 400]
    assert plotter.meshes[0] == (
        "brain",
        {
            "color": "ivory",
            "opacity": 0.7,
            "smooth_shading": True,
            "specular": 0.4,
        },
    )
    assert plotter.meshes[1] == (
        "tumor",
        {"color": "crimson", "opacity": 0.8, "smooth_shading": True},
    )
    assert plotter.points[0][1]["point_size"] == 22.0
    assert plotter.background == "navy"


def test_render_selectable_electrode_class_adds_popup_controls(monkeypatch, tmp_path):
    class FakeProperty:
        def __init__(self):
            self.color = None

        def SetColor(self, red, green, blue):
            self.color = (red, green, blue)

    class FakeActor:
        def __init__(self):
            self.visible = None
            self.property = FakeProperty()

        def SetVisibility(self, visible):
            self.visible = bool(visible)

        def GetProperty(self):
            return self.property

    class FakePlotter:
        instances = []

        def __init__(self, **_kwargs):
            self.meshes = []
            self.sliders = []
            self.text = []
            self.background = None
            self.render_count = 0
            FakePlotter.instances.append(self)

        def add_mesh(self, mesh, **kwargs):
            actor = FakeActor()
            actor.mesh = mesh
            actor.kwargs = kwargs
            self.meshes.append(actor)
            return actor

        def add_slider_widget(self, callback, rng, value=None, title=None, **kwargs):
            self.sliders.append(
                {"callback": callback, "range": rng, "value": value, "title": title, "kwargs": kwargs}
            )

        def add_text(self, text, **kwargs):
            self.text.append((text, kwargs))

        def set_background(self, color):
            self.background = color

        def render(self):
            self.render_count += 1

    class FakeTheme:
        notebook = True

    class FakePyVista:
        global_theme = FakeTheme()
        Plotter = FakePlotter

        @staticmethod
        def Sphere(**kwargs):
            return ("sphere", kwargs)

    monkeypatch.setitem(sys.modules, "pyvista", FakePyVista)

    csv_path = tmp_path / "electrodes.csv"
    pd.DataFrame(
        {
            "channel": ["ECoG_1", "ECoG_2", "ECoG_3"],
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "z": [1.0, 2.0, 3.0],
        }
    ).to_csv(csv_path, index=False)

    render_selectable_electrode_class(
        csv_path,
        {"ECoG_1": 0, "ECoG_2": 1, "ECoG_3": 1},
        selected_label=0,
        selected_color=(1.0, 0.0, 0.0),
        show=False,
    )

    plotter = FakePlotter.instances[-1]
    electrode_actors = plotter.meshes[-3:]
    assert [actor.visible for actor in electrode_actors] == [True, False, False]
    assert len(plotter.sliders) == 4

    plotter.sliders[0]["callback"](1.0)
    assert [actor.visible for actor in electrode_actors] == [False, True, True]

    plotter.sliders[2]["callback"](0.5)
    assert electrode_actors[1].property.color == (1.0, 0.5, 0.0)
    assert electrode_actors[2].property.color == (1.0, 0.5, 0.0)


def test_render_selectable_electrode_class_uses_compact_slider_controls(monkeypatch, tmp_path):
    class FakeActor:
        def SetVisibility(self, _visible):
            pass

        def GetProperty(self):
            return None

    class FakePlotter:
        instances = []

        def __init__(self, **_kwargs):
            self.sliders = []
            FakePlotter.instances.append(self)

        def add_mesh(self, *_args, **_kwargs):
            return FakeActor()

        def add_slider_widget(self, _callback, _rng, value=None, title=None, **kwargs):
            self.sliders.append({"value": value, "title": title, "kwargs": kwargs})

        def add_text(self, *_args, **_kwargs):
            pass

        def set_background(self, _color):
            pass

        def render(self):
            pass

    class FakeTheme:
        notebook = True

    class FakePyVista:
        global_theme = FakeTheme()
        Plotter = FakePlotter

        @staticmethod
        def Sphere(**kwargs):
            return ("sphere", kwargs)

    monkeypatch.setitem(sys.modules, "pyvista", FakePyVista)

    csv_path = tmp_path / "electrodes.csv"
    pd.DataFrame(
        {
            "channel": ["ECoG_1"],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
        }
    ).to_csv(csv_path, index=False)

    render_selectable_electrode_class(
        csv_path,
        {"ECoG_1": 0},
        slider_length=0.18,
        slider_width=0.015,
        slider_tube_width=0.004,
        slider_title_height=0.018,
        show=False,
    )

    slider_kwargs = FakePlotter.instances[-1].sliders[0]["kwargs"]
    assert slider_kwargs["pointa"] == (0.02, 0.93)
    assert slider_kwargs["pointb"] == pytest.approx((0.2, 0.93))
    assert slider_kwargs["slider_width"] == 0.015
    assert slider_kwargs["tube_width"] == 0.004
    assert slider_kwargs["title_height"] == 0.018


def test_plot_beta_cluster_grid_smooths_display_curves():
    epochs = SimpleNamespace(
        pre=np.array([[[0.0, 0.0, 0.0]]]),
        speech=np.array([[[0.0, 10.0, 0.0, 0.0, 0.0]]]),
        post=np.array([[[0.0, 0.0, 0.0]]]),
        audio_pre=np.zeros((1, 3)),
        audio_speech=np.array([[0.0, 1.0, 0.0, 0.0, 0.0]]),
        audio_post=np.zeros((1, 3)),
        progress_percent=np.linspace(0.0, 100.0, 5),
        pre_times=np.array([-0.2, -0.1, 0.0]),
        post_times=np.array([0.0, 0.1, 0.2]),
        events=pd.DataFrame({"duration": [0.4]}),
        channel_names=("ECoG_1",),
        sfreq=10.0,
        pre_post_window=0.2,
    )

    figure, axes = plot_beta_cluster_grid(epochs, [0], sigma_seconds=0.1)

    y_data = axes[0, 0].lines[0].get_ydata()
    assert np.nanmax(y_data) < 10.0
    assert y_data[2] > 0.0
    figure.clf()
