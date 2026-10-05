from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import visualization


RENDERERS = (
    visualization.render_electrode_clusters,
    visualization.render_selectable_electrode_class,
)


@pytest.fixture
def electrode_csv(tmp_path):
    path = tmp_path / "electrodes.csv"
    pd.DataFrame(
        {
            "grid id": [42, 7, 101],
            "x": [-10., 0., 10.],
            "y": [0., 0., 0.],
            "z": [0., 0., 0.],
        }
    ).to_csv(path, index=False)
    return path


def render(renderer, csv_path, **kwargs):
    return renderer(
        csv_path, [3, -1, 3],
        channel_order=["ECoG_7", "ECoG_101", "ECoG_42"], **kwargs,
    )


@pytest.fixture
def fake_pyvista(monkeypatch):
    class Actor:
        def __init__(self, **data):
            self.visible = True
            self.color = data.get("color")
            self.__dict__.update(data)

        def SetVisibility(self, visible):
            self.visible = bool(visible)

        def GetProperty(self):
            return self

        def SetColor(self, *color):
            self.color = color

    class Cloud(dict):
        def __init__(self, points):
            super().__init__()
            self.points = np.array(points, copy=True)

    class Plotter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.electrodes = []
            self.label_actors = []
            self.active_labels = []
            self.checkboxes = []
            self.sliders = []
            self.text = []
            self.events = []
            self.camera_position = None
            self.render_count = 0

        def add_mesh(self, mesh, **kwargs):
            actor = Actor(mesh=mesh, **kwargs)
            self.electrodes.append(actor)
            return actor

        def add_points(self, cloud, **kwargs):
            return self.add_mesh(cloud, **kwargs)

        def add_point_labels(self, points, labels, **kwargs):
            assert kwargs["reset_camera"] is False
            assert kwargs["show_points"] is False
            assert kwargs["render"] is False
            actor = Actor(
                points=np.array(points, copy=True), labels=list(labels), **kwargs,
            )
            self.label_actors.append(actor)
            self.active_labels.append(actor)
            return actor

        def remove_actor(self, actor, **kwargs):
            assert kwargs["reset_camera"] is False
            assert kwargs["render"] is False
            self.active_labels.remove(actor)

        def add_checkbox_button_widget(self, callback, **kwargs):
            self.checkboxes.append((callback, kwargs))
            callback(kwargs.get("value", False))

        def add_slider_widget(self, callback, rng, **kwargs):
            self.sliders.append(callback)
            callback(kwargs["value"])

        def add_text(self, text, **kwargs):
            self.text.append((text, kwargs))

        def set_background(self, color):
            self.background = color

        def render(self):
            self.render_count += 1

        def reset_camera(self):
            self.events.append("reset_camera")

        def add_axes(self):
            pass

        def enable_trackball_style(self):
            pass

        def show(self, **kwargs):
            self.events.append(("show", kwargs))

        def screenshot(self, filename, **kwargs):
            self.events.append(("screenshot", filename))

    module = SimpleNamespace(
        Plotter=Plotter, PolyData=Cloud, Sphere=lambda **kwargs: kwargs,
        global_theme=SimpleNamespace(notebook=True), set_jupyter_backend=lambda _: None,
    )
    monkeypatch.setitem(sys.modules, "pyvista", module)
    return module


def electrode_state(plotter):
    return [(actor.visible, actor.color) for actor in plotter.electrodes]


@pytest.mark.parametrize("renderer", RENDERERS)
def test_defaults_do_not_add_channel_controls(renderer, electrode_csv, fake_pyvista):
    plotter = render(renderer, electrode_csv, show=False)
    assert plotter.checkboxes == []
    assert plotter.label_actors == []


@pytest.mark.parametrize("renderer", RENDERERS)
def test_checkbox_reveals_csv_identity_without_changing_electrodes(
    renderer, electrode_csv, fake_pyvista,
):
    plotter = render(renderer, electrode_csv, show=False, channel_label_toggle=True)
    assert plotter.label_actors == []
    callback, options = plotter.checkboxes[0]
    assert options["value"] is False
    assert options["size"] <= 24
    assert any(text == "Channel labels" for text, _ in plotter.text)
    before = electrode_state(plotter)
    plotter.camera_position = "user-adjusted-camera"

    callback(True)
    actor = plotter.active_labels[0]
    expected = (
        ["ECoG_42", "ECoG_7", "ECoG_101"]
        if renderer is RENDERERS[0] else ["ECoG_42", "ECoG_7"]
    )
    assert actor.labels == expected
    np.testing.assert_array_equal(
        actor.points[:, 0], [-10., 0., 10.] if len(expected) == 3 else [-10., 0.],
    )
    callback(False)
    assert actor.visible is False
    callback(True)
    assert plotter.active_labels == [actor]
    assert actor.visible is True
    assert len(plotter.label_actors) == 1
    assert electrode_state(plotter) == before
    assert plotter.camera_position == "user-adjusted-camera"
    assert "reset_camera" not in plotter.events


@pytest.mark.parametrize("renderer", RENDERERS)
@pytest.mark.parametrize("toggle", [False, True])
def test_initial_labels_are_explicit_opt_in(renderer, toggle, electrode_csv, fake_pyvista):
    plotter = render(
        renderer, electrode_csv, show=False,
        channel_label_toggle=toggle, show_channel_labels=True,
    )
    assert len(plotter.active_labels) == 1
    assert plotter.active_labels[0].visible is True
    assert len(plotter.checkboxes) == int(toggle)
    if toggle:
        assert plotter.checkboxes[0][1]["value"] is True


@pytest.mark.parametrize("renderer", RENDERERS)
def test_channel_names_stay_attached_to_transformed_csv_coordinates(
    renderer, fake_pyvista, monkeypatch, tmp_path,
):
    path = tmp_path / "named.csv"
    pd.DataFrame(
        {"channel": ["Grid_A09", "Strip-B2"], "x": [2., 4.],
         "y": [1., 3.], "z": [5., 6.]}
    ).to_csv(path, index=False)
    reference = tmp_path / "reference.mgz"
    reference.touch()
    monkeypatch.setattr(
        visualization, "scanner_ras_to_tkras",
        lambda coordinates, _: coordinates + [100., 200., 300.],
    )
    plotter = renderer(
        path, {"Strip-B2": 3, "Grid_A09": 3},
        reference_mri=reference, show=False, show_channel_labels=True,
    )
    assert plotter.active_labels[0].labels == ["Grid_A09", "Strip-B2"]
    np.testing.assert_array_equal(
        plotter.active_labels[0].points, [[102., 201., 305.], [104., 203., 306.]],
    )


def test_initial_selectable_labels_follow_class_without_checkbox(electrode_csv, fake_pyvista):
    plotter = render(RENDERERS[1], electrode_csv, show=False, show_channel_labels=True)
    assert plotter.checkboxes == []
    plotter.sliders[0](0.)
    assert len(plotter.active_labels) == 1
    assert plotter.active_labels[0].labels == ["ECoG_101"]


def test_selectable_labels_follow_class_switches_while_on_and_off(electrode_csv, fake_pyvista):
    plotter = render(RENDERERS[1], electrode_csv, show=False, channel_label_toggle=True)
    toggle = plotter.checkboxes[0][0]
    select = plotter.sliders[0]
    toggle(True)
    first = plotter.active_labels[0]
    select(0.)
    assert plotter.active_labels[0].labels == ["ECoG_101"]
    assert first not in plotter.active_labels
    assert [actor.visible for actor in plotter.electrodes] == [False, False, True]
    assert len(plotter.active_labels) == 1
    current = plotter.active_labels[0]
    plotter.sliders[2](0.5)
    assert plotter.active_labels == [current]
    assert plotter.electrodes[2].color == (1., 0.5, 0.)

    toggle(False)
    select(1.)
    assert not any(actor.visible for actor in plotter.active_labels)
    toggle(True)
    assert plotter.active_labels[0].labels == ["ECoG_42", "ECoG_7"]
    assert len(plotter.active_labels) == 1
    assert [actor.visible for actor in plotter.electrodes] == [True, True, False]
    select(1.)
    assert len(plotter.label_actors) == 3


@pytest.mark.parametrize("renderer", RENDERERS)
def test_channel_control_preserves_native_popup_lifecycle(
    renderer, electrode_csv, fake_pyvista, tmp_path,
):
    kwargs = {"screenshot": tmp_path / "surface.png"} if renderer is RENDERERS[0] else {}
    plotter = render(renderer, electrode_csv, show=True, channel_label_toggle=True, **kwargs)
    assert plotter.kwargs["off_screen"] is False
    assert plotter.kwargs["notebook"] is False
    assert plotter.events[:2] == [
        "reset_camera", ("show", {"notebook": False, "interactive": True, "auto_close": False}),
    ]
    if kwargs:
        assert plotter.events[2] == ("screenshot", str(kwargs["screenshot"]))
    plotter.checkboxes[0][0](True)
    assert plotter.events.count("reset_camera") == 1


@pytest.mark.parametrize("renderer", RENDERERS)
def test_real_offscreen_labels_preserve_camera_and_meshes(
    renderer, electrode_csv, monkeypatch, tmp_path,
):
    pv = pytest.importorskip("pyvista")

    class OffscreenPlotter(pv.Plotter):
        label_callback = None
        class_callback = None

        def add_checkbox_button_widget(self, callback, **kwargs):
            self.label_callback = callback
            return super().add_checkbox_button_widget(callback, **kwargs)

        def add_slider_widget(self, callback, rng, **kwargs):
            if kwargs.get("title") == "Cluster":
                self.class_callback = callback
            return super().add_slider_widget(callback, rng, **kwargs)

    monkeypatch.setattr(pv, "Plotter", OffscreenPlotter)
    kwargs = {"screenshot": tmp_path / "initial.png"} if renderer is RENDERERS[0] else {}
    plotter = render(
        renderer, electrode_csv, show=False, channel_label_toggle=True,
        window_size=(600, 400), **kwargs,
    )
    try:
        if kwargs:
            assert kwargs["screenshot"].stat().st_size > 0
        plotter.reset_camera()
        plotter.camera.azimuth = 12
        camera = tuple(plotter.camera_position)
        actors = [
            actor for actor in plotter.actors.values() if isinstance(actor, pv.Actor)
        ]
        state = [
            (actor.GetVisibility(), actor.prop.color, actor.GetBounds()) for actor in actors
        ]
        plotter.render()
        off = plotter.screenshot()
        plotter.label_callback(True)
        on = plotter.screenshot(tmp_path / "labels.png")
        assert on.shape == (400, 600, 3)
        assert np.count_nonzero(on != off) > 30
        assert np.count_nonzero(off != 255) > 30
        np.testing.assert_allclose(tuple(plotter.camera_position), camera)
        assert [
            (actor.GetVisibility(), actor.prop.color, actor.GetBounds()) for actor in actors
        ] == state
        label_actors = [
            actor for actor in plotter.actors.values()
            if actor.IsA("vtkActor2D") and actor.GetMapper() is not None
            and actor.GetMapper().IsA("vtkLabelPlacementMapper")
        ]
        assert len(label_actors) == 1
        actor = label_actors[0]
        points = actor.GetMapper().GetInputAlgorithm().GetInput()
        names = points.GetPointData().GetAbstractArray("labels")
        actual_names = [names.GetValue(i) for i in range(names.GetNumberOfValues())]
        assert actual_names == (
            ["ECoG_42", "ECoG_7", "ECoG_101"]
            if renderer is RENDERERS[0] else ["ECoG_42", "ECoG_7"]
        )
        plotter.label_callback(False)
        assert not actor.GetVisibility()
        np.testing.assert_array_equal(plotter.screenshot(), off)
        plotter.label_callback(True)
        if renderer is RENDERERS[1]:
            plotter.class_callback(0.)
            current = [
                actor for actor in plotter.actors.values()
                if actor.IsA("vtkActor2D") and actor.GetMapper() is not None
                and actor.GetMapper().IsA("vtkLabelPlacementMapper")
            ]
            assert len(current) == 1
            points = current[0].GetMapper().GetInputAlgorithm().GetInput()
            names = points.GetPointData().GetAbstractArray("labels")
            assert names.GetNumberOfValues() == 1
            assert names.GetValue(0) == "ECoG_101"
            np.testing.assert_allclose(tuple(plotter.camera_position), camera)
    finally:
        plotter.close()
