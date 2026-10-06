"""Diagnostic-only native interaction/close probes around the real GUI worker."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "3-bands"), str(ROOT)]

if __name__ == "__mp_main__":
    from fourband.display_logging import configure_numerical_logging
    configure_numerical_logging()

import pyvista as pv
import visualization
from fourband import display_worker as worker


directory = Path(sys.argv[1]).resolve().parent
report = {"worker_pid": os.getpid(), "surfaces": [], "picks": [], "curve_draws": 0, "numerical_pids": []}
explorers = []
native_show = visualization.show_pyvista_window
native_init = worker.FourBandExplorer.__init__
native_draw = worker.FourBandExplorer._draw_continuous
native_pick = worker.FourBandExplorer.select_channel
native_explorer_show = worker.FourBandExplorer.show


def remember(self, *args, **kwargs):
    native_init(self, *args, **kwargs)
    explorers.append(self)


def drawn(self, data):
    native_draw(self, data)
    report["curve_draws"] += 1
    report["numerical_pids"].append(data.metadata["numerical_pid"])
    self.figure.savefig(directory / "verified_explorer.png")


def picked(self, channel):
    report["picks"].append(channel)
    native_pick(self, channel)


def explorer_show(self, block=True):
    if not block:
        return native_explorer_show(self, block=False)
    native_explorer_show(self, block=False)
    started = time.monotonic()
    selected = [False]
    errors = []
    def check():
        if self._render_valid and self._future is None:
            if not selected[0]:
                selected[0] = True
                self.variables["channel"].set(self.state.result.channels[1])
                self._changed("channel")
            else:
                self.save()
                report["standalone_complete"] = True
                self.close()
                return
        if time.monotonic() - started > 180:
            errors.append("Standalone explorer did not finish in 180 seconds")
            self.close()
            return
        self.root.after(25, check)
    self.root.after(25, check)
    try:
        self.root.mainloop()
    finally:
        self.close()
    if errors:
        raise RuntimeError(errors[0])


def probed_show(plotter, **kwargs):
    explorer = explorers[-1] if explorers else None
    ticks = [0]
    closed = [False]
    picked_once = [False]
    label_checked = [False]
    errors = []
    started = time.monotonic()
    linked_tick = kwargs["on_window_tick"]

    def probe():
        linked_tick()
        ticks[0] += 1
        if closed[0]:
            return
        # The native callback does not call Tk; it queues a channel request.
        if explorer is not None and not picked_once[0] and explorer._render_valid:
            picked_once[0] = True
            picker = plotter.iren.picker
            actors = picker.GetPickList()
            for index in range(actors.GetNumberOfItems()):
                actor = actors.GetItemAsObject(index)
                if not actor.GetVisibility():
                    continue
                points = actor.GetMapper().GetInput()
                point = points.GetPoint(min(1, points.GetNumberOfPoints() - 1))
                renderer = plotter.renderer
                renderer.SetWorldPoint(*point, 1.)
                renderer.WorldToDisplay()
                x, y, _ = renderer.GetDisplayPoint()
                before = len(report["picks"])
                picker.Pick(x, y, 0, renderer)
                if len(report["picks"]) > before:
                    break
            else:
                errors.append("Native electrode picking did not select a channel")
        if not label_checked[0]:
            label_checked[0] = True
            for widget in getattr(plotter, "button_widgets", []):
                representation = widget.GetRepresentation()
                representation.SetState(1)
                widget.InvokeEvent("StateChangedEvent")
        ready = explorer is None or (explorer._future is None and explorer._render_valid)
        if time.monotonic() - started > 180:
            errors.append("Native explorer did not finish in 180 seconds")
            ready = True
        if time.monotonic() - started > 3 and ready and (picked_once[0] or explorer is None):
            closed[0] = True
            report["surfaces"].append(dict(ticks=ticks[0], curves_ready=ready, picked=picked_once[0]))
            plotter.render_window.Finalize()
            plotter._clear_ren_win()
            plotter.close()

    kwargs["on_window_tick"] = probe
    result = native_show(plotter, **kwargs)
    if errors:
        raise RuntimeError(errors[0])
    return result


worker.FourBandExplorer.__init__ = remember
worker.FourBandExplorer._draw_continuous = drawn
worker.FourBandExplorer.select_channel = picked
worker.FourBandExplorer.show = explorer_show
visualization.show_pyvista_window = probed_show
if __name__ == "__main__":
    try:
        code = worker.main()
        (directory / "native_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit(code)
    finally:
        for explorer in explorers:
            explorer.close()
