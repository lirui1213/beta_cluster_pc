"""Dispose an embedded Matplotlib canvas before destroying its Tk owner."""
from matplotlib.backends.backend_agg import FigureCanvasAgg


def dispose_tk_canvas(canvas, figure):
    from tkinter import TclError

    widget = canvas.get_tk_widget()
    for name in ("_idle_draw_id", "_event_loop_id"):
        timer = getattr(canvas, name, None)
        if timer is not None:
            try:
                widget.after_cancel(timer)
            except TclError:
                pass
            setattr(canvas, name, None)
    # Figure.clear touches its toolbar; detach it while the Tk widget is live.
    FigureCanvasAgg(figure)
    figure.clear()
    try:
        widget.destroy()
    except TclError:
        pass
