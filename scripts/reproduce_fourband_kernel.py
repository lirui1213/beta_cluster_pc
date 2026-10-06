"""Reproduce current notebook parameters and saved selections with native UIs."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from queue import Empty
import time

from jupyter_client import KernelManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="E:/anaconda/envs/ECoG_LR_v1/python.exe")
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--second-config", type=Path, required=True)
    parser.add_argument("--through", choices=("fit", "review", "surface"), default="surface")
    parser.add_argument("--inprocess", action="store_true", help="Fallback when local ZeroMQ cannot start a kernel")
    parser.add_argument("--trace-text", action="store_true")
    parser.add_argument("--resume-surface", type=Path, help="Numerical checkpoint from an earlier reproduction")
    parser.add_argument("--native-close", action="store_true", help="Finalize native OpenGL windows before exit, like Win32 close")
    parser.add_argument("--no-watchdog", action="store_true", help="Compatibility flag; periodic native stack dumps are disabled")
    args = parser.parse_args()
    args.review_csv = args.review_csv.resolve()
    args.second_config = args.second_config.resolve()
    if args.resume_surface is not None:
        args.resume_surface = args.resume_surface.resolve()
    project = Path(__file__).resolve().parents[1]
    notebook = json.loads((project / "3-bands/ECoG_4band_clustering_pipeline.ipynb").read_text(encoding="utf-8"))
    output = project / "3-bands/outputs/verification/kernel_reproduction" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    manager = KernelManager()
    manager.kernel_spec.argv = [args.python, "-X", "faulthandler", "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    log = (output / "execution.log").open("w", encoding="utf-8")
    native = (output / "kernel_stderr.log").open("w", encoding="utf-8")
    client = None

    def report(value):
        print(value, flush=True)
        if not args.inprocess or not hasattr(__import__('sys').stdout, 'stream'):
            log.write(str(value) + "\n")
            log.flush()

    def execute(source, name, timeout=240):
        report("ENTER " + name)
        if args.inprocess:
            exec(compile(source, name, "exec"), namespace)
            report("LEAVE " + name)
            return
        message_id = client.execute(source)
        started = time.monotonic()
        error = None
        while time.monotonic() - started < timeout:
            if not manager.is_alive():
                raise RuntimeError("Kernel exited during " + name)
            try:
                message = client.get_iopub_msg(timeout=1)
            except Empty:
                continue
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue
            kind, content = message["msg_type"], message["content"]
            if kind == "stream":
                report(content["text"].rstrip())
            elif kind == "error":
                error = "\n".join(content["traceback"])
                report(error)
            elif kind in ("display_data", "execute_result"):
                data = content.get("data", {})
                if "image/png" in data:
                    report("DISPLAY PNG bytes=" + str(len(data["image/png"])))
                elif "text/plain" in data:
                    report(data["text/plain"][:1500])
            elif kind == "status" and content["execution_state"] == "idle":
                if error:
                    raise RuntimeError(error)
                report("LEAVE " + name + " elapsed=" + str(round(time.monotonic() - started, 2)))
                return
        raise TimeoutError("Kernel did not finish " + name + "; inspect the native stack dump.")

    hooks = '''
import csv, faulthandler, traceback
faulthandler.enable()
_review_csv = Path(REVIEW_CSV)
with _review_csv.open(encoding="utf-8-sig", newline="") as _stream:
    _pairs = tuple((r["channel"], int(r["original_trial_id"])) for r in csv.DictReader(_stream))
_config = json.loads(Path(SECOND_CONFIG).read_text(encoding="utf-8"))
_qc_show = TrialQualityReviewWindow.show
_review_show = FourBandReviewWindow.show
def _auto_popup(window, original, action):
    import tkinter as tk
    factory = tk.Tk
    errors = []
    def finish():
        try:
            action(window)
        except Exception:
            errors.append(traceback.format_exc())
            window.cancel()
    def create(*args, **kwargs):
        root = factory(*args, **kwargs)
        root.after(400, finish)
        return root
    tk.Tk = create
    try:
        result = original(window)
        if errors:
            raise RuntimeError(errors[0])
        return result
    finally:
        tk.Tk = factory
def _accept_qc(window):
    window._set_selection(_pairs)
    window.confirm()
def _show_qc(window):
    return _auto_popup(window, _qc_show, _accept_qc)
def _show_review(window):
    window.initial = ReviewSettings(n_clusters=_config["n_clusters"],
        bad_clusters=tuple(_config["bad_clusters"]),
        manual_bad_channels=tuple(_config["manual_bad_channels"]),
        pca_distance_threshold=_config["pca_distance_threshold"],
        distance_reference=_config["distance_reference"])
    return _auto_popup(window, _review_show, lambda item: item.apply())
TrialQualityReviewWindow.show = _show_qc
FourBandReviewWindow.show = _show_review
print("REPRODUCTION: native popup confirmation uses", len(_pairs), "saved channel-trial selections", flush=True)
'''
    surface_hook = '''
import time, pyvista as pv
_native_show = pv.Plotter.show
_surface_count = 0
def _timed_show(plotter, *args, **kwargs):
    global _surface_count
    _surface_count += 1
    number = _surface_count
    plotter.iren.initialize()
    iren = plotter.iren
    started = time.monotonic()
    ticks = [0]
    closing = [False]
    errors = []
    def probe(caller, event):
        if closing[0]:
            return
        ticks[0] += 1
        if ticks[0] == 1:
            print("SURFACE", number, "first native timer event", flush=True)
        explorer = globals().get("surface_explorer")
        pending = explorer is not None and explorer._future is not None
        if pending and ticks[0] == 5:
            print("SURFACE", number, "native event loop responsive during background computation", flush=True)
        if time.monotonic() - started >= 180:
            errors.append("Linked explorer did not finish within 180 seconds")
            pending = False
        if time.monotonic() - started >= 4 and not pending:
            closing[0] = True
            if NATIVE_CLOSE:
                print("SURFACE", number, "finalizing native OpenGL window before ExitEvent", flush=True)
                plotter.render_window.Finalize()
            print("SURFACE", number, "requesting native ExitEvent", flush=True)
            caller.InvokeEvent("ExitEvent")
    observer = iren.add_observer("TimerEvent", probe)
    timer = iren.create_timer(50, repeating=True)
    succeeded = False
    def cleanup():
        iren.destroy_timer(timer)
        iren.remove_observer(observer)
    try:
        result = _native_show(plotter, *args, **kwargs)
        succeeded = True
        print("SURFACE", number, "show returned, ticks=", ticks[0], flush=True)
        if errors:
            raise RuntimeError(errors[0])
        return result
    finally:
        if succeeded and kwargs.get("interactive_update", False):
            from visualization import _on_plotter_close
            _on_plotter_close(plotter, cleanup)
        else:
            cleanup()
pv.Plotter.show = _timed_show
'''
    report("Artifacts: " + str(output))
    namespace = {}
    try:
        if args.inprocess:
            import sys
            os.chdir(project / "3-bands")
            class Tee:
                def __init__(self, stream):
                    self.stream = stream
                def write(self, value):
                    self.stream.write(value)
                    log.write(value)
                    log.flush()
                def flush(self):
                    self.stream.flush()
                    log.flush()
            sys.stdout = Tee(sys.stdout)
            sys.stderr = Tee(sys.stderr)
            namespace.update(__name__="__main__", NATIVE_LOG=native)
        else:
            manager.start_kernel(cwd=str(project / "3-bands"), stdout=native, stderr=native)
            client = manager.blocking_client()
            client.start_channels()
            client.wait_for_ready(timeout=45)
        execute("import os; os.environ['OMP_NUM_THREADS']='1'; import json, faulthandler", "kernel ready")
        code_cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        for number, source in enumerate(code_cells, 1):
            if number > {"fit": 7, "review": 9, "surface": 10}[args.through]:
                break
            if args.resume_surface is not None and 5 <= number <= 9:
                continue
            if number == 6:
                if args.inprocess:
                    hooks = hooks.replace("faulthandler.enable()", "faulthandler.enable(file=NATIVE_LOG)")
                execute("REVIEW_CSV=" + repr(str(args.review_csv.resolve())) + "\nSECOND_CONFIG="
                        + repr(str(args.second_config.resolve())) + "\n" + hooks, "install native confirmation hooks")
            if number == 10:
                if args.resume_surface is not None:
                    old_model = args.second_config.parent / "joint_model.joblib"
                    execute("import joblib\nextraction=joblib.load(" + repr(str(args.resume_surface)) + ")[\"extraction\"]\n"
                            "second_result=joblib.load(" + repr(str(old_model)) + ")\n"
                            "qc_raw=load_ecog_fif(FIF_PATH, preload=True)\n"
                            "TRIAL_QC_READY=SECOND_ROUND_READY=True\n"
                            "first_extraction=extraction\nextracted_settings=extraction_settings()\n"
                            "reviewed_trial_settings=TRIAL_QC_SETTINGS\n"
                            "first_fit_settings=dict(fit_settings(),first_k=FIRST_N_CLUSTERS)\n"
                            "ROUND2_DIR=Path(" + repr(str(output / 'surface')) + ")\nROUND2_DIR.mkdir()",
                            "restore numerical checkpoint for native surface regression")
                execute("NATIVE_CLOSE=" + repr(args.native_close) + "\n" + surface_hook, "install surface watchdog")
            if number == 7 and args.inprocess:
                execute("import joblib; joblib.dump({k: v for k, v in globals().copy().items() "
                        "if k in ('extraction', 'reviewed_extraction', 'EXTRACTION_CONFIG', 'FEATURE_SETTINGS')}, "
                        + repr(str(output / 'extraction.joblib')) + ")", "save numerical reproduction checkpoint")
                if args.trace_text:
                    execute("from matplotlib.backends.backend_agg import RendererAgg\n"
                            "_draw_text = RendererAgg.draw_text\n"
                            "_text_log = open(" + repr(str(output / 'text.log')) + ", 'w', encoding='utf-8')\n"
                            "def _trace_text(self, gc, x, y, s, prop, angle, **kwargs):\n"
                            "    _text_log.write(repr((x, y, s, angle)) + '\\n'); _text_log.flush()\n"
                            "    return _draw_text(self, gc, x, y, s, prop, angle, **kwargs)\n"
                            "RendererAgg.draw_text = _trace_text", "trace native text drawing")
            execute(source, "cell " + str(number))
            if number == 10:
                execute("if surface_explorer is not None:\n"
                        "    assert surface_explorer._future is None\n"
                        "    assert 'unavailable' not in surface_explorer.variables['status'].get().lower()\n"
                        "    surface_explorer.figure.savefig(ROUND2_DIR / 'verified_explorer.png')\n"
                        "    print('EXPLORER RENDER VERIFIED', surface_explorer.variables['status'].get())",
                        "verify linked curves completed")
            if number == 2:
                execute("OUTPUT_ROOT=Path(" + repr(str(output)) + ")", "redirect verification output only")
        execute("close_explorers(EXPLORERS); faulthandler.cancel_dump_traceback_later(); print('KERNEL STILL ALIVE')", "final cleanup")
        report("Actual notebook execution passed through " + args.through)
    finally:
        if client is not None:
            client.stop_channels()
        if not args.inprocess:
            manager.shutdown_kernel(now=True)
        else:
            import sys
            import faulthandler
            faulthandler.cancel_dump_traceback_later()
            faulthandler.disable()
            sys.stdout = sys.stdout.stream
            sys.stderr = sys.stderr.stream
        log.close()
        native.close()


if __name__ == "__main__":
    main()
