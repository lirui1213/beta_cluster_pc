"""Own all native brain/curve windows in a fresh, non-kernel process."""
import json
import os
from pathlib import Path
import sys
import traceback


if __package__ in (None, ""):
    project = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(project / "3-bands"), str(project)]

if __name__ == "__mp_main__":
    from fourband.display_logging import configure_numerical_logging
    configure_numerical_logging()

from audio_cross_3band_utils import load_ecog_fif, load_speech_events, extract_audio_envelope
from fourband.display_computation import DisplayComputation
from fourband.explorer import FourBandExplorer
from visualization import render_electrode_clusters, render_selectable_electrode_class


def _status(directory, stage, **details):
    (Path(directory) / "worker_status.json").write_text(
        json.dumps(dict(stage=stage, pid=os.getpid(), **details), indent=2), encoding="utf-8")
    print(stage, flush=True)


def run_request(request, directory):
    """Render a trusted local snapshot without modifying classification data."""
    output = Path(request["output_dir"])
    explorer = None
    computation = None
    try:
        if request["link_explorer"] or request["explorer_only"]:
            sources = request["sources"]
            _status(directory, "loading_recording")
            # The GUI needs recording metadata only; full samples live in the numerical process.
            raw = load_ecog_fif(sources["fif_path"], preload=False)
            events = load_speech_events(sources["onset_path"], sources["offset_path"])
            audio = extract_audio_envelope(sources["wav_path"])
            computation = DisplayComputation(Path(directory) / "request.joblib")
            explorer = FourBandExplorer(raw, events, audio, request["extraction"], request["result"],
                continuous_submitter=computation.submit, cache_clearer=computation.clear,
                output_dir=output / "explorer", **request["explorer_settings"])
        if request["explorer_only"]:
            _status(directory, "explorer")
            explorer.show(block=True)
        else:
            if explorer is not None:
                explorer.show(block=False)
            options = dict(request["surface_options"])
            csv_path = options.pop("electrode_csv")
            options.setdefault("channel_label_toggle", True)
            options.update(on_channel_pick=explorer.select_channel if explorer is not None else None,
                           on_window_tick=explorer.pump if explorer is not None else None)
            result = request["result"]
            labels = dict(zip(result.channels, map(int, result.labels)))
            if request["show_surface"]:
                _status(directory, "brain_surface")
                render_electrode_clusters(csv_path, labels, screenshot=output / "joint_electrode_clusters.png", **options)
            if request["show_selectable"]:
                _status(directory, "selectable_surface")
                render_selectable_electrode_class(csv_path, labels, **options)
        if computation is not None:
            computation.close()
            computation.raise_if_failed()
        _status(directory, "complete")
    finally:
        if explorer is not None:
            explorer.close()
        if computation is not None:
            computation.close()


def main():
    import faulthandler
    import joblib

    faulthandler.enable()
    request_path = Path(sys.argv[1]).resolve()
    directory = request_path.parent
    try:
        if os.environ.get("FOURBAND_DISPLAY_GATE") == "1" and not sys.stdin.buffer.read(1):
            raise RuntimeError("Launcher disconnected before display process ownership was established.")
        _status(directory, "loading_snapshot", parent_pid=os.getppid())
        request = joblib.load(request_path)
        run_request(request, directory)
    except Exception as error:
        _status(directory, "failed", error=str(error))
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
