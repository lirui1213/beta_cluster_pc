"""Exercise the isolated display boundary with an existing local fitted snapshot."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "3-bands"), str(ROOT)]

import joblib
import numpy as np
from fourband import display_process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--explorer-only", action="store_true")
    args = parser.parse_args()
    if args.request:
        request = joblib.load(args.request)
        extraction, result = request.pop("extraction"), request.pop("result")
        request.pop("output_dir")
        sources = request.pop("sources")
        if args.explorer_only:
            request.update(explorer_only=True, show_surface=False, show_selectable=False)
        labels = result.labels.copy()
        output = ROOT / "3-bands/outputs/verification/isolated_display" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        display_process._WORKER_SCRIPT = ROOT / "tests/fourband_display_native_probe.py"
        run = display_process.run_display(extraction, result, output_dir=output, **sources, **request)
        evidence = json.loads((run.directory / "native_verification.json").read_text(encoding="utf-8"))
        assert evidence["curve_draws"] >= 2 and evidence["canvas_nonwhite_samples"] >= 10
        np.testing.assert_array_equal(labels, result.labels)
        print("Saved request replay passed with nonblank native Tk canvas; labels unchanged.")
        print(run.directory)
        return
    if args.checkpoint is None or args.model is None:
        parser.error("Provide --request or both --checkpoint and --model.")
    extraction = joblib.load(args.checkpoint)["extraction"]
    result = joblib.load(args.model)
    labels = result.labels.copy()
    nb = json.loads((ROOT / "3-bands/ECoG_4band_clustering_pipeline.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    settings = dict(Path=Path, FOURBAND_DIR=ROOT / "3-bands", print=lambda *a: None)
    exec(compile(cells[1], "current notebook parameters", "exec"), settings)
    output = ROOT / "3-bands/outputs/verification/isolated_display" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    display_process._WORKER_SCRIPT = ROOT / "tests/fourband_display_native_probe.py"
    run = display_process.run_display(extraction, result, output_dir=output,
        fif_path=settings["FIF_PATH"], onset_path=settings["ONSET_PATH"], offset_path=settings["OFFSET_PATH"],
        wav_path=settings["WAV_PATH"], show_surface=not args.explorer_only, show_selectable=not args.explorer_only,
        link_explorer=not args.explorer_only, explorer_only=args.explorer_only,
        explorer_settings=dict(sigma_seconds=settings["DISPLAY_SMOOTH_SIGMA_SECONDS"],
                               display_sfreq=settings["DISPLAY_SFREQ"]),
        surface_options=dict(electrode_csv=settings["ELECTRODE_CSV"], subjects_dir=settings["SUBJECTS_DIR"],
            subject=settings["SURFACE_SUBJECT"], reference_mri=settings["REFERENCE_MRI"],
            hemispheres=settings["HEMISPHERES"], brain_color=settings["SURFACE_COLOR"],
            brain_opacity=settings["SURFACE_OPACITY"], channel_label_toggle=True))
    evidence = json.loads((run.directory / "native_verification.json").read_text(encoding="utf-8"))
    assert evidence["curve_draws"] >= 2
    if args.explorer_only:
        assert evidence["standalone_complete"]
    else:
        assert len(evidence["surfaces"]) == 2
        assert all(surface["curves_ready"] for surface in evidence["surfaces"])
        assert evidence["picks"]
    assert evidence["worker_pid"] != __import__("os").getpid()
    assert all(pid != evidence["worker_pid"] and pid != __import__("os").getpid() for pid in evidence["numerical_pids"])
    np.testing.assert_array_equal(labels, result.labels)
    print("Isolated native brain/curve display passed; parent labels unchanged.")
    print(run.directory)


if __name__ == "__main__":
    main()
