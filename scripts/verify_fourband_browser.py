"""Replay a trusted fitted snapshot through the real owned browser launcher."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "3-bands"), str(ROOT)]


def main():
    import joblib
    import numpy as np
    from fourband.display_process import run_display
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--trial-pair", nargs=2, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    request = joblib.load(args.request)
    extraction, result = request.pop("extraction"), request.pop("result")
    before = result.labels.copy()
    sources = request.pop("sources")
    request.pop("output_dir")
    request["backend"] = "browser"
    if args.trial_pair:
        request["explorer_settings"]["trial_ids"] = tuple(args.trial_pair)
    output = args.output or ROOT / "3-bands/outputs/verification/browser_display" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run = run_display(extraction, result, output_dir=output, **sources, **request)
    np.testing.assert_array_equal(before, result.labels)
    (run.directory / "verification_parent.json").write_text(json.dumps(dict(
        labels_unchanged=True, returncode=run.returncode, display_pid=run.pid)), encoding="utf-8")
    print("Browser session closed; parent survived and fitted labels are unchanged.", flush=True)


if __name__ == "__main__":
    main()
