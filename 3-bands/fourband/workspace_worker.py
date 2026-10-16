"""Trusted task-file entry point; heavy scientific work never runs in HTTP threads."""
import json
import os
from pathlib import Path
import sys
import traceback

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("OMP_NUM_THREADS", "1")
if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "3-bands"), str(root)]


def main():
    from fourband.workspace_task import atomic_json, copy_action_state
    directory = Path(sys.argv[1]).resolve().parent
    if os.environ.get("FOURBAND_WORKSPACE_GATE") == "1" and not sys.stdin.buffer.read(1):
        return 1
    import faulthandler
    faulthandler.enable()
    try:
        task = json.loads((directory / "task.json").read_text(encoding="utf-8"))
        os.environ["FOURBAND_WORKSPACE_TASK_DIR"] = str(directory)
        atomic_json(directory / "progress.json", dict(stage="Loading numerical modules"))
        if task["action"] == "prepare_display":
            from fourband.workspace_display import prepare_display
            atomic_json(directory / "progress.json", dict(stage="Loading saved model and anatomy"))
            response = prepare_display(task["payload"]["request_path"], task["run_dir"])
        else:
            state_dir = directory / "state"
            atomic_json(directory / "progress.json", dict(stage="Preparing isolated analysis state"))
            copy_action_state(task["source_state_dir"], state_dir)
            from fourband.workspace_actions import execute_action
            response = execute_action(state_dir, task["action"], task["payload"])
            response["numerical_state_dir"] = str(state_dir)
        from fourband.browser_data import json_safe
        atomic_json(directory / "result.json", json_safe(response))
        return 0
    except BaseException as error:
        traceback.print_exc()
        atomic_json(directory / "result.json", dict(error=str(error)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
