"""Destructive retention acceptance ONLY against an isolated verification root."""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    outputs = Path(__file__).resolve().parents[1] / "3-bands" / "outputs"
    if not root.is_relative_to(outputs) or root == outputs or not (root.name == "rt19" or "verification" in root.parts):
        raise ValueError("Refusing replacement tests outside a dedicated verification root.")
    base = args.url.rstrip("/") + "/"
    parsed = urlsplit(base)
    if parsed.hostname != "127.0.0.1":
        raise ValueError("Local test service required.")
    origin = f"{parsed.scheme}://{parsed.netloc}"
    context = dict(subject="HS0015", block="B03")

    def api(route, body=None):
        request = Request(base + "api/workspace/" + route,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Origin": origin, "Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    def wait():
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            state = api("state")
            task = state.get("task") or {}
            if task.get("status") not in {"running", "opening", "cancelling"}:
                if task.get("status") == "failed":
                    raise RuntimeError(task["error"])
                return state
            time.sleep(.25)
        raise TimeoutError("Verification task did not finish")

    state = wait()
    current = api("current?" + urlencode(context))["analysis"]
    assert current and state["run_id"] == current["run_id"]
    numerical = Path(state["numerical_state_dir"]).resolve()
    assert numerical.is_relative_to(root), "Refuse to replace a user/production analysis"
    old = root / context["subject"] / context["block"] / current["run_id"]
    sentinel = old.parent / "notebook_verification" / "run_config.json"
    sentinel.parent.mkdir(exist_ok=True)
    sentinel.write_bytes(b'{"owner":"Notebook verification sentinel"}')
    original = sentinel.read_bytes()
    config = dict(first_k=3, second_k=2, baseline_mode="file")
    pending = api("start", dict(**context, config=config, replace_run_id=old.name))
    assert old.is_dir(), "Replacement extraction must not delete accepted analysis"
    api("cancel", dict(run_id=pending["run_id"], revision=pending["revision"]))
    assert old.is_dir() and api("current?" + urlencode(context))["analysis"]["run_id"] == old.name
    assert not (old.parent / pending["run_id"]).exists()
    api("start", dict(**context, config=config, replace_run_id=old.name))
    accepted = wait()
    assert accepted["stage"] == "trial_review" and accepted["data"]["qc_rows"]
    fresh = old.parent / accepted["run_id"]
    assert fresh.is_dir() and not old.exists()
    api("open", dict(**context, run_id=fresh.name))
    reopened = wait()
    assert reopened["stage"] == "trial_review" and reopened["data"]["qc_rows"]
    assert sentinel.read_bytes() == original
    runs = [path for path in old.parent.iterdir() if (path / "workspace_manifest.json").is_file()]
    assert runs == [fresh]
    assert len(list(fresh.glob("workspace_data_*.json"))) == 1
    assert not reopened.get("retention_warning")
    report = dict(verified=True, cancelled_replacement_preserved_old=True,
        successful_replacement_removed_old=True, notebook_preserved=True,
        current_run_id=fresh.name, browser_analysis_count=len(runs),
        metadata_count=1, retained_trial_count=len(reopened["data"]["events"]))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
