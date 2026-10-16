"""Exercise the real local-browser analysis API without modifying raw recordings."""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--subject", default="HS0015")
    parser.add_argument("--block", default="B03")
    parser.add_argument("--first-k", type=int, default=3)
    parser.add_argument("--second-k", type=int, default=2)
    parser.add_argument("--baseline-mode", choices=("file", "pre_onset"), default="file")
    parser.add_argument("--stability-resamples", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.url.rstrip("/") + "/"
    parsed = urlsplit(base)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if parsed.hostname != "127.0.0.1":
        raise ValueError("Verification is restricted to the local workspace.")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"subject": args.subject, "block": args.block, "steps": []}

    def api(route, body=None):
        headers = {"Origin": origin}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = Request(base + "api/workspace/" + route,
                          data=None if body is None else json.dumps(body).encode(), headers=headers)
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    def wait():
        previous, deadline = None, time.monotonic() + 3600
        while time.monotonic() < deadline:
            state = api("state")
            task = state.get("task") or {}
            key = (task.get("action"), task.get("status"), task.get("stage"))
            if key != previous:
                print(json.dumps(dict(stage=state["stage"], task=key, elapsed=task.get("elapsed_seconds")), ensure_ascii=False), flush=True)
                previous = key
            if task.get("status") not in {"running", "opening"}:
                if task.get("status") == "failed":
                    raise RuntimeError(task.get("error"))
                report["steps"].append(dict(stage=state["stage"], task=task))
                (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                return state
            time.sleep(1)
        raise TimeoutError("Workspace task did not finish within one hour.")

    def action(state, name, payload):
        api("action", dict(run_id=state["run_id"], revision=state["revision"], action=name, payload=payload))
        return wait()

    patients = api("catalog")["patients"]
    assert any(item["id"] == args.subject and args.block in item["blocks"] for item in patients)
    inputs = api("inputs?" + urlencode(dict(subject=args.subject, block=args.block)))
    report["availability"] = inputs.get("availability")
    api("start", dict(subject=args.subject, block=args.block,
        config=dict(first_k=args.first_k, second_k=args.second_k, baseline_mode=args.baseline_mode)))
    state = wait()
    assert state["stage"] == "trial_review"
    assert state["data"].get("qc_rows")
    # This diagnostic explicitly accepts zero manual exclusions, never all candidates.
    state = action(state, "confirm_trials", dict(pairs=[]))
    assert state["stage"] == "trial_confirmed"
    state = action(state, "fit_first", dict(first_k=args.first_k))
    assert state["stage"] == "first_ready"
    assert len(state["rounds"]) == 1
    review = dict(bad_clusters=[], manual_bad_channels=[], pca_distance_threshold=None,
                  distance_reference="global", second_k=args.second_k)
    state = action(state, "preview_review", review)
    assert state["stage"] == "first_ready"
    state = action(state, "fit_second", review)
    assert state["stage"] == "second_ready"
    assert len(state["rounds"]) == 2
    if args.stability_resamples:
        state = action(state, "stability", dict(round=2, n_resamples=args.stability_resamples))
    report.update(run_id=state["run_id"], revision=state["revision"], rounds=state["rounds"],
                  artifacts=state.get("artifacts"), explorer_url=state.get("explorer_url"), stage=state["stage"])
    # Reopen durable state, including the accepted second round.
    api("open", dict(subject=args.subject, block=args.block, run_id=state["run_id"]))
    reopened = wait()
    assert reopened["run_id"] == report["run_id"] and reopened["stage"] == "second_ready"
    assert reopened.get("explorer_url")
    (args.output / "final_state.json").write_text(json.dumps(reopened, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(verified=True, run_id=report["run_id"], url=base), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
