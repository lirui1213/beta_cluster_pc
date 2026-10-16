"""Check real cancellation, failure and patient switch using only owned API tasks."""
import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stability", action="store_true")
    args = parser.parse_args()
    base = args.url.rstrip("/") + "/"
    parsed = urlsplit(base)
    assert parsed.hostname == "127.0.0.1"
    origin = f"{parsed.scheme}://{parsed.netloc}"

    def api(route, body=None):
        request = Request(base + "api/workspace/" + route,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Origin": origin, "Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    def wait():
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            state = api("state")
            if (state.get("task") or {}).get("status") not in {"running", "opening", "cancelling"}:
                return state
            time.sleep(.25)
        raise TimeoutError("Owned task did not finish")

    def identity(state):
        return dict(run_id=state["run_id"], revision=state["revision"])

    initial = wait()
    assert initial["stage"] == "second_ready"
    saved = dict(subject=initial["subject"], block=initial["block"], run_id=initial["run_id"])
    snapshot = initial.get("numerical_state_dir")
    # Invalid review fails in an owned numerical worker and cannot publish new state.
    api("action", dict(**identity(initial), action="fit_second", payload=dict(second_k=2,
        manual_bad_channels=["__not_a_channel__"])))
    failed = wait()
    assert failed["task"]["status"] == "failed"
    assert failed["stage"] == initial["stage"]
    assert failed.get("numerical_state_dir") == snapshot
    # Display deserialization/surface preparation is also cancellable subprocess work.
    loading = api("select_round", dict(**identity(failed), round_id=failed["selected_round"]))
    start = time.monotonic()
    cancelled = api("cancel", identity(loading))
    assert cancelled["task"]["status"] == "cancelled"
    cancel_elapsed = time.monotonic() - start
    reopened = api("open", saved)
    reopened = wait()
    assert reopened.get("explorer_url") and reopened["stage"] == "second_ready"
    # Start another real discovered context, explicitly cancel, then return to history.
    patients = api("catalog")["patients"]
    alternate = next(item for item in patients if item["id"] == "HS0012")
    assert "B01" in alternate["blocks"]
    other = api("start", dict(subject="HS0012", block="B01", config=dict(baseline_mode="pre_onset", first_k=3, second_k=2)))
    stopped = api("cancel", identity(other))
    assert stopped["task"]["status"] == "cancelled" and stopped["subject"] == "HS0012"
    restored = api("open", saved)
    restored = wait()
    assert restored["subject"] == saved["subject"] and restored.get("numerical_state_dir") == snapshot
    # The earlier identity must never be accepted after context/revision changes.
    try:
        api("action", dict(**identity(initial), action="confirm_trials", payload=dict(pairs=[])))
    except HTTPError as error:
        assert error.code == 400
    else:
        raise AssertionError("Stale confirmation was accepted")
    stability_summary = None
    if args.stability:
        api("action", dict(**identity(restored), action="stability", payload=dict(round=2, n_resamples=2)))
        diagnosed = wait()
        assert diagnosed["task"]["status"] == "completed", diagnosed["task"]
        stability_summary = diagnosed["data"]["stability"]["summary"]
        api("open", saved)
        restored = wait()
        assert restored["data"]["stability"]["summary"] == stability_summary
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(verified=True, original_run=saved, cancelled_run=other["run_id"],
        display_cancel_seconds=cancel_elapsed, failed_review_error=failed["task"]["error"],
        restored_run=restored["run_id"], final_revision=restored["revision"], stability_summary=stability_summary)
    (args.output / "lifecycle.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
