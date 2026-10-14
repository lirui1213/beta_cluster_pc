"""Authenticated local browser sessions preserve classification ownership."""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
import base64
import http.client
import json
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))


class Computation:
    def __init__(self):
        self.jobs = []
        self.cleared = False

    def submit(self, settings):
        future = Future()
        self.jobs.append((settings, future))
        return future

    def submit_task(self, function, *args):
        return self.submit(args[-1])

    def clear(self):
        self.cleared = True


@pytest.fixture
def session(tmp_path):
    from fourband.browser_session import BrowserSession
    metadata = dict(channels=["E1", "E2"], labels=[0, 1], bands=["theta", "beta1"],
        pairs=[dict(first_trial_id=5, second_trial_id=6), dict(first_trial_id=6, second_trial_id=7)],
        settings=dict(trial_ids=[5, 6], sigma_seconds=.1, pre_seconds=1., post_seconds=1., display_sfreq=100.),
        curves_enabled=True, surface_enabled=False)
    return BrowserSession(dict(output_dir=str(tmp_path), result=SimpleNamespace(labels=[0, 1])),
                          tmp_path / "request.joblib", computation=Computation(), metadata=metadata)


def test_same_power_job_reused_for_channel_changes_and_old_jobs_superseded(session):
    first = session.compute(dict(trial_ids=[5, 6], channel="E1", sigma_seconds=.1))
    same = session.compute(dict(trial_ids=[5, 6], channel="E2", sigma_seconds=.1))
    assert same["job_id"] == first["job_id"] and len(session.computation.jobs) == 1
    changed = session.compute(dict(trial_ids=[6, 7], channel="E2", sigma_seconds=.1))
    assert changed["job_id"] != first["job_id"]
    assert session.read_job(first["job_id"], "E1")["state"] == "superseded"


def test_returning_to_running_configuration_reuses_its_future(session):
    first = session.compute(dict(trial_ids=[5, 6]))
    original = session.computation.jobs[0][1]
    original.set_running_or_notify_cancel()
    second = session.compute(dict(trial_ids=[6, 7]))
    session.computation.jobs[1][1].set_running_or_notify_cancel()
    latest = session.compute(dict(trial_ids=[5, 6]))
    assert len(session.computation.jobs) == 2
    assert session._jobs[latest["job_id"]]["future"] is original
    assert session.read_job(second["job_id"], "E1")["state"] == "superseded"
    assert session.read_job(first["job_id"], "E1")["state"] == "superseded"


@pytest.mark.parametrize("change", [dict(channel="missing"), dict(trial_ids=[5, 7]),
    dict(trial_ids=[True, 6]), dict(sigma_seconds=float("nan")), dict(sigma_seconds=-1),
    dict(view="exec"), dict(band="made_up"), dict(pre_seconds=999)])
def test_invalid_display_settings_never_submit(session, change):
    with pytest.raises(ValueError):
        session.compute(change)
    assert not session.computation.jobs


def test_numeric_failure_is_visible_and_clear_invalidates_old_result(session):
    job = session.compute({})
    session.computation.jobs[0][1].set_exception(RuntimeError("numerical child stopped"))
    assert "numerical child stopped" in session.read_job(job["job_id"], "E1")["error"]
    session.clear()
    assert session.computation.cleared
    assert session.read_job(job["job_id"], "E1")["state"] == "superseded"


def test_completed_job_elapsed_time_stops_increasing(session, monkeypatch):
    from fourband import browser_session as module
    clock = [100.]
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    job = session.compute({})
    clock[0] = 110.
    session.computation.jobs[0][1].set_exception(RuntimeError("test failure"))
    clock[0] = 140.
    assert session.read_job(job["job_id"], "E1")["elapsed_seconds"] == 10


@pytest.fixture
def http_session(session, tmp_path):
    from fourband.browser_server import make_server
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "index.html").write_text("<html>Explorer</html>", encoding="ascii")
    server = make_server(session, token="test-secret", assets_dir=assets)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def request(server, method, path, body=None, origin=None):
    conn = http.client.HTTPConnection(*server.server_address, timeout=10)
    headers = {}
    if body is not None:
        body = json.dumps(body)
        headers["Content-Type"] = "application/json"
    if origin is not None:
        headers["Origin"] = origin
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    result = response.status, response.read(), dict(response.getheaders())
    conn.close()
    return result


def test_loopback_and_token_boundary(http_session):
    assert http_session.server_address[0] == "127.0.0.1"
    assert request(http_session, "GET", "/api/meta")[0] == 404
    assert request(http_session, "GET", "/s/wrong/api/meta")[0] == 404
    assert request(http_session, "GET", "/s/test-secret/api/meta")[0] == 200
    assert request(http_session, "GET", "/s/test-secret/../../request.joblib")[0] == 404


def test_mutation_requires_matching_origin_and_json(http_session):
    endpoint = "/s/test-secret/api/compute"
    assert request(http_session, "POST", endpoint, {})[0] == 403
    assert request(http_session, "POST", endpoint, {}, "https://malicious.invalid")[0] == 403
    origin = "http://%s:%s" % http_session.server_address
    assert request(http_session, "POST", endpoint, {}, origin)[0] == 200
    assert request(http_session, "GET", "/s/test-secret/api/close")[0] == 404


def test_error_response_has_no_external_cors_and_exports_reject_arbitrary_paths(http_session):
    origin = "http://%s:%s" % http_session.server_address
    status, body, headers = request(http_session, "POST", "/s/test-secret/api/export",
        dict(settings={}, images={"../../escape": "data:image/png;base64,AA=="}), origin)
    assert status == 400
    assert "Access-Control-Allow-Origin" not in headers


def test_export_only_writes_valid_png_and_json_under_round(session):
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"local-test-image").decode("ascii")
    exported = session.export(dict(settings={"channel": "E1"}, images={"curves": "data:image/png;base64," + png}))
    target = Path(exported["directory"])
    assert target.is_relative_to(Path(session.request["output_dir"]) / "browser_exports")
    assert (target / "curves.png").is_file()
    assert json.loads((target / "settings.json").read_text())["channel"] == "E1"


def test_real_idle_worker_crash_becomes_parent_exception(tmp_path, monkeypatch):
    from fourband import display_process as module
    monkeypatch.setattr(module, "_WORKER_SCRIPT", Path(__file__).with_name("browser_idle_failure_probe.py"))
    monkeypatch.setenv("FOURBAND_NO_BROWSER", "1")
    result = SimpleNamespace(channels=("E1",), labels=[0])
    with pytest.raises(module.DisplayProcessError) as error:
        module.run_display({}, result, output_dir=tmp_path, show_surface=True, link_explorer=False)
    status = json.loads((error.value.directory / "worker_status.json").read_text())
    assert status["stage"] == "failed" and "Numerical display process failed" in status["error"]
    assert result.labels == [0]


def test_user_close_stops_busy_owned_numerical_process(tmp_path, monkeypatch):
    from fourband import display_process as module
    monkeypatch.setattr(module, "_WORKER_SCRIPT", Path(__file__).with_name("browser_idle_failure_probe.py"))
    monkeypatch.setenv("FOURBAND_NO_BROWSER", "1")
    monkeypatch.setenv("FOURBAND_PROBE_BUSY_CLOSE", "1")
    started = time.monotonic()
    run = module.run_display({}, {}, output_dir=tmp_path, show_surface=True, link_explorer=False)
    assert run.returncode == 0 and time.monotonic() - started < 15
    assert json.loads((run.directory / "worker_status.json").read_text())["stage"] == "complete"
