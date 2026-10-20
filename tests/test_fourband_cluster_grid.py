"""Saved summary curves are validated and served without numerical jobs."""
import csv
from copy import deepcopy
import http.client
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3-bands"))

from fourband import browser_data
from fourband.browser_session import BrowserSession


COLUMNS = ["band", "label", "display_time_seconds", "mean_power_z",
           "electrode_sem", "n_electrodes", "mean_audio"]
LIMITS = {"theta": [4., 8.], "beta1": [13., 24.],
          "beta2": [24., 40.], "high_gamma": [70., 150.]}


@pytest.fixture
def snapshot(tmp_path):
    metadata = dict(channels=["E9", "E2", "E7", "E4"], labels=[2, 2, -1, 5],
                    bands=list(LIMITS), summary_band_limits=deepcopy(LIMITS),
                    summary_speech_duration_seconds=1.25)
    result = SimpleNamespace(channels=tuple(metadata["channels"]),
                             labels=np.array(metadata["labels"]),
                             clean_mask=np.array([True, False, False, True]))
    request = dict(output_dir=str(tmp_path), result=result)
    rows = []
    for band in LIMITS:
        for label in (2, 5):
            for time, mean, sem, count, audio in [(-.5, 1., .25, 1, .2),
                                                  (0., "", "NaN", 0, ""),
                                                  (1.25, -2., 0., 1, .7)]:
                rows.append(dict(zip(COLUMNS, [band, label, time, mean, sem, count, audio])))
    write_summary(tmp_path, rows)
    return request, metadata, rows


def write_summary(directory, rows, columns=COLUMNS):
    path = Path(directory) / "fourband_curve_summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def payload(request, metadata):
    function = getattr(browser_data, "cluster_grid_payload", None)
    assert callable(function), "Saved cluster-grid adapter is missing"
    return function(request, metadata)


def assert_unavailable(value):
    assert value["available"] is False
    assert isinstance(value["message"], str) and value["message"]
    assert "panels" not in value
    json.dumps(value, allow_nan=False)


def test_saved_curves_preserve_gaps_nulls_populations_and_input(snapshot):
    request, metadata, _ = snapshot
    path = Path(request["output_dir"]) / "fourband_curve_summary.csv"
    before = path.read_bytes(), deepcopy(metadata), request["result"].labels.copy()
    value = payload(request, metadata)
    assert value["available"] is True
    assert value["bands"] == [dict(name=name, low=low, high=high)
                              for name, (low, high) in LIMITS.items()]
    assert value["classes"] == [dict(label=2, n_electrodes=2), dict(label=5, n_electrodes=1)]
    assert value["speech_offset_ms"] == 1250.
    assert len(value["panels"]) == 8
    assert value["panels"][0] == dict(band="theta", label=2, times_ms=[-500., 0., 1250.],
                                      mean=[1., None, -2.], sem=[.25, None, 0.],
                                      audio=[.2, None, .7], counts=[1, 0, 1])
    assert [(panel["band"], panel["label"]) for panel in value["panels"]] == [
        (band, label) for band in LIMITS for label in (2, 5)]
    assert path.read_bytes() == before[0] and metadata == before[1]
    np.testing.assert_array_equal(request["result"].labels, before[2])
    json.dumps(value, allow_nan=False)


def test_legacy_metadata_does_not_invent_limits_or_duration(snapshot):
    request, metadata, _ = snapshot
    del metadata["summary_band_limits"], metadata["summary_speech_duration_seconds"]
    value = payload(request, metadata)
    assert value["available"] is True
    assert value["bands"] == [dict(name=name, low=None, high=None) for name in LIMITS]
    assert value["speech_offset_ms"] is None


def test_missing_summary_ignores_arbitrary_request_paths(snapshot, tmp_path):
    request, metadata, _ = snapshot
    other = tmp_path / "other"
    other.mkdir()
    request.update(output_dir=str(other), path=str(tmp_path / "fourband_curve_summary.csv"),
                   summary_path=str(tmp_path / "fourband_curve_summary.csv"))
    assert_unavailable(payload(request, metadata))


@pytest.mark.parametrize("column,value", [
    ("band", "foreign"), ("label", 99), ("label", -1), ("label", 2.5),
    ("label", "NaN"), ("display_time_seconds", "NaN"),
    ("display_time_seconds", "inf"), ("display_time_seconds", "1e308"),
    ("n_electrodes", -1), ("n_electrodes", .5), ("n_electrodes", 3),
    ("n_electrodes", "NaN"), ("n_electrodes", "inf"),
    ("electrode_sem", -.1), ("electrode_sem", "inf"),
    ("mean_power_z", "inf"), ("mean_audio", "-inf"), ("mean_power_z", "bad"),
])
def test_invalid_summary_values_stay_local(snapshot, column, value):
    request, metadata, rows = snapshot
    rows[0][column] = value
    write_summary(request["output_dir"], rows)
    assert_unavailable(payload(request, metadata))


@pytest.mark.parametrize("change", ["duplicate_time", "reversed_time", "missing_band",
                                     "missing_class", "missing_panel", "empty", "columns",
                                     "duplicate_column", "extra_cell", "short_row", "encoding",
                                     "truncated_panel", "shifted_panel"])
def test_invalid_structure_and_panel_coverage(snapshot, change):
    request, metadata, rows = snapshot
    columns = COLUMNS
    if change == "duplicate_time":
        rows[1]["display_time_seconds"] = rows[0]["display_time_seconds"]
    elif change == "reversed_time":
        rows[0], rows[1] = rows[1], rows[0]
    elif change == "missing_band":
        rows = [row for row in rows if row["band"] != "theta"]
    elif change == "missing_class":
        rows = [row for row in rows if row["label"] != 5]
    elif change == "missing_panel":
        rows = [row for row in rows if (row["band"], row["label"]) != ("theta", 5)]
    elif change == "truncated_panel":
        rows = rows[:-2]
    elif change == "shifted_panel":
        rows[-1]["display_time_seconds"] = 1.5
    elif change == "empty":
        rows = []
    elif change == "columns":
        columns = COLUMNS[:-1]
    elif change == "duplicate_column":
        columns = COLUMNS + ["band"]
    path = write_summary(request["output_dir"], rows, columns)
    if change == "extra_cell":
        with path.open("a", encoding="utf-8", newline="") as stream:
            stream.write("theta,2,9,1,0,1,.5,extra\n")
    elif change == "short_row":
        with path.open("a", encoding="utf-8", newline="") as stream:
            stream.write("theta,2,9,1\n")
    elif change == "encoding":
        path.write_bytes(b"\xff\xfeinvalid")
    assert_unavailable(payload(request, metadata))


@pytest.mark.parametrize("change", ["channels", "labels", "fractional_labels", "duplicate_channels"])
def test_result_identity_must_match_metadata(snapshot, change):
    request, metadata, _ = snapshot
    if change == "channels":
        request["result"].channels = ("E2", "E9", "E7", "E4")
    elif change == "labels":
        request["result"].labels = np.array([5, 2, -1, 2])
    elif change == "fractional_labels":
        request["result"].labels = np.array([2.5, 2, -1, 5])
    else:
        metadata["channels"][1] = "E9"
        request["result"].channels = tuple(metadata["channels"])
    assert_unavailable(payload(request, metadata))


def test_metadata_uses_retained_event_median_and_extraction_band_limits(tmp_path):
    events = pd.DataFrame(dict(duration=[.5, 1.25, 9., np.nan]))
    channels = ("E1", "E2")
    extraction = SimpleNamespace(bands=deepcopy(LIMITS), epochs={
        name: SimpleNamespace(channel_names=channels, events=events.copy()) for name in LIMITS})
    request = dict(extraction=extraction, result=SimpleNamespace(channels=channels, labels=[0, 0]),
                   output_dir=str(tmp_path), link_explorer=False, sources={})
    value = browser_data.session_metadata(request)
    assert value.get("summary_band_limits") == LIMITS
    assert value.get("summary_speech_duration_seconds") == 1.25


class NoComputation:
    def abort(self):
        pass

    def __getattr__(self, name):
        raise AssertionError(f"Summary must not use numerical computation: {name}")


def make_session(snapshot):
    request, metadata, _ = snapshot
    return BrowserSession(request, Path(request["output_dir"]) / "absent.joblib",
                          metadata=metadata, computation=NoComputation())


def test_session_caches_saved_payload_and_protects_it_from_caller_mutation(snapshot):
    session = make_session(snapshot)
    method = getattr(session, "cluster_grid", None)
    assert callable(method), "Session cluster-grid cache is missing"
    first = method()
    first["panels"][0]["mean"][0] = 999
    (Path(session.request["output_dir"]) / "fourband_curve_summary.csv").unlink()
    assert method()["panels"][0]["mean"][0] == 1.


def test_session_caches_unavailable_payload(snapshot):
    request, metadata, rows = snapshot
    path = Path(request["output_dir"]) / "fourband_curve_summary.csv"
    path.unlink()
    session = make_session(snapshot)
    method = getattr(session, "cluster_grid", None)
    assert callable(method), "Session cluster-grid cache is missing"
    assert_unavailable(method())
    write_summary(request["output_dir"], rows)
    assert_unavailable(method())


@pytest.fixture(params=["direct", "nested"])
def http_grid(request, snapshot):
    explorer = make_session(snapshot)
    workspace = None
    if request.param == "direct":
        from fourband.browser_server import make_server
        server = make_server(explorer, token="grid-secret")
        route = "/s/grid-secret/api/cluster-grid"
    else:
        from fourband.workspace_session import WorkspaceSession
        from fourband.workspace_server import make_workspace_server
        workspace = WorkspaceSession(SimpleNamespace(output_root=Path(explorer.request["output_dir"])))
        workspace._explorer_id, workspace._explorer = "active-revision", explorer
        server = make_workspace_server(workspace, token="grid-secret")
        route = "/s/grid-secret/explorer/active-revision/api/cluster-grid"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, route, explorer
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if workspace is not None:
            workspace.close()
            workspace._monitor.join(timeout=5)


def get(server, path):
    connection = http.client.HTTPConnection(*server.server_address, timeout=5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_http_grid_and_existing_access_checks(http_grid):
    server, route, explorer = http_grid
    status, value = get(server, route + "?path=C:/arbitrary.csv&output_dir=C:/elsewhere")
    assert status == 200 and value["available"] is True
    assert value["panels"][0]["mean"] == [1., None, -2.]
    assert get(server, route.replace("grid-secret", "wrong-token"))[0] == 404
    assert get(server, route.replace("api/cluster-grid", "../api/cluster-grid"))[0] == 404
    if "explorer/" in route:
        assert get(server, route.replace("active-revision", "stale-revision"))[0] == 400
    assert not explorer._jobs


def test_http_missing_summary_is_a_successful_local_unavailable_response(http_grid):
    server, route, explorer = http_grid
    (Path(explorer.request["output_dir"]) / "fourband_curve_summary.csv").unlink()
    status, value = get(server, route)
    assert status == 200
    assert_unavailable(value)
