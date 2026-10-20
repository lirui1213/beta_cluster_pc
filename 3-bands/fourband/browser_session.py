"""Display-only jobs and bounded export ownership for one browser session."""
import base64
from copy import deepcopy
from datetime import datetime
import math
from numbers import Integral, Real
from pathlib import Path
import threading
from time import monotonic
import json


class BrowserSession:
    def __init__(self, request, request_path, *, computation=None, metadata=None):
        from .display_computation import DisplayComputation
        self.request, self.request_path = request, Path(request_path)
        self.computation = computation or DisplayComputation(self.request_path)
        if metadata is None:
            from .browser_data import session_metadata
            metadata = session_metadata(request)
        self.metadata = metadata
        self._lock = threading.RLock()
        self._jobs = {}
        self._latest = None
        self._counter = 0
        self._surface = None
        self._cluster_grid = None
        self.closing = False

    def surface(self):
        with self._lock:
            if self._surface is None:
                from .browser_data import surface_payload
                self._surface = surface_payload(self.request)
            return self._surface

    def cluster_grid(self):
        """Keep one saved snapshot, independent of numerical jobs and clear()."""
        with self._lock:
            if self._cluster_grid is None:
                from .browser_data import cluster_grid_payload
                self._cluster_grid = cluster_grid_payload(self.request, self.metadata)
            return deepcopy(self._cluster_grid)

    def _validate(self, body):
        allowed = {"view", "trial_ids", "sigma_seconds", "channel", "band"}
        if not isinstance(body, dict) or set(body) - allowed:
            raise ValueError("Unknown display settings.")
        if not self.metadata["curves_enabled"]:
            raise ValueError("Curve display is disabled for this session.")
        view = body.get("view", "continuous")
        if not isinstance(view, str):
            raise ValueError("Unknown view.")
        view = view.lower()
        if view not in {"continuous", "detail"}:
            raise ValueError("Unknown view.")
        channel = body.get("channel", self.metadata["channels"][0])
        band = body.get("band", self.metadata["bands"][0])
        if channel not in self.metadata["channels"] or band not in self.metadata["bands"]:
            raise ValueError("Unknown channel or band.")
        sigma = body.get("sigma_seconds", self.metadata["settings"]["sigma_seconds"])
        if isinstance(sigma, bool) or not isinstance(sigma, Real) or not math.isfinite(sigma) or not 0 <= sigma <= 60:
            raise ValueError("Sigma must be finite and between 0 and 60 seconds.")
        pair = body.get("trial_ids", self.metadata["settings"]["trial_ids"])
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(
                isinstance(v, bool) or not isinstance(v, Integral) for v in pair):
            raise ValueError("Choose an adjacent original-trial pair.")
        if not any(pair[0] == p["first_trial_id"] and pair[1] == p["second_trial_id"] for p in self.metadata["pairs"]):
            raise ValueError("Choose an adjacent retained original-trial pair.")
        settings = dict(self.metadata["settings"], trial_ids=tuple(pair), sigma_seconds=float(sigma))
        if view == "detail":
            settings = dict(channel=channel, band=band, sigma_seconds=float(sigma))
        return view, settings

    def compute(self, body):
        with self._lock:
            if self.closing:
                raise ValueError("Session is closing.")
            view, settings = self._validate(body)
            if self._latest is not None:
                previous = self._jobs[self._latest]
                failed = previous["future"].done() and (previous["future"].cancelled() or
                    previous["future"].exception() is not None)
                if previous["view"] == view and previous["settings"] == settings and not failed:
                    return dict(job_id=self._latest, state="computing")
            reusable = None
            for candidate in reversed(list(self._jobs.values())):
                future = candidate["future"]
                if (candidate["view"] == view and candidate["settings"] == settings
                        and not future.cancelled() and (not future.done() or future.exception() is None)):
                    reusable = candidate
                    break
            if self._latest is not None:
                previous["future"].cancel()
            if reusable is not None:
                future = reusable["future"]
            elif view == "continuous":
                future = self.computation.submit(settings)
            else:
                from .browser_data import render_detail_snapshot
                future = self.computation.submit_task(render_detail_snapshot, self.request_path, settings)
            self._counter += 1
            identifier = str(self._counter)
            job = dict(view=view, settings=settings, future=future,
                       started=reusable["started"] if reusable is not None else monotonic())
            if reusable is not None and "finished" in reusable:
                job["finished"] = reusable["finished"]
            self._jobs[identifier] = job
            future.add_done_callback(lambda done: job.setdefault("finished", monotonic()))
            self._latest = identifier
            while len(self._jobs) > 3:
                self._jobs.pop(next(iter(self._jobs)))
            return dict(job_id=identifier, state="computing")

    def read_job(self, identifier, channel):
        with self._lock:
            if identifier != self._latest or identifier not in self._jobs:
                return dict(state="superseded", job_id=identifier)
            if channel not in self.metadata["channels"]:
                raise ValueError("Unknown channel.")
            job = self._jobs[identifier]
            future = job["future"]
            response = dict(job_id=identifier, elapsed_seconds=int(job.get("finished", monotonic()) - job["started"]))
            if not future.done():
                return dict(response, state="computing", stage="Computing continuous power" if job["view"] == "continuous" else "Preparing cluster detail")
            try:
                data = future.result()
                if job["view"] == "continuous":
                    from .browser_data import continuous_payload
                    payload = continuous_payload(data, self.request["result"], channel)
                else:
                    payload = dict(image_url=f"api/image/{identifier}", **job["settings"])
                return dict(response, state="ready", data=payload)
            except Exception as error:
                return dict(response, state="failed", error=str(error))

    def detail_image(self, identifier):
        with self._lock:
            job = self._jobs.get(identifier)
            if identifier != self._latest or job is None or job["view"] != "detail" or not job["future"].done():
                raise ValueError("Detail image is not available.")
            return job["future"].result()

    def clear(self):
        with self._lock:
            for job in self._jobs.values():
                job["future"].cancel()
            self._latest = None
            self._jobs.clear()
            self.computation.clear()
            return dict(state="cleared")

    def export(self, body):
        if not isinstance(body, dict) or set(body) != {"settings", "images"}:
            raise ValueError("Export requires settings and images.")
        settings, images = body["settings"], body["images"]
        if not isinstance(settings, dict) or not isinstance(images, dict) or not images or set(images) - {"brain", "curves"}:
            raise ValueError("Only brain and curves PNG exports are accepted.")
        encoded_settings = json.dumps(settings, allow_nan=False, indent=2)
        if len(encoded_settings) > 100_000:
            raise ValueError("Export settings are too large.")
        decoded = {}
        for name, source in images.items():
            prefix = "data:image/png;base64,"
            if not isinstance(source, str) or not source.startswith(prefix):
                raise ValueError("PNG data URLs are required.")
            try:
                value = base64.b64decode(source[len(prefix):], validate=True)
            except ValueError as error:
                raise ValueError("Invalid PNG encoding.") from error
            if not value.startswith(b"\x89PNG\r\n\x1a\n") or len(value) > 8 * 1024**2:
                raise ValueError("Invalid or oversized PNG image.")
            decoded[name] = value
        target = Path(self.request["output_dir"]).resolve() / "browser_exports" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target.mkdir(parents=True, exist_ok=False)
        for name, value in decoded.items():
            (target / f"{name}.png").write_bytes(value)
        (target / "settings.json").write_text(encoded_settings, encoding="utf-8")
        return dict(directory=str(target), files=[f"{name}.png" for name in decoded] + ["settings.json"])
