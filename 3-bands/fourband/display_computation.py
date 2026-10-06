"""Filtering and smoothing in a process that never runs native GUI events."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import threading

import joblib

from audio_cross_3band_utils import load_ecog_fif, load_speech_events, extract_audio_envelope
from .continuous import ContinuousPowerCache, prepare_continuous_trials


_SESSION = None


def numerical_executable():
    if os.name == "nt":
        executable = Path(sys.executable).with_name("pythonw.exe")
        if not executable.is_file():
            raise FileNotFoundError(f"Consoleless interpreter is missing: {executable}")
        return str(executable)
    return sys.executable


def _watch_display_parent():
    parent = multiprocessing.parent_process()
    if parent is not None:
        def stop_when_parent_exits():
            parent.join()
            os._exit(0)
        threading.Thread(target=stop_when_parent_exits, daemon=True, name="display-parent-watch").start()


def prepare_snapshot(request_path, settings):
    global _SESSION
    path = Path(request_path).resolve()
    if _SESSION is None or _SESSION[0] != path:
        request = joblib.load(path)
        sources = request["sources"]
        raw = load_ecog_fif(sources["fif_path"], preload=True)
        events = load_speech_events(sources["onset_path"], sources["offset_path"])
        audio = extract_audio_envelope(sources["wav_path"])
        cache = ContinuousPowerCache(max_bytes=int(request["cache_max_mb"] * 1024**2))
        _SESSION = (path, raw, events, audio, request["extraction"], cache, {})
    _, raw, events, audio, extraction, cache, last_display = _SESSION
    reused = last_display.get("settings") == settings
    if reused:
        data = last_display["data"]
    else:
        data = prepare_continuous_trials(raw, events, audio, extraction, cache=cache, **settings)
        last_display.update(settings=dict(settings), data=data)
    data.metadata["power_cache"] = cache.info()
    data.metadata["continuous_reused"] = reused
    data.metadata["numerical_pid"] = os.getpid()
    return data


def clear_snapshot_cache():
    if _SESSION is not None:
        _SESSION[-2].clear()
        _SESSION[-1].clear()


class DisplayComputation:
    """Send only small display settings across the per-session process boundary."""

    def __init__(self, request_path):
        self.request_path = Path(request_path)
        self.pool = None
        self.failure = None

    def _remember_result(self, future, *, reset_success=True):
        if not future.cancelled():
            error = future.exception()
            if error is not None or reset_success:
                self.failure = error

    def raise_if_failed(self):
        if self.failure is not None:
            raise RuntimeError(f"Numerical display process failed: {self.failure}") from self.failure

    def submit(self, settings):
        try:
            if self.pool is None:
                os.environ["FOURBAND_NUMERICAL_LOG_DIR"] = str(self.request_path.resolve().parent)
                multiprocessing.set_executable(numerical_executable())
                self.pool = ProcessPoolExecutor(max_workers=1,
                    mp_context=multiprocessing.get_context("spawn"), initializer=_watch_display_parent)
            future = self.pool.submit(prepare_snapshot, self.request_path, settings)
        except Exception as error:
            self.failure = error
            raise
        future.add_done_callback(self._remember_result)
        return future

    def clear(self):
        if self.pool is not None:
            try:
                future = self.pool.submit(clear_snapshot_cache)
            except Exception as error:
                self.failure = error
                raise
            future.add_done_callback(lambda done: self._remember_result(done, reset_success=False))
            return future
        return None

    def close(self):
        if self.pool is not None:
            pool = self.pool
            pool.shutdown(wait=True, cancel_futures=True)
            broken = getattr(pool, "_broken", False)
            if broken:
                self.failure = RuntimeError(str(broken))
            self.pool = None
