"""Own a complete display process tree, including spawn-bootstrap children."""
import ctypes
from ctypes import wintypes
import os
import signal
import time


class _BasicLimits(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
        ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
        ("max_working_set", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
        ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("basic", _BasicLimits), ("io", _IoCounters),
        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in
        ("user_time", "kernel_time", "period_user_time", "period_kernel_time")] + [
        (name, wintypes.DWORD) for name in ("page_faults", "total", "active", "terminated")]


class DisplayProcessTree:
    """Assign ownership before the launch gate permits GUI/numerical children."""

    def __init__(self, child):
        self.pid = child.pid
        self.job = None
        if os.name != "nt":
            return
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "QueryInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = arguments, result
        job = self.api.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = _ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.api.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not self.api.AssignProcessToJobObject(job, wintypes.HANDLE(int(child._handle))):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.api.CloseHandle(job)
            raise
        self.job = job

    def close(self):
        if os.name != "nt":
            if self.pid is not None:
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.pid = None
            return
        if self.job is None:
            return
        job, self.job = self.job, None
        try:
            if not self.api.TerminateJobObject(job, 1):
                raise ctypes.WinError(ctypes.get_last_error())
            deadline = time.monotonic() + 5
            while True:
                accounting = _Accounting()
                if not self.api.QueryInformationJobObject(job, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                if accounting.active == 0:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Owned display process tree did not exit within five seconds.")
                time.sleep(.01)
        finally:
            self.api.CloseHandle(job)
