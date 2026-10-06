"""Restore file logging before scientific imports in consoleless spawn children."""
import faulthandler
import os
from pathlib import Path
import sys


def configure_numerical_logging():
    folder = os.environ.get("FOURBAND_NUMERICAL_LOG_DIR")
    if folder:
        directory = Path(folder)
        sys.stdout = (directory / "numerical_stdout.log").open("a", encoding="utf-8", buffering=1)
        sys.stderr = (directory / "numerical_stderr.log").open("a", encoding="utf-8", buffering=1)
        faulthandler.enable(file=sys.stderr)
