"""Run real Tk lifecycle regressions outside Windows fd capture."""
from pathlib import Path
import subprocess
import sys


def test_native_tk_close_and_static_export_regressions():
    worker = Path(__file__).with_name("fourband_tk_lifecycle_checks.py")
    result = subprocess.run([sys.executable, "-X", "faulthandler", "-m", "pytest", str(worker),
                             "-q", "--capture=sys"],
                            cwd=worker.parents[1], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout
