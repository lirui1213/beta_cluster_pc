"""Exercise pending native UI without embedding Tk in the parent test runner."""
from pathlib import Path
import subprocess
import sys


def test_native_pending_curve_ui():
    worker = Path(__file__).with_name("fourband_explorer_loading_checks.py")
    result = subprocess.run([sys.executable, "-m", "pytest", str(worker), "-q", "--capture=sys"],
                            cwd=worker.parents[1], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "5 passed" in result.stdout
