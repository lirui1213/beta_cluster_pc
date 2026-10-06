"""Keep native event-loop regressions in a separate Windows process."""
from pathlib import Path
import subprocess
import sys


def test_native_explorer_async_regressions():
    worker = Path(__file__).with_name("fourband_reliability_gui_checks.py")
    result = subprocess.run([sys.executable, "-m", "pytest", str(worker), "-q", "--capture=sys"],
                            cwd=worker.parents[1], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "7 passed" in result.stdout
