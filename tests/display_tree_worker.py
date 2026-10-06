"""Diagnostic-only owned descendant used to verify interruption cleanup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if os.environ.get("FOURBAND_DISPLAY_GATE") == "1":
    sys.stdin.buffer.read(1)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
Path(sys.argv[1]).with_name("descendant.json").write_text(json.dumps({"pid": child.pid}), encoding="utf-8")
time.sleep(60)
