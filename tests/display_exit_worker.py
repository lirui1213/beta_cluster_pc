"""A diagnostic-only subprocess that exits while its parent stays alive."""
import os
import sys

if os.environ.get("FOURBAND_DISPLAY_GATE") == "1":
    sys.stdin.buffer.read(1)
print("Deliberate display process exit", file=sys.stderr, flush=True)
os._exit(17)
