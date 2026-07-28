"""Makes backend/'s modules (app, queueing, tiling, ...) importable by plain
`import app` / `from queueing import ...` from tests/ without turning backend/
into a package -- test files were moved here from backend/'s top level, where
pytest's rootdir-based sys.path insertion made that work for free."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
