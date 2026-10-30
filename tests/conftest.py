"""Pytest configuration: make the suite fast, offline and headless.

Why
---
Two environment facts on this machine would otherwise make tests flaky or
slow:

* matplotlib's default cache directory is not writable in the sandbox, so the
  backend must be told where to write *before* ``import matplotlib.pyplot``.
  The same ``.cache/mpl`` directory the analysis scripts use is reused.
* test runs must never open a window or need a display, hence the Agg backend.

The repo root is put on ``sys.path`` so ``import fem`` works when pytest is
launched from any directory (``pytest`` from the root normally suffices, but
``python -m pytest tests/test_x.py`` should work too).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Must happen before matplotlib is imported anywhere (including by test
# modules collected later).
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl"))
(ROOT / ".cache" / "mpl").mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402  (environment must be configured first)

matplotlib.use("Agg", force=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
