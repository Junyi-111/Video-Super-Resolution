"""Thin wrapper: run upstream-style course batch inference from repo root.

Delegates to ``streaming_distillation_upstream/examples/WanVSR/infer_course_dataset.py``
with the same command-line arguments (``--weights-dir``, ``--variant``, …).

Example::

    python -m part3_exploration.direction_d_distilled_streaming.run_course_infer --limit 1 --tiled
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]  # CV repo root (…/direction_d → part3_exploration → CV)
_INFER = _HERE / "streaming_distillation_upstream" / "examples" / "WanVSR" / "infer_course_dataset.py"


def main() -> None:
    subprocess.run([sys.executable, str(_INFER), *sys.argv[1:]], cwd=str(_REPO), check=True)


if __name__ == "__main__":
    main()
