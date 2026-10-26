"""Data and artifact roots, configurable through RK_DATA/WORK/RUNS/EXT.

Local defaults are relative to the current working directory (run from the
repository root). On Kaggle, both competition input layouts are supported.
"""

import os
from pathlib import Path


def default_paths():
    if Path("/kaggle/working").is_dir():
        inputs = Path("/kaggle/input")
        data = inputs / "competitions" / "rsna-knee-abnormality-detection"
        if not data.is_dir():
            data = inputs / "rsna-knee-abnormality-detection"
        return {
            "DATA": str(data),
            "WORK": "/kaggle/working",
            "RUNS": "/kaggle/working/runs",
            "EXT": str(inputs),
        }
    return {"DATA": "data/raw", "WORK": "runs/work", "RUNS": "runs", "EXT": "data/ext"}


_defaults = default_paths()
DATA, WORK, RUNS, EXT = (
    os.environ.get(f"RK_{name}", _defaults[name])
    for name in ("DATA", "WORK", "RUNS", "EXT")
)
