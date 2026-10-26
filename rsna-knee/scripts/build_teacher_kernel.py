"""Generate kaggle/teacher/kernel.py: nartaa's public 0.949 CoAtNet scoring script, run on TRAIN studies.

  python scripts/build_teacher_kernel.py runs/t4-scope/nartaa-0949/rsna-knee-0949-anatomical-mirror.ipynb

Source notebook: nartaa/rsna-knee-0949-anatomical-mirror (public; its script is embedded verbatim).
Patches to the script: the data root points at a scratch directory that presents training studies
as test.csv/test_series (the two series tables share one schema), and the per-study probabilities
are saved before the script converts them to ranks. The outer kernel runs the script on chunks of
studies in fresh subprocesses: its memory grows with the number of studies (it was written for
~1,300) and one process over the full training set was killed at ~870. Used for S5 T4.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = Path(__file__).resolve().parent / "teacher_templates"
WEIGHT_SHA = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
SAVE = (TEMPLATES / "save_patch.py.txt").read_text()


def build(notebook):
    cells = json.loads(Path(notebook).read_text())["cells"]
    script = "".join(cells[4]["source"])
    if WEIGHT_SHA not in "".join(cells[3]["source"]):
        raise ValueError("The notebook's pinned checkpoint hash changed")
    anchors = {
        "def find_test_root():\n": "def find_test_root():\n    return str(_t_root)\n\n\ndef _original_find_test_root():\n",
        "    models.clear(); gc.collect()\n": SAVE,
        'out = "/kaggle/working/submission.csv"': 'out = "/tmp/rk_teacher_ranks.csv"',
    }
    for old, new in anchors.items():
        if script.count(old) != 1:
            raise ValueError(f"Patch anchor not unique: {old!r}")
        script = script.replace(old, new)
    inner = (TEMPLATES / "inner_preamble.py.txt").read_text() + script
    compile(inner, "inner", "exec")
    outer = (TEMPLATES / "outer.py.txt").read_text()
    if outer.count("__INNER_SOURCE__") != 1 or outer.count("__WEIGHT_SHA__") != 1:
        raise ValueError("Outer template placeholders are missing")
    outer = outer.replace("__INNER_SOURCE__", repr(inner)).replace("__WEIGHT_SHA__", WEIGHT_SHA)
    out = ROOT / "kaggle" / "teacher" / "kernel.py"
    out.write_text(outer)
    compile(outer, str(out), "exec")
    print(out, len(outer), "chars")


if __name__ == "__main__":
    build(sys.argv[1])
