"""Generate kaggle/submit-blend/kernel.py: nartaa's public CoAtNet blended with our reader.

  python scripts/build_blend_kernel.py runs/t4-scope/nartaa-0949/rsna-knee-0949-anatomical-mirror.ipynb

Source notebook: nartaa/rsna-knee-0949-anatomical-mirror (public; its script is embedded verbatim,
only its output path is patched). The kernel runs the CoAtNet in a subprocess, then our checkpoints
from the config (as kaggle/submit-own does), and writes rank(w * rank(CoAtNet) + (1 - w) * rank(ours))
per finding. The weight is fixed in the config before submission (S6 plan: w = 0.7), never tuned on
the public LB. On the hidden set a failed side falls back to the other; on the 3-study commit run
any failure raises, so a fallback cannot pass as a blend.
"""

import json
import sys
from pathlib import Path

from build_teacher_kernel import WEIGHT_SHA  # same notebook, same pinned checkpoint as the T4 teacher

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parent / "blend_template.py.txt"


def build(notebook):
    cells = json.loads(Path(notebook).read_text())["cells"]
    script = "".join(cells[4]["source"])
    if WEIGHT_SHA not in "".join(cells[3]["source"]):
        raise ValueError("The notebook's pinned checkpoint hash changed")
    old = 'out = "/kaggle/working/submission.csv"'
    if script.count(old) != 1:
        raise ValueError(f"Patch anchor not unique: {old!r}")
    script = script.replace(old, 'out = os.environ["RK_COATNET_OUT"]')
    compile(script, "coatnet", "exec")
    outer = TEMPLATE.read_text()
    if outer.count("__COATNET_SOURCE__") != 1 or outer.count("__WEIGHT_SHA__") != 1:
        raise ValueError("Template placeholders are missing")
    outer = outer.replace("__COATNET_SOURCE__", repr(script)).replace("__WEIGHT_SHA__", WEIGHT_SHA)
    out = ROOT / "kaggle" / "submit-blend" / "kernel.py"
    out.write_text(outer)
    compile(outer, str(out), "exec")
    print(out, len(outer), "chars")


if __name__ == "__main__":
    build(sys.argv[1])
