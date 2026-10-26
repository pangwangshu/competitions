"""Generate kaggle/teacher-s6/kernel.py: nartaa's S6 ConvNeXt (public 0.950, no OAI) as the T4 teacher.

  python scripts/build_teacher_s6_kernel.py <kernels pull dir>/rsna-knee-s6fall-solo.py

Source: the public kernel nartaa/rsna-knee-s6fall-solo (weights nartaa/rsna-knee-s6fall-weights, code and anatomy network
heliosli/rsna-orthofoundation-fold0-epoch7-20261004). Its script is embedded unchanged except that the base64 source of
component A (the OAI-trained CoAtNet, used there only for studies B cannot read) is blanked; the teacher never runs A.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parent / "teacher_s6_template.py.txt"
MODEL_SHA = "0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12"


def s6_source(script):
    """nartaa's script with component A's embedded source blanked (shared with build_blend_s6_kernel.py)."""
    text = Path(script).read_text()
    if "W_A, W_B = 0.0, 1.0" not in text or MODEL_SHA not in text:
        raise ValueError("Not the S6-alone scoring script (weights or model hash changed)")
    text, n = re.subn(r'^A_B64 = "[^"]*"', 'A_B64 = ""  # component A (CoAtNet) removed for the T4 teacher', text, flags=re.M)
    if n != 1:
        raise ValueError("A_B64 constant not found exactly once")
    compile(text, "k1", "exec")
    return text


def build(script):
    text = s6_source(script)
    outer = TEMPLATE.read_text()
    if outer.count("__K1_SOURCE__") != 1:
        raise ValueError("Template placeholder missing")
    outer = outer.replace("__K1_SOURCE__", repr(text))
    out = ROOT / "kaggle" / "teacher-s6" / "kernel.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(outer)
    compile(outer, str(out), "exec")
    print(out, len(outer), "chars")


if __name__ == "__main__":
    build(sys.argv[1])
