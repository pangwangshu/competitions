"""Generate kaggle/submit-blend-s6/kernel.py: the no-OAI final ② candidate, nartaa's S6 ConvNeXt blended with our reader.

  python scripts/build_blend_s6_kernel.py runs/t4s6-scope/rsna-knee-s6fall-solo.py

Same source as the T4 teacher (build_teacher_s6_kernel.py: component A's CoAtNet source blanked), here on the test studies.
The kernel writes rank(w * rank(S6) + (1 - w) * rank(ours)) per finding; w is fixed in the config before submission, never
tuned on the public LB. A study S6 cannot read takes our rank.
"""

import sys
from pathlib import Path

from build_teacher_s6_kernel import s6_source

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parent / "blend_s6_template.py.txt"


def build(script):
    text = s6_source(script)
    outer = TEMPLATE.read_text()
    if outer.count("__K1_SOURCE__") != 1:
        raise ValueError("Template placeholder missing")
    outer = outer.replace("__K1_SOURCE__", repr(text))
    out = ROOT / "kaggle" / "submit-blend-s6" / "kernel.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(outer)
    compile(outer, str(out), "exec")
    print(out, len(outer), "chars")


if __name__ == "__main__":
    build(sys.argv[1])
