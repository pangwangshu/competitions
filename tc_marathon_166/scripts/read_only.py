"""A do-nothing solver: reads one case and outputs zero moves.

Used with the tester's -showOriginal and -saveSolInput flags to capture each
seed's hidden (pre-shuffle) grid; see scripts/summarize_run.py.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from hextiles.model import read_input  # noqa: E402

read_input(sys.stdin)
print(0, flush=True)
