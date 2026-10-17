"""Golden no-drift regression: replay deterministic games through build/WordGame
and require byte-identical action traces to tests/golden/*.txt (captured from the
pre-refactor binary and the O-01C submission package by tests/golden_lib.py).

The games are small enough that every turn finishes far inside its wall-clock
slice, so the policy stack decides from the deterministic cost model; a mismatch
means behaviour changed, not timing. If the very first game already differs from
itself across two runs, the machine is too loaded for this test - that is
reported as a separate failure mode.
"""
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selfplay

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "build" / "WordGame"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


@unittest.skipUnless(EXE.exists(), "build/WordGame not built (run make build)")
@unittest.skipUnless(selfplay.DICTIONARY.exists(), "tools/enable2k.txt missing (see tools/README.md)")
class GoldenTraceTest(unittest.TestCase):
    def test_invalid_flag_rejected(self):
        proc = subprocess.run([str(EXE), "--bogus-flag"], input=b"", capture_output=True)
        self.assertEqual(proc.returncode, 2)

    def test_traces_match_golden(self):
        for name, n, t, x, penalty, density, seed, flags in selfplay.GAMES:
            with self.subTest(game=name):
                trace = selfplay.run_game(EXE, n, t, x, penalty, density, seed, flags)
                golden = (GOLDEN_DIR / (name + ".txt")).read_text().splitlines()
                header, expected = golden[0], golden[1:]
                self.assertEqual(trace, expected,
                                 "solver behaviour drifted from the golden trace "
                                 + name + " (" + header + ")")


if __name__ == "__main__":
    unittest.main()
