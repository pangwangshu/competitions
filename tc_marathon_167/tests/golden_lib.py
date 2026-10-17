#!/usr/bin/env python3
"""Capture golden solver traces: run every GAMES config twice against a binary,
require both runs to be identical (self-determinism), and write the trace to
tests/golden/<name>.txt.  Usage: python3 tests/golden_lib.py [--exe PATH] [--force].
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selfplay

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def capture(exe, force=False):
    GOLDEN_DIR.mkdir(exist_ok=True)
    for name, n, t, x, penalty, density, seed, flags in selfplay.GAMES:
        out = GOLDEN_DIR / (name + ".txt")
        if out.exists() and not force:
            print("skip (exists):", name)
            continue
        first = selfplay.run_game(exe, n, t, x, penalty, density, seed, flags)
        second = selfplay.run_game(exe, n, t, x, penalty, density, seed, flags)
        if first != second:
            raise SystemExit("non-deterministic game: " + name)
        header = ["# n=%d t=%d x=%d penalty=%d density=%.2f seed=%d flags=%s"
                  % (n, t, x, penalty, density, seed, " ".join(flags) or "default")]
        out.write_text("\n".join(header + first) + "\n")
        print("captured %-16s %3d actions" % (name, len(first) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", type=Path, default=Path("build/WordGame"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    capture(args.exe, args.force)


if __name__ == "__main__":
    main()
