#!/usr/bin/env python3
"""Run Topcoder's offline tester over a range of seeds and print one CSV row each.

    python3 scripts/run_tester.py --exe build/WordGame --seeds 1:100 --jobs 4 > runs/scores.csv
    python3 scripts/run_tester.py --exe build/WordGame --seeds 7 --args="--preset int-01 --diagnostics"

Needs tools/tester.jar and tools/enable2k.txt (see tools/README.md) and Java 11+.
The tester runs with the official 10 s limit from inside tools/, where it
expects the dictionary. The solver's search is budgeted in deterministic effort
units, so parallel jobs do not change its moves unless the machine is so
loaded that the wall-clock guard engages (the diagnostics' `sched_guard_turns`).
"""
import argparse
import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SCORE = re.compile(r"Score\s*=\s*(-?[\d.]+)")
RUNTIME = re.compile(r"RunTime\s*=\s*(\d+)")


def seeds_of(spec):
    if ":" in spec:
        lo, hi = map(int, spec.split(":"))
        return list(range(lo, hi + 1))
    return [int(s) for s in spec.split(",")]


def java():
    for cand in (os.environ.get("JAVA"), shutil.which("java"), "/opt/homebrew/opt/openjdk/bin/java"):
        if cand and subprocess.run([cand, "-version"], capture_output=True).returncode == 0:
            return cand
    raise SystemExit("no working java found; set JAVA=/path/to/java")


def run(seed, exe, args, java_bin, save):
    command = " ".join([shlex.quote(str(exe.resolve()))] + shlex.split(args))
    cmd = [java_bin, "-Djava.awt.headless=true", "-jar", str(TOOLS / "tester.jar"), "-exec", command,
           "-seed", str(seed), "-tl", "10000", "-novis", "-noanimate", "-pr"]
    if save:
        save.mkdir(parents=True, exist_ok=True)
        cmd += ["-so", str(save.resolve())]
    proc = subprocess.run(cmd, cwd=TOOLS, capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    if save:
        # The tester relays the solver's standard error into its own output.
        diag = [line for line in text.splitlines() if line.startswith("DIAG ")]
        if diag:
            (save / ("%d.diag" % seed)).write_text("\n".join(diag) + "\n")
    score, runtime = SCORE.search(text), RUNTIME.search(text)
    return dict(seed=seed, score=score.group(1) if score else "", runtime_ms=runtime.group(1) if runtime else "",
                ok=int(bool(score) and float(score.group(1)) >= 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", type=Path, default=ROOT / "build" / "WordGame")
    ap.add_argument("--seeds", default="1:10", help="lo:hi or a comma-separated list")
    ap.add_argument("--args", default="", help="solver arguments, e.g. '--preset int-01'")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--save", type=Path, help="directory for solver outputs and DIAG lines")
    a = ap.parse_args()
    for need in ("tester.jar", "enable2k.txt"):
        if not (TOOLS / need).exists():
            raise SystemExit("tools/%s missing; see tools/README.md" % need)
    java_bin = java()
    writer = csv.DictWriter(sys.stdout, fieldnames=["seed", "score", "runtime_ms", "ok"])
    writer.writeheader()
    with ThreadPoolExecutor(a.jobs) as pool:
        for row in pool.map(lambda s: run(s, a.exe, a.args, java_bin, a.save), seeds_of(a.seeds)):
            writer.writerow(row)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
