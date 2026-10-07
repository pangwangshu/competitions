"""Paired evaluation of a candidate agent against a control agent.

Both agents play the same opponents on the same seeds from both seats, so each
game of the candidate has a matched game of the control; the statistic is the
paired difference in final margin (own money minus opponent money).

    python scripts/paired_eval.py --candidate ../my_variant --control . \\
        --opponents starter pass path/to/bot/main.py --seeds 0:20 --jobs 8

Reported:
  * mean paired delta-margin, with a 95% bootstrap CI resampled by seed (the two
    seats and all opponents of one seed are not independent);
  * flips (control lost, candidate won) and reversals (the opposite);
  * the share of pairs whose shop sequence is identical. The engine draws a weed
    check for every empty tile before drawing each new shop, so any change in
    land use can change the shops BOTH players get. Pairs whose shops diverge
    measure strategy plus a different environment and are much noisier.

Each game runs in a fresh process: the agent keeps per-game state, and two
copies of a package with the same name cannot share one process.
"""

import argparse
import multiprocessing as mp
import os
import random
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _play(spec):
    agent_dir, opponent, seed, seat = spec
    os.chdir(tempfile.mkdtemp())  # no stray package in cwd can shadow the agent's
    from play import run_game

    return run_game(os.path.join(agent_dir, "main.py"), opponent, seed, seat)


def _parse_seeds(text):
    if ":" in text:
        lo, hi = text.split(":")
        return list(range(int(lo), int(hi)))
    return [int(s) for s in text.split(",")]


def _bootstrap_ci(deltas_by_seed, reps=5000, rng=random.Random(0)):
    seeds = list(deltas_by_seed)
    means = []
    for _ in range(reps):
        sample = [d for s in (rng.choice(seeds) for _ in seeds) for d in deltas_by_seed[s]]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, help="directory containing the candidate main.py")
    ap.add_argument("--control", required=True, help="directory containing the control main.py")
    ap.add_argument("--opponents", nargs="+", default=["starter"])
    ap.add_argument("--seeds", default="0:10", help="'lo:hi' or comma list")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    cand, ctrl = os.path.abspath(args.candidate), os.path.abspath(args.control)
    opponents = [o if o in ("pass", "random", "starter") else os.path.abspath(o) for o in args.opponents]
    cells = [(o, s, seat) for o in opponents for s in _parse_seeds(args.seeds) for seat in (0, 1)]
    specs = [(d, o, s, seat) for (o, s, seat) in cells for d in (ctrl, cand)]
    with mp.get_context("spawn").Pool(args.jobs, maxtasksperchild=1) as pool:
        results = pool.map(_play, specs, chunksize=1)

    rows = []
    for i, (o, s, seat) in enumerate(cells):
        c, k = results[2 * i], results[2 * i + 1]
        rows.append(dict(opponent=os.path.basename(os.path.dirname(o)) if os.sep in o else o, seed=s, seat=seat,
                         control=c["margin"], candidate=k["margin"], delta=k["margin"] - c["margin"],
                         same_shops=c["shops"] == k["shops"]))

    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r["delta"])
    deltas = [r["delta"] for r in rows]
    lo, hi = _bootstrap_ci(by_seed)
    flips = sum(r["control"] < 0 < r["candidate"] for r in rows)
    reversals = sum(r["candidate"] < 0 < r["control"] for r in rows)
    print(f"pairs: {len(rows)}  ({len(opponents)} opponents x {len(by_seed)} seeds x 2 seats)")
    print(f"mean delta-margin: {statistics.mean(deltas):+,.0f}  95% CI [{lo:+,.0f}, {hi:+,.0f}] (seed bootstrap)")
    print(f"candidate better in {sum(d > 0 for d in deltas)}/{len(rows)}, worse in {sum(d < 0 for d in deltas)}")
    print(f"flips {flips}, reversals {reversals}, net {flips - reversals:+d}")
    print(f"identical shop sequence in {sum(r['same_shops'] for r in rows)}/{len(rows)} pairs")
    print("\nby opponent:")
    for o in sorted({r["opponent"] for r in rows}):
        sub = [r for r in rows if r["opponent"] == o]
        print(f"  {o:<28} n={len(sub):<4} delta {statistics.mean(r['delta'] for r in sub):+10,.0f}   "
              f"control wins {sum(r['control'] > 0 for r in sub)}, candidate wins {sum(r['candidate'] > 0 for r in sub)}")


if __name__ == "__main__":
    main()
