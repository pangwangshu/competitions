"""Microbenchmark: full evaluate_grid vs. IncrementalEvaluator.preview_flips,
on a realistic (post-construct_grid) grid, across a range of N.

Usage: python3 scripts/benchmark_incremental.py
"""
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from hextiles.geometry import S  # noqa: E402
from hextiles.incremental import IncrementalEvaluator  # noqa: E402
from hextiles.solver import (  # noqa: E402
    candidate_orders,
    construct_grid,
    evaluate_grid,
)
from test_incremental import build_target_matches, make_random_state  # noqa: E402

TRIALS = 300


def benchmark(n, seed):
    state, exits, exit_ids = make_random_state(n, seed, bonus_count=5)
    initial_grid = [row[:] for row in state.grid]
    orders = candidate_orders(state, exits)
    state.grid = [row[:] for row in initial_grid]
    construct_grid(state, exits, exit_ids, orders[0])
    constructed_grid = [row[:] for row in state.grid]

    tile_locations = [
        (row, column)
        for row in range(state.w)
        for column in range(state.w)
        if state.grid[row][column] >= 0
    ]
    rng = random.Random(seed * 31 + 7)
    trial_flips = [
        {rng.choice(tile_locations): rng.randrange(S)} for _ in range(TRIALS)
    ]

    # Full evaluate_grid, mimicking the old improve_grid's per-trial mutate+call.
    state.grid = [row[:] for row in constructed_grid]
    started = time.perf_counter()
    for flip in trial_flips:
        for (row, column), orientation in flip.items():
            state.grid[row][column] = orientation
        evaluate_grid(state, exits, exit_ids, initial_grid)
    full_elapsed = time.perf_counter() - started
    full_per_call = full_elapsed / TRIALS

    # Incremental preview_flips (no commit -- matches the hot-loop trial pattern).
    state.grid = [row[:] for row in constructed_grid]
    target_matches = build_target_matches(state)
    evaluator = IncrementalEvaluator(state, exits, exit_ids, initial_grid, target_matches)
    started = time.perf_counter()
    for flip in trial_flips:
        evaluator.preview_flips(flip)
    preview_elapsed = time.perf_counter() - started
    preview_per_call = preview_elapsed / TRIALS

    # Incremental apply_flips (preview + commit), matching a committed move.
    state.grid = [row[:] for row in constructed_grid]
    evaluator = IncrementalEvaluator(state, exits, exit_ids, initial_grid, target_matches)
    started = time.perf_counter()
    for flip in trial_flips:
        evaluator.apply_flips(flip)
    apply_elapsed = time.perf_counter() - started
    apply_per_call = apply_elapsed / TRIALS

    return full_per_call, preview_per_call, apply_per_call


def main():
    print(f"{'N':>4} {'W':>4} {'full evaluate_grid':>20} {'preview_flips':>16} {'apply_flips':>14} {'speedup (preview)':>18}")
    for n in (5, 10, 16, 20):
        from hextiles.geometry import grid_width
        w = grid_width(n)
        full_per_call, preview_per_call, apply_per_call = benchmark(n, seed=n * 1000 + 3)
        speedup = full_per_call / preview_per_call
        print(
            f"{n:>4} {w:>4} {full_per_call * 1e6:>17.1f} us {preview_per_call * 1e6:>13.1f} us "
            f"{apply_per_call * 1e6:>11.1f} us {speedup:>17.2f}x"
        )


if __name__ == "__main__":
    main()
