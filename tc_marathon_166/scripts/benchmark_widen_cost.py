"""Microbenchmark: how much more expensive does reroute_pair get as the
Tier 2.5 widen strategy escalates across more rounds, on a realistic
post-construction grid? Reports per-call cost AND hit rate together so a
cost increase can be weighed against what it actually buys.

Run once over ALL currently-unmatched pairs per config (not a fixed trial
count) -- reroute_pair never mutates state/evaluator on return (verified
by tests/test_pair_operators.py), so repeated calls across configs against
the same starting grid are independent and directly comparable, same
spirit as benchmark_incremental.py's tight timing loop.

Usage: python3 scripts/benchmark_widen_cost.py
"""
import pathlib
import sys
import time
from random import Random

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from hextiles.incremental import IncrementalEvaluator  # noqa: E402
from hextiles.solver import (  # noqa: E402
    candidate_orders,
    construct_grid,
    reroute_pair,
    safe_move_budget,
)
from test_incremental import build_target_matches, make_random_state  # noqa: E402

WIDEN_PAIRS_PER_ROUND = 5
ROUND_SWEEP = (1, 2, 3)
STRATEGIES = ("random", "spatial", "frontier")


def timed_pass(state, exits, exit_ids, evaluator, unmatched_pairs, safe_budget,
                random_source, widen_strategy, max_widen_rounds):
    started = time.perf_counter()
    hits = 0
    for first_exit, second_exit in unmatched_pairs:
        result = reroute_pair(
            state, exits, exit_ids, evaluator, first_exit, second_exit, safe_budget,
            random_source=random_source, widen_strategy=widen_strategy,
            max_widen_rounds=max_widen_rounds, widen_pairs_per_round=WIDEN_PAIRS_PER_ROUND,
        )
        if result is not None:
            hits += 1
    elapsed = time.perf_counter() - started
    return elapsed / len(unmatched_pairs), hits


def benchmark(n, seed):
    state, exits, exit_ids = make_random_state(n, seed, bonus_count=5)
    initial_grid = [row[:] for row in state.grid]
    orders = candidate_orders(state, exits)
    state.grid = [row[:] for row in initial_grid]
    construct_grid(state, exits, exit_ids, orders[0])
    target_matches = build_target_matches(state)
    evaluator = IncrementalEvaluator(state, exits, exit_ids, initial_grid, target_matches)
    safe_budget = safe_move_budget(n)
    unmatched_pairs = [pair for pair in state.pairs if not evaluator.path_of[pair[0]].matched]

    print(f"# N={n} seed={seed}: {len(unmatched_pairs)}/{len(state.pairs)} pairs unmatched after construction")
    if not unmatched_pairs:
        print("  (no unmatched pairs -- skipping)\n")
        return

    random_source = Random(seed * 71 + 3)
    baseline_us, baseline_hits = timed_pass(
        state, exits, exit_ids, evaluator, unmatched_pairs, safe_budget,
        random_source, None, 0,
    )
    print(f"  {'config':>22} {'us/call':>10} {'hits':>6} {'hit%':>7}")
    print(
        f"  {'baseline (round 0)':>22} {baseline_us * 1e6:>10.1f} {baseline_hits:>6} "
        f"{100 * baseline_hits / len(unmatched_pairs):>6.1f}%"
    )

    for strategy in STRATEGIES:
        for rounds in ROUND_SWEEP:
            per_call_us, hits = timed_pass(
                state, exits, exit_ids, evaluator, unmatched_pairs, safe_budget,
                random_source, strategy, rounds,
            )
            label = f"{strategy} rounds={rounds}"
            print(
                f"  {label:>22} {per_call_us * 1e6:>10.1f} {hits:>6} "
                f"{100 * hits / len(unmatched_pairs):>6.1f}%"
            )
    print()


def main():
    for n in (8, 13, 16, 20):
        benchmark(n, seed=n * 500 + 9)


if __name__ == "__main__":
    main()
