"""Mechanism-level check for Tier 2.5: with a REAL wall-clock deadline (not
max_passes), does escalating reroute_pair's freed-tile-set (widen_strategy /
max_widen_rounds) net-improve score, or does the extra per-call Dijkstra
cost (measured separately in benchmark_widen_cost.py) crowd out enough
passes to net-hurt it? Mirrors benchmark_pair_operators.py's structure and
output columns, extended with the widen config as an explicit row label so
baseline (today's Tier 2) sits directly above each candidate configuration.

Usage: python3 scripts/benchmark_widen_reroute.py [seconds_per_run]
"""
import pathlib
import sys
from random import Random
from time import perf_counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from hextiles.geometry import S  # noqa: E402
from hextiles.incremental import IncrementalEvaluator  # noqa: E402
from hextiles.solver import (  # noqa: E402
    KICK_PATIENCE,
    candidate_orders,
    construct_grid,
    drop_pair,
    evaluate_grid,
    kick_grid,
    kick_pairs,
    reroute_pair,
    safe_move_budget,
)
from test_incremental import build_target_matches, make_random_state  # noqa: E402

WIDEN_PAIRS_PER_ROUND = 5

# (label, widen_strategy, max_widen_rounds) -- widen_strategy=None ignores
# max_widen_rounds (reroute_pair's own defensive fallback), used here for
# the baseline row.
CONFIGS = [
    ("baseline", None, 0),
    ("random r1", "random", 1),
    ("random r2", "random", 2),
    ("spatial r1", "spatial", 1),
    ("spatial r2", "spatial", 2),
    ("frontier r1", "frontier", 1),
    ("frontier r2", "frontier", 2),
]


def improve_grid_counted(
    state, exits, exit_ids, initial_grid, evaluation, deadline,
    widen_strategy, max_widen_rounds,
):
    """Instrumented mirror of solver.improve_grid (which exposes no
    counters), extended with the Tier 2.5 escalation params -- same
    approach benchmark_pair_operators.py already uses for tier1 vs tier2."""
    tile_locations = [
        (row, column)
        for row in range(state.w)
        for column in range(state.w)
        if state.grid[row][column] >= 0
    ]
    pair_order = list(state.pairs)
    safe_budget = safe_move_budget(state.n)
    random_source = Random(len(tile_locations) * state.n + len(state.pairs))
    target_matches = build_target_matches(state)
    evaluator = IncrementalEvaluator(state, exits, exit_ids, initial_grid, target_matches)
    current_evaluation = evaluation
    best_grid = [row[:] for row in state.grid]
    best_evaluation = evaluation
    stale_kicks = 0
    pass_count = 0
    tile_kick_count = 0
    pair_kick_count = 0
    reroute_attempts = 0
    reroute_hits = 0
    drop_attempts = 0
    drop_hits = 0

    def snapshot_if_best():
        nonlocal best_grid, best_evaluation, stale_kicks
        if current_evaluation > best_evaluation:
            best_grid = [row[:] for row in state.grid]
            best_evaluation = current_evaluation
            stale_kicks = 0

    def counters():
        return (
            best_evaluation, pass_count, tile_kick_count, pair_kick_count,
            reroute_attempts, reroute_hits, drop_attempts, drop_hits,
        )

    while perf_counter() < deadline:
        random_source.shuffle(tile_locations)
        improved = False
        for row, column in tile_locations:
            if perf_counter() >= deadline:
                snapshot_if_best()
                return counters()
            original_orientation = state.grid[row][column]
            best_local_evaluation = current_evaluation
            best_result = None
            for target_orientation in range(S):
                if target_orientation == original_orientation:
                    continue
                result = evaluator.preview_flips({(row, column): target_orientation})
                if result.evaluation > best_local_evaluation:
                    best_local_evaluation = result.evaluation
                    best_result = result
            if best_result is not None:
                evaluator.commit_flips(best_result)
                current_evaluation = best_local_evaluation
                improved = True

        random_source.shuffle(pair_order)
        for first_exit, second_exit in pair_order:
            if perf_counter() >= deadline:
                snapshot_if_best()
                return counters()
            was_matched = evaluator.path_of[first_exit].matched
            if was_matched:
                drop_attempts += 1
                result = drop_pair(state, evaluator, first_exit, second_exit)
            else:
                reroute_attempts += 1
                result = reroute_pair(
                    state, exits, exit_ids, evaluator, first_exit, second_exit, safe_budget,
                    random_source=random_source,
                    widen_strategy=widen_strategy,
                    max_widen_rounds=max_widen_rounds,
                    widen_pairs_per_round=WIDEN_PAIRS_PER_ROUND,
                )
            if result is not None and result.evaluation > current_evaluation:
                if was_matched:
                    drop_hits += 1
                else:
                    reroute_hits += 1
                evaluator.commit_flips(result)
                current_evaluation = result.evaluation
                improved = True

        snapshot_if_best()
        pass_count += 1
        if not improved:
            stale_kicks += 1
            if perf_counter() >= deadline:
                break
            if stale_kicks > KICK_PATIENCE:
                state.grid = [row[:] for row in best_grid]
                evaluator = IncrementalEvaluator(state, exits, exit_ids, initial_grid, target_matches)
                current_evaluation = best_evaluation
                stale_kicks = 0
            if random_source.random() < 0.5:
                kick_flips = kick_pairs(evaluator, state.pairs, random_source)
                pair_kick_count += 1
            else:
                kick_flips = kick_grid(tile_locations, random_source)
                tile_kick_count += 1
            current_evaluation = evaluator.apply_flips(kick_flips)

    snapshot_if_best()
    return counters()


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    print(f"per-run budget: {budget}s\n")
    header = (
        f"{'N':>4} {'config':>12} {'passes':>7} {'tkicks':>7} {'pkicks':>7} "
        f"{'reroute':>12} {'drop':>12} {'score':>10} {'matched':>8} {'total':>8} {'-moves':>7}"
    )
    print(header)
    for n in (8, 13, 16, 20):
        state, exits, exit_ids = make_random_state(n, seed=n * 500 + 9, bonus_count=5)
        initial_grid = [row[:] for row in state.grid]
        orders = candidate_orders(state, exits)
        state.grid = [row[:] for row in initial_grid]
        construct_grid(state, exits, exit_ids, orders[0])
        constructed_grid = [row[:] for row in state.grid]
        base_evaluation = evaluate_grid(state, exits, exit_ids, initial_grid)
        initial_matched = base_evaluation[1]
        print(f"# N={n}: {initial_matched}/{len(state.pairs)} pairs matched after construction")

        for label, widen_strategy, max_widen_rounds in CONFIGS:
            state.grid = [row[:] for row in constructed_grid]
            deadline = perf_counter() + budget
            (
                score_eval, passes, tkicks, pkicks,
                reroute_attempts, reroute_hits, drop_attempts, drop_hits,
            ) = improve_grid_counted(
                state, exits, exit_ids, initial_grid, base_evaluation, deadline,
                widen_strategy, max_widen_rounds,
            )
            reroute_str = f"{reroute_hits}/{reroute_attempts}"
            drop_str = f"{drop_hits}/{drop_attempts}"
            print(
                f"{n:>4} {label:>12} {passes:>7} {tkicks:>7} {pkicks:>7} "
                f"{reroute_str:>12} {drop_str:>12} "
                f"{score_eval[0]:>10} {score_eval[1]:>8} {score_eval[2]:>8} {score_eval[3]:>7}"
            )
        print()


if __name__ == "__main__":
    main()
