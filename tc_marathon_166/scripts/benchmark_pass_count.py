"""Mechanism-level check: with a REAL wall-clock deadline (not max_passes),
how many local-search passes/kicks does old vs. new improve_grid complete?
This is what should explain any score difference -- more throughput, not a
different algorithm.

Usage: python3 scripts/benchmark_pass_count.py [seconds_per_run]
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
from hextiles.solver import candidate_orders, construct_grid, evaluate_grid  # noqa: E402
from test_incremental import build_target_matches, make_random_state  # noqa: E402


def old_improve_grid_counted(state, exits, exit_ids, initial_grid, evaluation, deadline):
    KICK_PATIENCE = 6
    MAX_KICK_SIZE = 6
    tile_locations = [
        (row, column)
        for row in range(state.w)
        for column in range(state.w)
        if state.grid[row][column] >= 0
    ]
    random_source = Random(len(tile_locations) * state.n + len(state.pairs))
    current_evaluation = evaluation
    best_grid = [row[:] for row in state.grid]
    best_evaluation = evaluation
    stale_kicks = 0
    pass_count = 0
    kick_count = 0

    def snapshot_if_best():
        nonlocal best_grid, best_evaluation, stale_kicks
        if current_evaluation > best_evaluation:
            best_grid = [row[:] for row in state.grid]
            best_evaluation = current_evaluation
            stale_kicks = 0

    while perf_counter() < deadline:
        random_source.shuffle(tile_locations)
        improved = False
        for row, column in tile_locations:
            if perf_counter() >= deadline:
                snapshot_if_best()
                return best_evaluation, pass_count, kick_count
            original_orientation = state.grid[row][column]
            best_orientation = original_orientation
            best_local_evaluation = current_evaluation
            for target_orientation in range(S):
                if target_orientation == original_orientation:
                    continue
                state.grid[row][column] = target_orientation
                candidate_evaluation = evaluate_grid(state, exits, exit_ids, initial_grid)
                if candidate_evaluation > best_local_evaluation:
                    best_orientation = target_orientation
                    best_local_evaluation = candidate_evaluation
            state.grid[row][column] = best_orientation
            if best_local_evaluation > current_evaluation:
                current_evaluation = best_local_evaluation
                improved = True

        snapshot_if_best()
        pass_count += 1
        if not improved:
            stale_kicks += 1
            if perf_counter() >= deadline:
                break
            if stale_kicks > KICK_PATIENCE:
                state.grid = [row[:] for row in best_grid]
                current_evaluation = best_evaluation
                stale_kicks = 0
            kick_size = random_source.randint(1, min(MAX_KICK_SIZE, len(tile_locations)))
            for row, column in random_source.sample(tile_locations, kick_size):
                state.grid[row][column] = random_source.randrange(S)
            kick_count += 1
            current_evaluation = evaluate_grid(state, exits, exit_ids, initial_grid)

    snapshot_if_best()
    return best_evaluation, pass_count, kick_count


def new_improve_grid_counted(state, exits, exit_ids, initial_grid, evaluation, deadline):
    KICK_PATIENCE = 6
    MAX_KICK_SIZE = 6
    tile_locations = [
        (row, column)
        for row in range(state.w)
        for column in range(state.w)
        if state.grid[row][column] >= 0
    ]
    random_source = Random(len(tile_locations) * state.n + len(state.pairs))
    target_matches = build_target_matches(state)
    evaluator = IncrementalEvaluator(state, exits, exit_ids, initial_grid, target_matches)
    current_evaluation = evaluation
    best_grid = [row[:] for row in state.grid]
    best_evaluation = evaluation
    stale_kicks = 0
    pass_count = 0
    kick_count = 0

    def snapshot_if_best():
        nonlocal best_grid, best_evaluation, stale_kicks
        if current_evaluation > best_evaluation:
            best_grid = [row[:] for row in state.grid]
            best_evaluation = current_evaluation
            stale_kicks = 0

    while perf_counter() < deadline:
        random_source.shuffle(tile_locations)
        improved = False
        for row, column in tile_locations:
            if perf_counter() >= deadline:
                snapshot_if_best()
                return best_evaluation, pass_count, kick_count
            original_orientation = state.grid[row][column]
            best_orientation = original_orientation
            best_local_evaluation = current_evaluation
            best_result = None
            for target_orientation in range(S):
                if target_orientation == original_orientation:
                    continue
                result = evaluator.preview_flips({(row, column): target_orientation})
                if result.evaluation > best_local_evaluation:
                    best_orientation = target_orientation
                    best_local_evaluation = result.evaluation
                    best_result = result
            if best_result is not None:
                evaluator.commit_flips(best_result)
                current_evaluation = best_local_evaluation
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
            kick_size = random_source.randint(1, min(MAX_KICK_SIZE, len(tile_locations)))
            kick_flips = {
                (row, column): random_source.randrange(S)
                for row, column in random_source.sample(tile_locations, kick_size)
            }
            kick_count += 1
            current_evaluation = evaluator.apply_flips(kick_flips)

    snapshot_if_best()
    return best_evaluation, pass_count, kick_count


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    print(f"per-run budget: {budget}s\n")
    print(f"{'N':>4} {'impl':>6} {'passes':>8} {'kicks':>7} {'score':>12} {'matched':>9} {'total':>10} {'-moves':>8}")
    for n in (8, 13, 16, 20):
        state, exits, exit_ids = make_random_state(n, seed=n * 500 + 9, bonus_count=5)
        initial_grid = [row[:] for row in state.grid]
        orders = candidate_orders(state, exits)
        state.grid = [row[:] for row in initial_grid]
        construct_grid(state, exits, exit_ids, orders[0])
        constructed_grid = [row[:] for row in state.grid]
        base_evaluation = evaluate_grid(state, exits, exit_ids, initial_grid)

        state.grid = [row[:] for row in constructed_grid]
        deadline = perf_counter() + budget
        old_eval, old_passes, old_kicks = old_improve_grid_counted(
            state, exits, exit_ids, initial_grid, base_evaluation, deadline
        )
        print(f"{n:>4} {'old':>6} {old_passes:>8} {old_kicks:>7} {old_eval[0]:>12} {old_eval[1]:>9} {old_eval[2]:>10} {old_eval[3]:>8}")

        state.grid = [row[:] for row in constructed_grid]
        deadline = perf_counter() + budget
        new_eval, new_passes, new_kicks = new_improve_grid_counted(
            state, exits, exit_ids, initial_grid, base_evaluation, deadline
        )
        print(f"{n:>4} {'new':>6} {new_passes:>8} {new_kicks:>7} {new_eval[0]:>12} {new_eval[1]:>9} {new_eval[2]:>10} {new_eval[3]:>8}")
        print()


if __name__ == "__main__":
    main()
