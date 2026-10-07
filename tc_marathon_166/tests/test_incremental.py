"""Correctness harness for IncrementalEvaluator (Tier 1).

The failure mode this guards against is not "crashes" -- it's "silently
returns a plausible but wrong score tuple," which can drift back into
apparent agreement with ground truth by pure chance over a long run of
commits. So every check here compares against a *fresh* full evaluate_grid
call after *every single commit*, not just at the end of a run.
"""

import random

import pytest

from hextiles.geometry import S, get_end, get_start, grid_width
from hextiles.incremental import IncrementalEvaluator
from hextiles.model import GridState
from hextiles.solver import build_exits, evaluate_grid


def make_random_state(n, seed, bonus_count=3):
    rng = random.Random(seed)
    w = grid_width(n)
    grid = []
    valid_cells = []
    for r in range(w):
        start, end = get_start(r, n, w), get_end(r, n, w)
        row = []
        for c in range(w):
            if start <= c <= end:
                row.append(rng.randrange(S))
                valid_cells.append((r, c))
            else:
                row.append(-1)
        grid.append(row)

    bonus = set(rng.sample(valid_cells, min(bonus_count, len(valid_cells))))
    state = GridState(n=n, m=rng.randint(1, 5), b=len(bonus), pairs=[], grid=grid, bonus=bonus)
    exits, exit_ids = build_exits(state)

    exit_indices = list(range(len(exits)))
    rng.shuffle(exit_indices)
    state.pairs = [
        (exit_indices[i], exit_indices[i + 1]) for i in range(0, len(exit_indices) - 1, 2)
    ]
    return state, exits, exit_ids


def build_target_matches(state):
    target_matches = {}
    for first_exit, second_exit in state.pairs:
        target_matches[first_exit] = second_exit
        target_matches[second_exit] = first_exit
    return target_matches


def assert_matches_ground_truth(state, exits, exit_ids, initial_grid, evaluator):
    expected = evaluate_grid(state, exits, exit_ids, initial_grid)
    actual = evaluator.evaluation()
    assert actual == expected, (
        f"incremental evaluator diverged from evaluate_grid: cached={actual} full={expected}"
    )


def assert_internally_consistent(evaluator):
    for exit_id, record in evaluator.path_of.items():
        for tile in record.tiles:
            assert exit_id in evaluator.tile_owners.get(tile, ()), (
                f"tile_owners missing exit {exit_id} for tile {tile} it's recorded as touching"
            )
        if record.partner is None:
            continue
        partner_record = evaluator.path_of.get(record.partner)
        assert partner_record is not None, f"partner exit {record.partner} missing from path_of"
        assert partner_record.partner == exit_id, (
            f"asymmetric partner link: {exit_id}->{record.partner} "
            f"but {record.partner}->{partner_record.partner}"
        )
        assert partner_record.length == record.length
        assert partner_record.bonus_count == record.bonus_count
        assert partner_record.matched == record.matched
        assert partner_record.tiles == record.tiles


def run_random_walk(n, seed, num_commits, multi_owner_bias=0.6, kick_prob=0.35):
    state, exits, exit_ids = make_random_state(n, seed)
    initial_grid = [row[:] for row in state.grid]
    target_matches = build_target_matches(state)

    evaluator = IncrementalEvaluator(state, exits, exit_ids, initial_grid, target_matches)
    assert_matches_ground_truth(state, exits, exit_ids, initial_grid, evaluator)
    assert_internally_consistent(evaluator)

    rng = random.Random(seed * 7919 + 17)
    tile_locations = [
        (row, column)
        for row in range(state.w)
        for column in range(state.w)
        if state.grid[row][column] >= 0
    ]

    for _ in range(num_commits):
        if rng.random() < multi_owner_bias:
            candidates = [t for t in tile_locations if len(evaluator.tile_owners.get(t, ())) >= 2]
            if not candidates:
                candidates = tile_locations
        else:
            candidates = tile_locations

        if rng.random() < kick_prob:
            kick_size = min(rng.randint(2, 6), len(candidates))
            flip_tiles = rng.sample(candidates, kick_size)
        else:
            flip_tiles = [rng.choice(candidates)]

        flips = {tile: rng.randrange(S) for tile in flip_tiles}

        before_grid = [row[:] for row in state.grid]
        before_eval = evaluator.evaluation()
        result = evaluator.preview_flips(flips)
        assert state.grid == before_grid, "preview_flips must not leave state.grid mutated"
        assert evaluator.evaluation() == before_eval, "preview_flips must not mutate cached state"

        evaluator.commit_flips(result)
        assert_matches_ground_truth(state, exits, exit_ids, initial_grid, evaluator)
        assert_internally_consistent(evaluator)


WALK_CASES = [(n, seed) for n in (3, 4, 5, 6, 8, 10, 13, 16, 20) for seed in range(4)]


@pytest.mark.parametrize("n,seed", WALK_CASES)
def test_random_walk_matches_ground_truth(n, seed):
    run_random_walk(n, seed, num_commits=120)


LONG_WALK_CASES = [(6, 1001), (10, 1002), (16, 1003), (20, 1004)]


@pytest.mark.parametrize("n,seed", LONG_WALK_CASES)
def test_long_random_walk_matches_ground_truth(n, seed):
    run_random_walk(n, seed, num_commits=400)


def test_trial_pattern_matches_full_evaluate():
    """Mimics improve_grid's actual hot-loop usage: preview several candidate
    orientations for one tile against the same shared baseline (none
    committed), then commit only the winner. This is the exact access
    pattern that a subtly stateful preview_flips implementation could get
    wrong even if isolated single-flip checks pass."""
    state, exits, exit_ids = make_random_state(10, seed=99)
    initial_grid = [row[:] for row in state.grid]
    target_matches = build_target_matches(state)
    evaluator = IncrementalEvaluator(state, exits, exit_ids, initial_grid, target_matches)

    tile_locations = [
        (row, column)
        for row in range(state.w)
        for column in range(state.w)
        if state.grid[row][column] >= 0
    ]
    rng = random.Random(4242)

    for _ in range(50):
        row, column = rng.choice(tile_locations)
        original = state.grid[row][column]
        results = {}
        for target_orientation in range(S):
            if target_orientation == original:
                continue
            assert state.grid[row][column] == original
            result = evaluator.preview_flips({(row, column): target_orientation})
            results[target_orientation] = result

            state.grid[row][column] = target_orientation
            expected = evaluate_grid(state, exits, exit_ids, initial_grid)
            state.grid[row][column] = original
            assert result.evaluation == expected, (
                f"trial preview mismatch at ({row},{column})->{target_orientation}: "
                f"incremental={result.evaluation} full={expected}"
            )

        best_orientation = max(results, key=lambda o: results[o].evaluation)
        evaluator.commit_flips(results[best_orientation])
        assert state.grid[row][column] == best_orientation
        assert_matches_ground_truth(state, exits, exit_ids, initial_grid, evaluator)
        assert_internally_consistent(evaluator)
