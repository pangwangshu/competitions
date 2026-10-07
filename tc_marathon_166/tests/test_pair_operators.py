"""Correctness harness for Tier 2's pair-level operators (reroute_pair,
drop_pair, kick_pairs, IncrementalEvaluator.pair_tiles).

Same discipline as test_incremental.py: every commit is cross-checked
against a fresh evaluate_grid call plus the internal tile_owners/path_of
consistency invariant, not just at the end of a run.
"""

import random
from time import perf_counter

import pytest

from hextiles.geometry import S
from hextiles.incremental import IncrementalEvaluator, trace_path
from hextiles.solver import (
    candidate_orders,
    construct_grid,
    drop_pair,
    evaluate_grid,
    hex_distance,
    improve_grid,
    kick_pairs,
    reroute_pair,
    safe_move_budget,
    try_connect,
    widen_frontier,
    widen_random,
    widen_spatial,
)

from test_incremental import (
    assert_internally_consistent,
    assert_matches_ground_truth,
    build_target_matches,
    make_random_state,
)


def _constructed_state(n, seed, bonus_count=3):
    """A realistic post-construction grid: some pairs matched, some not --
    exactly the mixed state reroute_pair/drop_pair are meant to operate on,
    unlike a purely random grid where target pairing has no relationship
    to physical connectivity."""
    state, exits, exit_ids = make_random_state(n, seed, bonus_count)
    initial_grid = [row[:] for row in state.grid]
    orders = candidate_orders(state, exits)
    state.grid = [row[:] for row in initial_grid]
    construct_grid(state, exits, exit_ids, orders[0])
    target_matches = build_target_matches(state)
    evaluator = IncrementalEvaluator(state, exits, exit_ids, initial_grid, target_matches)
    return state, exits, exit_ids, initial_grid, evaluator


CONSTRUCTED_CASES = [(6, 201), (10, 202), (16, 203), (20, 204)]


@pytest.mark.parametrize("n,seed", CONSTRUCTED_CASES)
def test_pair_tiles_matches_ground_truth(n, seed):
    state, exits, exit_ids, initial_grid, evaluator = _constructed_state(n, seed)
    rng = random.Random(seed * 17 + 5)
    sample_pairs = rng.sample(state.pairs, min(15, len(state.pairs)))

    for first_exit, second_exit in sample_pairs:
        _, _, _, first_tiles = trace_path(state, exits, exit_ids, first_exit)
        _, _, _, second_tiles = trace_path(state, exits, exit_ids, second_exit)
        expected = first_tiles | second_tiles
        actual = evaluator.pair_tiles(first_exit, second_exit)
        assert actual == expected, f"N={n} seed={seed} pair=({first_exit},{second_exit})"

        record = evaluator.path_of[first_exit]
        if record.matched:
            assert record.partner == second_exit
            assert actual == record.tiles == evaluator.path_of[second_exit].tiles


@pytest.mark.parametrize("n,seed", CONSTRUCTED_CASES)
def test_reroute_pair_is_noop_for_matched_pairs(n, seed):
    """Regression-guards the central Tier 2 design proof: the zero-cost
    Dijkstra path through a claimed backdrop of everything-except-this-
    pair's-own-tiles is unique and equals the pair's existing trace, so
    reroute_pair can never find a different (or any) flip for an already-
    matched pair."""
    state, exits, exit_ids, initial_grid, evaluator = _constructed_state(n, seed)
    safe_budget = safe_move_budget(n)
    matched_pairs = [pair for pair in state.pairs if evaluator.path_of[pair[0]].matched]
    assert matched_pairs, f"N={n} seed={seed}: expected at least one matched pair"

    before_grid = [row[:] for row in state.grid]
    for first_exit, second_exit in matched_pairs:
        result = reroute_pair(state, exits, exit_ids, evaluator, first_exit, second_exit, safe_budget)
        assert result is None, f"N={n} seed={seed} pair=({first_exit},{second_exit}) expected no-op"
        assert state.grid == before_grid, "reroute_pair must not leave state.grid mutated"


@pytest.mark.parametrize("n,seed", [(6, 401), (10, 402), (16, 403)])
def test_reroute_and_drop_pair_ground_truth(n, seed):
    """Random-walk mixing pair operators with occasional small tile kicks
    (to keep diversifying the state, mirroring improve_grid's own
    dynamics), asserting ground truth + internal consistency after every
    single commit."""
    state, exits, exit_ids, initial_grid, evaluator = _constructed_state(n, seed)
    assert_matches_ground_truth(state, exits, exit_ids, initial_grid, evaluator)
    assert_internally_consistent(evaluator)

    safe_budget = safe_move_budget(n)
    rng = random.Random(seed * 13 + 3)
    tile_locations = [
        (row, column)
        for row in range(state.w)
        for column in range(state.w)
        if state.grid[row][column] >= 0
    ]
    pair_list = list(state.pairs)

    for _ in range(80):
        if rng.random() < 0.2:
            flip_tiles = rng.sample(tile_locations, min(3, len(tile_locations)))
            flips = {tile: rng.randrange(S) for tile in flip_tiles}
            evaluator.commit_flips(evaluator.preview_flips(flips))
            assert_matches_ground_truth(state, exits, exit_ids, initial_grid, evaluator)
            assert_internally_consistent(evaluator)
            continue

        first_exit, second_exit = rng.choice(pair_list)
        record = evaluator.path_of[first_exit]
        if record.matched:
            result = drop_pair(state, evaluator, first_exit, second_exit)
        else:
            result = reroute_pair(state, exits, exit_ids, evaluator, first_exit, second_exit, safe_budget)

        if result is None:
            continue

        evaluator.commit_flips(result)
        assert_matches_ground_truth(state, exits, exit_ids, initial_grid, evaluator)
        assert_internally_consistent(evaluator)


@pytest.mark.parametrize("n,seed", [(8, 501), (13, 502)])
def test_kick_pairs_ground_truth(n, seed):
    state, exits, exit_ids, initial_grid, evaluator = _constructed_state(n, seed)
    rng = random.Random(seed * 29 + 11)

    for _ in range(20):
        flips = kick_pairs(evaluator, state.pairs, rng)
        assert flips, "kick_pairs should always affect at least one tile"
        evaluator.commit_flips(evaluator.preview_flips(flips))
        assert_matches_ground_truth(state, exits, exit_ids, initial_grid, evaluator)
        assert_internally_consistent(evaluator)


@pytest.mark.parametrize("n,seed", [(6, 601), (10, 602), (16, 603)])
def test_improve_grid_never_regresses_input(n, seed):
    state, exits, exit_ids, initial_grid, evaluator = _constructed_state(n, seed)
    base_evaluation = evaluate_grid(state, exits, exit_ids, initial_grid)

    deadline = perf_counter() + 0.5
    result = improve_grid(state, exits, exit_ids, initial_grid, base_evaluation, deadline)
    assert result >= base_evaluation, f"N={n} seed={seed}: {result} < {base_evaluation}"


# ---------------------------------------------------------------------------
# Tier 2.5: escalating reroute_pair (widen_strategy / max_widen_rounds)
# ---------------------------------------------------------------------------

WIDEN_STRATEGIES = ["random", "spatial", "frontier"]


@pytest.mark.parametrize("widen_strategy", WIDEN_STRATEGIES)
@pytest.mark.parametrize("n,seed", CONSTRUCTED_CASES)
def test_reroute_pair_is_noop_for_matched_pairs_with_widening(n, seed, widen_strategy):
    """The matched-pair no-op guard fires before any freed-tile-set
    computation (see reroute_pair's docstring), so it must hold regardless
    of widen_strategy/rounds -- widening code must never even run for an
    already-matched pair."""
    state, exits, exit_ids, initial_grid, evaluator = _constructed_state(n, seed)
    safe_budget = safe_move_budget(n)
    matched_pairs = [pair for pair in state.pairs if evaluator.path_of[pair[0]].matched]
    assert matched_pairs, f"N={n} seed={seed}: expected at least one matched pair"

    rng = random.Random(seed * 53 + 7)
    before_grid = [row[:] for row in state.grid]
    for first_exit, second_exit in matched_pairs:
        result = reroute_pair(
            state, exits, exit_ids, evaluator, first_exit, second_exit, safe_budget,
            random_source=rng, widen_strategy=widen_strategy,
            max_widen_rounds=3, widen_pairs_per_round=5,
        )
        assert result is None, (
            f"N={n} seed={seed} strategy={widen_strategy} pair=({first_exit},{second_exit}) "
            "expected no-op even with widening enabled"
        )
        assert state.grid == before_grid, "reroute_pair must not leave state.grid mutated"


@pytest.mark.parametrize("widen_strategy", WIDEN_STRATEGIES)
@pytest.mark.parametrize("n,seed", CONSTRUCTED_CASES)
def test_widen_finds_at_least_as_many_routes_as_narrow(n, seed, widen_strategy):
    """Round 0 of an escalating call is bit-identical to a narrow-only
    call (same freed set, same try_connect inputs, no side effects from
    either call since neither commits) -- so whenever the narrow call
    finds a route, the widened call must find the identical one at round
    0, before ever escalating. This makes widening a strict superset of
    narrow reroute_pair's hit rate, checked here exactly rather than just
    trending in the right direction."""
    state, exits, exit_ids, initial_grid, evaluator = _constructed_state(n, seed)
    safe_budget = safe_move_budget(n)
    rng = random.Random(seed * 41 + 19)
    unmatched_pairs = [pair for pair in state.pairs if not evaluator.path_of[pair[0]].matched]
    assert unmatched_pairs, f"N={n} seed={seed}: expected at least one unmatched pair"

    narrow_hits = 0
    widened_hits = 0
    for first_exit, second_exit in unmatched_pairs:
        narrow_result = reroute_pair(
            state, exits, exit_ids, evaluator, first_exit, second_exit, safe_budget
        )
        widened_result = reroute_pair(
            state, exits, exit_ids, evaluator, first_exit, second_exit, safe_budget,
            random_source=rng, widen_strategy=widen_strategy,
            max_widen_rounds=2, widen_pairs_per_round=5,
        )
        if narrow_result is not None:
            narrow_hits += 1
            assert widened_result is not None, (
                f"N={n} seed={seed} strategy={widen_strategy} pair=({first_exit},{second_exit}): "
                "widened reroute_pair failed where narrow succeeded"
            )
            assert widened_result.evaluation == narrow_result.evaluation, (
                "widened round 0 must reproduce narrow's exact result"
            )
        if widened_result is not None:
            widened_hits += 1

    assert widened_hits >= narrow_hits, (
        f"N={n} seed={seed} strategy={widen_strategy}: widened={widened_hits} < narrow={narrow_hits}"
    )


@pytest.mark.parametrize("widen_strategy", WIDEN_STRATEGIES)
@pytest.mark.parametrize("n,seed", [(10, 701), (16, 702), (20, 703)])
def test_widen_reroute_ground_truth(n, seed, widen_strategy):
    """Random-walk mixing widened reroute_pair with drop_pair and tile
    kicks, asserting ground truth + internal consistency after every
    commit -- same discipline as test_reroute_and_drop_pair_ground_truth,
    extended to exercise escalation (max_widen_rounds=2 makes most
    unmatched pairs, which have a 7-13% narrow hit rate per the Tier 2.5
    investigation, actually reach the widening code path)."""
    state, exits, exit_ids, initial_grid, evaluator = _constructed_state(n, seed)
    assert_matches_ground_truth(state, exits, exit_ids, initial_grid, evaluator)
    assert_internally_consistent(evaluator)

    safe_budget = safe_move_budget(n)
    rng = random.Random(seed * 13 + 3)
    tile_locations = [
        (row, column)
        for row in range(state.w)
        for column in range(state.w)
        if state.grid[row][column] >= 0
    ]
    pair_list = list(state.pairs)

    for _ in range(80):
        if rng.random() < 0.2:
            flip_tiles = rng.sample(tile_locations, min(3, len(tile_locations)))
            flips = {tile: rng.randrange(S) for tile in flip_tiles}
            evaluator.commit_flips(evaluator.preview_flips(flips))
            assert_matches_ground_truth(state, exits, exit_ids, initial_grid, evaluator)
            assert_internally_consistent(evaluator)
            continue

        first_exit, second_exit = rng.choice(pair_list)
        record = evaluator.path_of[first_exit]
        if record.matched:
            result = drop_pair(state, evaluator, first_exit, second_exit)
        else:
            result = reroute_pair(
                state, exits, exit_ids, evaluator, first_exit, second_exit, safe_budget,
                random_source=rng, widen_strategy=widen_strategy,
                max_widen_rounds=2, widen_pairs_per_round=5,
            )

        if result is None:
            continue

        evaluator.commit_flips(result)
        assert_matches_ground_truth(state, exits, exit_ids, initial_grid, evaluator)
        assert_internally_consistent(evaluator)


@pytest.mark.parametrize("widen_strategy", WIDEN_STRATEGIES)
@pytest.mark.parametrize("n,seed", [(6, 601), (10, 602), (16, 603)])
def test_improve_grid_never_regresses_input_with_widening(n, seed, widen_strategy):
    state, exits, exit_ids, initial_grid, evaluator = _constructed_state(n, seed)
    base_evaluation = evaluate_grid(state, exits, exit_ids, initial_grid)

    deadline = perf_counter() + 0.5
    result = improve_grid(
        state, exits, exit_ids, initial_grid, base_evaluation, deadline,
        widen_strategy=widen_strategy, max_widen_rounds=2, widen_pairs_per_round=5,
    )
    assert result >= base_evaluation, (
        f"N={n} seed={seed} strategy={widen_strategy}: {result} < {base_evaluation}"
    )


def test_frontier_tiles_are_claimed_and_mostly_owned():
    """Direct check of _connect_once's frontier capture: on a genuine
    connectivity failure, every captured tile must be currently claimed
    (foreign territory outside the pair's own freed tiles). Most captured
    tiles should also appear in tile_owners (a real pair's current trace),
    which is what widen_frontier's ranking depends on -- but not all of
    them have to: a tile whose all 3 chords lie on a closed interior loop
    that never touches the border is legitimately claimed and reachable
    with zero border-reaching owners (confirmed empirically: 2 of 1141
    tiles at N=20 seed=204). widen_frontier's tile_owners.get(tile, ())
    already handles that tile-with-no-owner case gracefully (contributes
    nothing, doesn't crash) -- this test just documents that it's an
    expected, not-vanishingly-rare occurrence, not a bug."""
    state, exits, exit_ids, initial_grid, evaluator = _constructed_state(20, 204)
    safe_budget = safe_move_budget(20)
    unmatched_pairs = [pair for pair in state.pairs if not evaluator.path_of[pair[0]].matched]
    assert unmatched_pairs, "expected at least one unmatched pair"

    found_a_failure = False
    total_frontier_tiles = 0
    owned_frontier_tiles = 0
    for first_exit, second_exit in unmatched_pairs:
        freed = set(evaluator.pair_tiles(first_exit, second_exit))
        claimed = [[True] * state.w for _ in range(state.w)]
        for row, column in freed:
            claimed[row][column] = False
        snapshot = {tile: state.grid[tile[0]][tile[1]] for tile in freed}

        frontier_tiles = set()
        try_connect(
            state, exits, exit_ids, claimed, [],
            first_exit, second_exit, safe_budget,
            frontier_tiles=frontier_tiles,
        )
        for tile, orientation in snapshot.items():
            state.grid[tile[0]][tile[1]] = orientation

        if not frontier_tiles:
            continue
        found_a_failure = True
        for row, column in frontier_tiles:
            assert claimed[row][column], f"frontier tile ({row},{column}) was not claimed"
            total_frontier_tiles += 1
            if (row, column) in evaluator.tile_owners:
                owned_frontier_tiles += 1

    assert found_a_failure, "expected at least one narrow reroute failure to inspect"
    assert owned_frontier_tiles / total_frontier_tiles > 0.9, (
        f"expected the vast majority of frontier tiles to be owned: "
        f"{owned_frontier_tiles}/{total_frontier_tiles}"
    )


def test_widen_random_excludes_stuck_pair_and_respects_count():
    state, exits, exit_ids, initial_grid, evaluator = _constructed_state(16, 203)
    first_exit, second_exit = state.pairs[0]
    rng = random.Random(999)
    chosen = widen_random(state, {first_exit, second_exit}, 5, rng)
    assert len(chosen) <= 5
    for a, b in chosen:
        assert first_exit not in (a, b) and second_exit not in (a, b)


def test_widen_spatial_ranks_by_proximity_to_corridor():
    state, exits, exit_ids, initial_grid, evaluator = _constructed_state(16, 203)
    first_exit, second_exit = state.pairs[0]
    exclude = {first_exit, second_exit}
    chosen = widen_spatial(state, exits, first_exit, second_exit, exclude, len(state.pairs))

    source_position = exits[first_exit]
    target_position = exits[second_exit]
    direct_distance = hex_distance(source_position, target_position)

    def detour(pair):
        return min(
            hex_distance(source_position, exits[exit_id])
            + hex_distance(exits[exit_id], target_position)
            - direct_distance
            for exit_id in pair
        )

    detours = [detour(pair) for pair in chosen]
    assert detours == sorted(detours)


def test_widen_frontier_ranks_by_contention_count():
    """widen_frontier must rank candidate pairs by how many frontier tiles
    they own, highest first -- verified against an independently computed
    contention count rather than just checking membership."""
    state, exits, exit_ids, initial_grid, evaluator = _constructed_state(20, 204)
    first_exit, second_exit = state.pairs[0]
    exclude = {first_exit, second_exit}
    frontier_tiles = set(evaluator.tile_owners.keys())

    expected_counts = {}
    for tile in frontier_tiles:
        for exit_id in evaluator.tile_owners.get(tile, ()):
            if exit_id in exclude:
                continue
            expected_counts[exit_id] = expected_counts.get(exit_id, 0) + 1
    assert expected_counts, "expected at least one contention candidate"

    chosen = widen_frontier(evaluator, frontier_tiles, exclude, len(state.pairs))
    chosen_exits = [exit_id for pair in chosen for exit_id in pair if exit_id in expected_counts]
    counts_in_order = [expected_counts[exit_id] for exit_id in chosen_exits]
    assert counts_in_order == sorted(counts_in_order, reverse=True), (
        "widen_frontier's output must be ranked by descending contention count"
    )
