import os
from collections import Counter
from heapq import heappop, heappush
from random import Random
from time import perf_counter

from hextiles.geometry import DC, DR, S, in_grid, minimum_rotation, partner_at, rotation_distance
from hextiles.incremental import IncrementalEvaluator


def build_exits(state):
    exit_locations = []
    exit_ids = {}
    row = 0
    column = state.n - 1
    direction = 1

    while len(exit_locations) < S * state.w:
        outward_direction = (direction + 1) % S
        for _ in range(S):
            next_row = row + DR[outward_direction]
            next_column = column + DC[outward_direction]
            if not in_grid(next_row, next_column, state.n, state.w):
                exit_id = len(exit_locations)
                exit_locations.append((row, column, outward_direction))
                exit_ids[(row, column, outward_direction)] = exit_id
            outward_direction = (outward_direction + 1) % S

        next_row = row + DR[direction]
        next_column = column + DC[direction]
        if not in_grid(next_row, next_column, state.n, state.w):
            direction = (direction + 1) % S
            next_row = row + DR[direction]
            next_column = column + DC[direction]
        row, column = next_row, next_column

    return exit_locations, exit_ids


def hex_distance(first_location, second_location):
    row_delta = first_location[0] - second_location[0]
    column_delta = first_location[1] - second_location[1]
    return (abs(row_delta) + abs(column_delta) + abs(row_delta + column_delta)) // 2


def move_to_neighbor(state, exit_ids, row, column, exit_edge, regular_nodes):
    next_row = row + DR[exit_edge]
    next_column = column + DC[exit_edge]
    if in_grid(next_row, next_column, state.n, state.w):
        entry_edge = (exit_edge + 3) % S
        return ((next_row * state.w + next_column) * S) + entry_edge
    return regular_nodes + exit_ids[(row, column, exit_edge)]


def rotate_tile(state, moves, row, column, target_orientation):
    current_orientation = state.grid[row][column]
    clockwise = (target_orientation - current_orientation) % S
    anti_clockwise = (current_orientation - target_orientation) % S
    direction = 1 if clockwise <= anti_clockwise else -1
    for _ in range(min(clockwise, anti_clockwise)):
        moves.append((row, column, direction))
    state.grid[row][column] = target_orientation


MAX_CONFLICT_RETRIES = 4


def try_connect(
    state,
    exits,
    exit_ids,
    claimed,
    moves,
    source_exit,
    target_exit,
    safe_budget,
    required_bonuses=(),
    frontier_tiles=None,
):
    blocked_tiles = set()
    for _ in range(MAX_CONFLICT_RETRIES):
        conflict_tile = _connect_once(
            state,
            exits,
            exit_ids,
            claimed,
            moves,
            source_exit,
            target_exit,
            safe_budget,
            required_bonuses,
            blocked_tiles,
            frontier_tiles,
        )
        if conflict_tile is None:
            return
        # The route found required visiting this tile a second time with an
        # edge requirement its first visit's orientation can't also satisfy.
        # Exclude it and re-route rather than giving up on the pair outright
        # -- a valid (if pricier) route around it usually still exists.
        blocked_tiles.add(conflict_tile)


def _connect_once(
    state,
    exits,
    exit_ids,
    claimed,
    moves,
    source_exit,
    target_exit,
    safe_budget,
    required_bonuses,
    blocked_tiles,
    frontier_tiles=None,
):
    """Single Dijkstra-and-commit attempt.

    Returns None if the pair was connected (or is unreachable outright), or
    the (row, column) of a same-path revisit conflict for the caller to
    retry around.

    If frontier_tiles is given and the goal turns out unreachable, it gets
    populated (accumulated across try_connect's own retries) with every
    claimed tile the search actually reached before running out of reachable
    states -- the claimed tiles it was forced to pass through en route to
    nowhere useful. A claimed tile is never a dead end (it always has one
    fixed zero-cost exit), so this reflects the full reachable component,
    not a truncated frontier. Callers use this to identify which other
    pairs' territory is actually in the way; no-op (and free) when None.
    """
    start_row, start_column, start_edge = exits[source_exit]
    regular_nodes = state.w * state.w * S
    exit_count = len(exits)
    start_node = ((start_row * state.w + start_column) * S) + start_edge
    goal_node = regular_nodes + target_exit
    node_count = regular_nodes + exit_count
    bonus_bits = {
        bonus: 1 << index for index, bonus in enumerate(required_bonuses)
    }
    required_bonus_mask = (1 << len(required_bonuses)) - 1
    state_layers = required_bonus_mask + 1

    def state_id(node, visited_bonus_mask):
        return node + visited_bonus_mask * node_count

    start_bonus_mask = bonus_bits.get((start_row, start_column), 0)
    start_state = state_id(start_node, start_bonus_mask)
    goal_state = state_id(goal_node, required_bonus_mask)
    distances = [float("inf")] * (node_count * state_layers)
    previous_nodes = [-1] * (node_count * state_layers)
    previous_edges = [-1] * (node_count * state_layers)
    distances[start_state] = 0
    queue = [(0, start_state)]

    while queue:
        distance, current_state = heappop(queue)
        if distance != distances[current_state]:
            continue
        if current_state == goal_state:
            break
        node = current_state % node_count
        if node >= regular_nodes:
            continue

        entry_edge = node % S
        tile_index = node // S
        row, column = divmod(tile_index, state.w)
        visited_bonus_mask = current_state // node_count
        visited_bonus_mask |= bonus_bits.get((row, column), 0)
        if claimed[row][column]:
            candidate_edges = [(partner_at(entry_edge, state.grid[row][column]), 0)]
        elif (row, column) in blocked_tiles:
            candidate_edges = []
        else:
            opposite_edge = (entry_edge + 3) % S
            candidate_edges = [
                (exit_edge, minimum_rotation(entry_edge, exit_edge, state.grid[row][column]))
                for exit_edge in range(S)
                if exit_edge not in (entry_edge, opposite_edge)
            ]

        for exit_edge, rotation_cost in candidate_edges:
            next_node = move_to_neighbor(
                state, exit_ids, row, column, exit_edge, regular_nodes
            )
            next_distance = distance + rotation_cost
            next_state = state_id(next_node, visited_bonus_mask)
            if next_distance < distances[next_state]:
                distances[next_state] = next_distance
                previous_nodes[next_state] = current_state
                previous_edges[next_state] = exit_edge
                heappush(queue, (next_distance, next_state))

    if distances[goal_state] == float("inf"):
        if frontier_tiles is not None:
            for state_index, distance in enumerate(distances):
                if distance == float("inf"):
                    continue
                node = state_index % node_count
                if node >= regular_nodes:
                    continue
                row, column = divmod(node // S, state.w)
                if claimed[row][column]:
                    frontier_tiles.add((row, column))
        return None
    if len(moves) + distances[goal_state] > safe_budget:
        return None

    path = []
    current_state = goal_state
    while current_state != start_state:
        previous_state = previous_nodes[current_state]
        path.append((previous_state % node_count, previous_edges[current_state]))
        current_state = previous_state
    path.reverse()

    tile_orientations = {}
    for node, exit_edge in path:
        entry_edge = node % S
        tile_index = node // S
        row, column = divmod(tile_index, state.w)
        if claimed[row][column]:
            continue

        if (row, column) in tile_orientations:
            if partner_at(entry_edge, tile_orientations[(row, column)]) != exit_edge:
                return (row, column)
            continue

        current_orientation = state.grid[row][column]
        target_orientation = min(
            (
                orientation
                for orientation in range(S)
                if partner_at(entry_edge, orientation) == exit_edge
            ),
            key=lambda orientation: rotation_distance(current_orientation, orientation),
        )
        tile_orientations[(row, column)] = target_orientation

    for (row, column), target_orientation in tile_orientations.items():
        rotate_tile(state, moves, row, column, target_orientation)
        claimed[row][column] = True
    return None


def evaluate_grid(state, exits, exit_ids, initial_grid):
    target_matches = {}
    for first_exit, second_exit in state.pairs:
        target_matches[first_exit] = second_exit
        target_matches[second_exit] = first_exit

    matched_paths = 0
    total_path_score = 0
    visited_exits = set()
    for start_exit, (row, column, entry_edge) in enumerate(exits):
        if start_exit in visited_exits:
            continue

        path_length = 0
        visited_bonus_tiles = set()
        visited_entries = set()
        while True:
            entry = (row, column, entry_edge)
            if entry in visited_entries:
                break
            visited_entries.add(entry)
            path_length += 1
            if (row, column) in state.bonus:
                visited_bonus_tiles.add((row, column))

            exit_edge = partner_at(entry_edge, state.grid[row][column])
            next_row = row + DR[exit_edge]
            next_column = column + DC[exit_edge]
            if not in_grid(next_row, next_column, state.n, state.w):
                end_exit = exit_ids[(row, column, exit_edge)]
                visited_exits.add(start_exit)
                visited_exits.add(end_exit)
                if target_matches[start_exit] == end_exit:
                    matched_paths += 1
                    total_path_score += path_length * (len(visited_bonus_tiles) + 1)
                break

            row = next_row
            column = next_column
            entry_edge = (exit_edge + 3) % S

    move_count = sum(
        rotation_distance(initial_grid[row][column], state.grid[row][column])
        for row in range(state.w)
        for column in range(state.w)
        if initial_grid[row][column] >= 0
    )
    score = matched_paths * (total_path_score - move_count * state.m)
    return max(score, 0), matched_paths, total_path_score, -move_count


def safe_move_budget(n):
    """Move budget for try_connect callers, a small margin below the hard
    24*n*n submission cap the tester enforces (exceeding it is a fatal
    error, not just a lower score)."""
    return 24 * n * n - max(6, n)


def construct_grid(state, exits, exit_ids, ordered_pairs, bonus_plan=None):
    claimed = [[False] * state.w for _ in range(state.w)]
    moves = []
    safe_budget = safe_move_budget(state.n)

    for source_exit, target_exit in ordered_pairs:
        required_bonuses = ()
        if bonus_plan is not None:
            required_bonuses = bonus_plan.get((source_exit, target_exit), ())
        try_connect(
            state,
            exits,
            exit_ids,
            claimed,
            moves,
            source_exit,
            target_exit,
            safe_budget,
            required_bonuses,
        )

    return [row[:] for row in state.grid]


def candidate_orders(state, exits):
    nearest_first = sorted(
        state.pairs,
        key=lambda pair: hex_distance(exits[pair[0]], exits[pair[1]]),
    )
    farthest_first = list(reversed(nearest_first))

    def bonus_detour(pair):
        start_exit, end_exit = pair
        return min(
            hex_distance(exits[start_exit], bonus)
            + hex_distance(bonus, exits[end_exit])
            for bonus in state.bonus
        )

    bonus_first = sorted(
        state.pairs,
        key=lambda pair: (bonus_detour(pair), -hex_distance(exits[pair[0]], exits[pair[1]])),
    )
    random_seed = sum(
        (row * state.w + column + 1) * (orientation + 1)
        for row, grid_row in enumerate(state.grid)
        for column, orientation in enumerate(grid_row)
        if orientation >= 0
    )
    orders = [nearest_first, farthest_first, bonus_first]
    for salt in range(3):
        shuffled_pairs = list(state.pairs)
        Random(random_seed + salt).shuffle(shuffled_pairs)
        orders.append(shuffled_pairs)

    unique_orders = []
    seen_orders = set()
    for ordered_pairs in orders:
        order_key = tuple(ordered_pairs)
        if order_key not in seen_orders:
            seen_orders.add(order_key)
            unique_orders.append(ordered_pairs)
    return unique_orders


def bonus_plan(state, exits, pair_count):
    candidates = []
    for bonus in state.bonus:
        for pair in state.pairs:
            start_exit, end_exit = pair
            direct_distance = hex_distance(exits[start_exit], exits[end_exit])
            through_bonus_distance = (
                hex_distance(exits[start_exit], bonus)
                + hex_distance(bonus, exits[end_exit])
            )
            detour_distance = through_bonus_distance - direct_distance
            value = 3 * direct_distance - 2 * detour_distance
            candidates.append((value, pair, bonus))

    plan = {}
    used_bonuses = set()
    for _, pair, bonus in sorted(candidates, reverse=True):
        if pair in plan or bonus in used_bonuses:
            continue
        plan[pair] = (bonus,)
        used_bonuses.add(bonus)
        if len(plan) == pair_count:
            break
    return plan


def prioritize_bonus_pairs(ordered_pairs, plan):
    planned_pairs = [pair for pair in plan if pair in ordered_pairs]
    planned_set = set(planned_pairs)
    return planned_pairs + [pair for pair in ordered_pairs if pair not in planned_set]


# Internal deadline, 0.5s inside TopCoder's 10s hard limit. Combined with
# probe selection ON it was the only locally-positive configuration measured
# (+2.28% on seeds 1-40, 22W/16L/2T, 0 failures); either change alone was
# net negative. It gives back part of the earlier 8.5s safety margin (Tier
# 2.5 take-1: 9.0s scored 50.85 on real hardware, trimming to 8.5s jumped to
# 55.01). What makes that trade plausible is that reroute_pair's widening
# rounds now stop at the deadline, bounding a late call's overrun to about
# one try_connect. Unset, the env override falls back to 9.5s.
SOLVE_DEADLINE_SECONDS = float(os.environ.get("HEXTILES_SOLVE_DEADLINE", "9.5"))
MIN_LOCAL_SEARCH_SECONDS = 5.0
MAX_LOCAL_SEARCH_SECONDS = 7.5
CONSTRUCTION_BUFFER_SECONDS = 0.05


def target_local_search_seconds(n):
    scale = (n - 3) / 17
    return MIN_LOCAL_SEARCH_SECONDS + (
        MAX_LOCAL_SEARCH_SECONDS - MIN_LOCAL_SEARCH_SECONDS
    ) * scale


def _construction_attempt_key(ordered_pairs, plan):
    plan_key = None
    if plan is not None:
        plan_key = tuple(sorted(plan.items()))
    return tuple(ordered_pairs), plan_key


def construction_attempts(state, exits):
    nearest_first = sorted(
        state.pairs,
        key=lambda pair: hex_distance(exits[pair[0]], exits[pair[1]]),
    )
    attempts = []
    seen = set()

    def append_attempt(ordered_pairs, plan):
        attempt_key = _construction_attempt_key(ordered_pairs, plan)
        if attempt_key in seen:
            return
        seen.add(attempt_key)
        attempts.append((ordered_pairs, plan))

    def append_bonus_attempt(pair_count):
        plan = bonus_plan(state, exits, pair_count)
        if plan:
            append_attempt(prioritize_bonus_pairs(nearest_first, plan), plan)

    if state.n <= 12:
        for ordered_pairs in candidate_orders(state, exits):
            append_attempt(ordered_pairs, None)
        bonus_pair_counts = sorted({1, min(3, state.b), state.b})
    else:
        bonus_pair_counts = (1,)

    for pair_count in bonus_pair_counts:
        append_bonus_attempt(pair_count)

    if state.n > 12:
        for ordered_pairs in candidate_orders(state, exits):
            append_attempt(ordered_pairs, None)
        for pair_count in sorted({min(3, state.b), state.b}):
            append_bonus_attempt(pair_count)

    return attempts


def can_start_construction_attempt(now, solve_deadline, search_reserve, attempt_durations):
    search_start_target = solve_deadline - search_reserve
    if now >= search_start_target:
        return False
    if not attempt_durations:
        return True

    average_duration = sum(attempt_durations) / len(attempt_durations)
    estimated_next_duration = max(attempt_durations[-1], average_duration)
    return (
        now + estimated_next_duration + CONSTRUCTION_BUFFER_SECONDS
        <= search_start_target
    )


def moves_to_reach_grid(initial_grid, target_grid):
    moves = []
    width = len(initial_grid)
    for row in range(width):
        for column in range(width):
            original_orientation = initial_grid[row][column]
            target_orientation = target_grid[row][column]
            if original_orientation < 0 or original_orientation == target_orientation:
                continue
            clockwise = (target_orientation - original_orientation) % S
            anti_clockwise = (original_orientation - target_orientation) % S
            direction = 1 if clockwise <= anti_clockwise else -1
            for _ in range(min(clockwise, anti_clockwise)):
                moves.append((row, column, direction))
    return moves


class _MoveBudgetCounter:
    """Duck-types the len()/.append() surface try_connect's `moves` param
    needs for its safe_budget check, without allocating an O(move_count)
    list just to seed a starting count."""

    __slots__ = ("count",)

    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def append(self, _item):
        self.count += 1


# Widening defaults for solve()'s real call. Take-2 note (full history in
# report/writeup.md, section 5): random/rounds=2 with
# pairs_per_round=10 was selected by real-tester sweeps (see below) -- the
# original take-1 tuning never swept pairs_per_round at all and capped
# rounds at 2 based on synthetic data. The deadline guard in reroute_pair
# (rounds >= 1 skipped once the local-search deadline passes) bounds a
# late-starting call's worst-case overrun to ~1 try_connect, addressing the
# tail-latency failure mode that sank take-1's expensive configs on
# TopCoder hardware twice. The env-var overrides exist only so
# scripts/run_seeds.sh can sweep configs without editing this file between
# runs; unset, they fall back to the shipped defaults below.
def _env_widen_strategy():
    value = os.environ.get("HEXTILES_WIDEN_STRATEGY", "random")
    return None if value.lower() in ("", "none", "off") else value


DEFAULT_WIDEN_STRATEGY = _env_widen_strategy()
DEFAULT_MAX_WIDEN_ROUNDS = int(os.environ.get("HEXTILES_WIDEN_ROUNDS", "2"))
DEFAULT_WIDEN_PAIRS_PER_ROUND = int(os.environ.get("HEXTILES_WIDEN_PAIRS", "10"))


def _other_pairs(pairs, exclude_exits):
    return [
        (first_exit, second_exit)
        for first_exit, second_exit in pairs
        if first_exit not in exclude_exits and second_exit not in exclude_exits
    ]


def widen_random(state, exclude_exits, count, random_source):
    """Widen strategy: `count` other pairs chosen uniformly at random.
    Validated directly (Tier 2.5 investigation): freeing 5 random other
    pairs' tiles unblocked 33-78% of stuck pairs with no targeting logic
    at all, since match rate is limited by construction's greedy
    claim-and-lock, not by infeasibility -- most nearby contention is as
    good as any other."""
    candidates = _other_pairs(state.pairs, exclude_exits)
    if not candidates:
        return []
    return random_source.sample(candidates, min(count, len(candidates)))


def widen_spatial(state, exits, first_exit, second_exit, exclude_exits, count):
    """Widen strategy: the `count` other pairs whose own exits sit closest
    to the direct corridor between first_exit/second_exit, using the same
    detour-distance idiom bonus_plan already uses for waypoint ranking
    (through-distance minus direct-distance -- smaller means closer to the
    line)."""
    source_position = exits[first_exit]
    target_position = exits[second_exit]
    direct_distance = hex_distance(source_position, target_position)
    candidates = _other_pairs(state.pairs, exclude_exits)

    def detour(pair):
        return min(
            hex_distance(source_position, exits[exit_id])
            + hex_distance(exits[exit_id], target_position)
            - direct_distance
            for exit_id in pair
        )

    return sorted(candidates, key=detour)[:count]


def widen_frontier(evaluator, frontier_tiles, exclude_exits, count):
    """Widen strategy: the `count` other pairs owning the most tiles the
    immediately-preceding failed try_connect attempt actually reached --
    a direct read of which pairs' claimed territory the search got
    funneled through, via tile_owners, rather than a geometric proxy for
    it."""
    contention = Counter()
    for tile in frontier_tiles:
        for exit_id in evaluator.tile_owners.get(tile, ()):
            if exit_id not in exclude_exits:
                contention[exit_id] += 1

    chosen = []
    seen_exits = set()
    for exit_id, _ in contention.most_common():
        if exit_id in seen_exits:
            continue
        partner = evaluator.path_of[exit_id].partner
        seen_exits.add(exit_id)
        if partner is not None:
            seen_exits.add(partner)
        chosen.append((exit_id, partner) if partner is not None else (exit_id, exit_id))
        if len(chosen) >= count:
            break
    return chosen


def reroute_pair(
    state,
    exits,
    exit_ids,
    evaluator,
    first_exit,
    second_exit,
    safe_budget,
    random_source=None,
    widen_strategy=None,
    max_widen_rounds=0,
    widen_pairs_per_round=DEFAULT_WIDEN_PAIRS_PER_ROUND,
    deadline=None,
):
    """Try to connect first_exit/second_exit, starting from only the tiles
    their own current traces already touch (round 0 -- identical to the
    original Tier 2 behavior). On failure, if widen_strategy is given,
    escalate up to max_widen_rounds times: union in widen_pairs_per_round
    more other pairs' current tiles (chosen by "random", "spatial", or
    "frontier") and retry try_connect against the widened backdrop.
    Widening is cumulative -- each round's freed set is a superset of the
    previous one's -- and stops at the first round that finds a route.

    Take-2 change vs. the original Tier 2.5 restore: widening rounds (round
    index >= 1) are skipped once `deadline` has passed. The first widening
    round that DOES start is still non-preemptible, but this bounds a
    late-starting call's worst-case overrun to roughly one try_connect
    instead of (1 + max_widen_rounds) chained Dijkstras -- the exact tail
    risk believed to have sunk the more expensive configurations on
    TopCoder's hardware twice. Round 0 is unconditional (it is the cheap
    Tier 2 baseline behavior); away from the deadline all rounds still run,
    so average-case search behavior is unchanged.

    Provably a no-op whenever the pair is ALREADY mutually matched, at
    ANY round and regardless of widen_strategy: at an unclaimed tile,
    minimum_rotation is 0 for exactly one of the 4 candidate exit edges
    (the tile's own current routing) and >0 for the rest, so a total path
    cost of 0 forces every tile on it to be used unchanged -- meaning the
    only zero-cost path is the pair's own existing trace. This is a pure
    property of minimum_rotation, independent of which pair a given freed
    tile belongs to, so the guard below is checked once, unconditionally,
    before any freed-set computation -- widening never runs for an
    already-matched pair.
    """
    record = evaluator.path_of[first_exit]
    if record.matched:
        return None

    freed = set(evaluator.pair_tiles(first_exit, second_exit))
    exclude_exits = {first_exit, second_exit}
    # `claimed` is maintained incrementally across rounds: freed only grows,
    # so each round just flips the newly-added tiles to False instead of
    # rebuilding the whole W x W grid per round.
    claimed = [[True] * state.w for _ in range(state.w)]
    for row, column in freed:
        claimed[row][column] = False

    for round_index in range(max_widen_rounds + 1):
        if round_index > 0 and deadline is not None and perf_counter() >= deadline:
            break
        snapshot = {tile: state.grid[tile[0]][tile[1]] for tile in freed}
        moves_placeholder = _MoveBudgetCounter(evaluator.move_count)
        capturing_frontier = widen_strategy == "frontier" and round_index < max_widen_rounds
        frontier_tiles = set() if capturing_frontier else None
        try_connect(
            state, exits, exit_ids, claimed, moves_placeholder,
            first_exit, second_exit, safe_budget,
            frontier_tiles=frontier_tiles,
        )

        flips = {
            tile: state.grid[tile[0]][tile[1]]
            for tile in freed
            if state.grid[tile[0]][tile[1]] != snapshot[tile]
        }
        for tile, orientation in snapshot.items():
            state.grid[tile[0]][tile[1]] = orientation

        if flips:
            return evaluator.preview_flips(flips)

        if round_index >= max_widen_rounds:
            break

        if widen_strategy == "random" and random_source is not None:
            new_pairs = widen_random(state, exclude_exits, widen_pairs_per_round, random_source)
        elif widen_strategy == "spatial":
            new_pairs = widen_spatial(
                state, exits, first_exit, second_exit, exclude_exits, widen_pairs_per_round
            )
        elif widen_strategy == "frontier":
            new_pairs = widen_frontier(evaluator, frontier_tiles, exclude_exits, widen_pairs_per_round)
        else:
            break

        if not new_pairs:
            break
        added = set()
        for pair_first, pair_second in new_pairs:
            added |= evaluator.pair_tiles(pair_first, pair_second)
            exclude_exits.add(pair_first)
            exclude_exits.add(pair_second)
        added -= freed
        freed |= added
        for row, column in added:
            claimed[row][column] = False

    return None


def drop_pair(state, evaluator, first_exit, second_exit):
    """Test reverting a currently-matched pair's tiles to their pre-solve
    orientation -- directly measures whether keeping this pair connected
    is worth its own move cost. No-op if the pair isn't currently mutually
    matched to second_exit specifically, or if its tiles are already at
    initial orientation.
    """
    record = evaluator.path_of[first_exit]
    if not (record.matched and record.partner == second_exit):
        return None

    initial_grid = evaluator.initial_grid
    flips = {
        tile: initial_grid[tile[0]][tile[1]]
        for tile in record.tiles
        if initial_grid[tile[0]][tile[1]] != state.grid[tile[0]][tile[1]]
    }
    if not flips:
        return None
    return evaluator.preview_flips(flips)


KICK_PATIENCE = 6
MAX_KICK_SIZE = 6
MIN_KICK_PAIRS = 2
MAX_KICK_PAIRS = 5


def kick_grid(tile_locations, random_source):
    kick_size = random_source.randint(1, min(MAX_KICK_SIZE, len(tile_locations)))
    return {
        (row, column): random_source.randrange(S)
        for row, column in random_source.sample(tile_locations, kick_size)
    }


def kick_pairs(evaluator, pairs, random_source):
    """Coarser-grained alternative to kick_grid: reorient every tile
    touched by a handful of random pairs' current traces, rather than a
    handful of scattered single tiles -- meant to escape multi-pair local
    optima that single/few-tile kicks can't. Unconditional, like
    kick_grid -- caller applies via evaluator.apply_flips without gating.

    Unlike kick_grid, this has no explicit cap on tiles touched per call,
    but that's still safe against the hard 24*n*n move cap: move_count is
    always a final-state diff against initial_grid (at most S//2 per
    tile), never a running total of every historical flip, so its
    worst-case total across the whole grid stays well under the cap at
    every N regardless of how many times or how broadly tiles get kicked.
    """
    pair_count = random_source.randint(MIN_KICK_PAIRS, min(MAX_KICK_PAIRS, len(pairs)))
    affected_tiles = set()
    for first_exit, second_exit in random_source.sample(pairs, pair_count):
        affected_tiles |= evaluator.pair_tiles(first_exit, second_exit)
    return {tile: random_source.randrange(S) for tile in affected_tiles}


def improve_grid(
    state,
    exits,
    exit_ids,
    initial_grid,
    evaluation,
    deadline,
    max_passes=None,
    widen_strategy=None,
    max_widen_rounds=0,
    widen_pairs_per_round=DEFAULT_WIDEN_PAIRS_PER_ROUND,
):
    tile_locations = [
        (row, column)
        for row in range(state.w)
        for column in range(state.w)
        if state.grid[row][column] >= 0
    ]
    pair_order = list(state.pairs)
    safe_budget = safe_move_budget(state.n)
    random_source = Random(len(tile_locations) * state.n + len(state.pairs))
    target_matches = {}
    for first_exit, second_exit in state.pairs:
        target_matches[first_exit] = second_exit
        target_matches[second_exit] = first_exit
    evaluator = IncrementalEvaluator(state, exits, exit_ids, initial_grid, target_matches)
    current_evaluation = evaluation
    best_grid = [row[:] for row in state.grid]
    best_evaluation = evaluation
    stale_kicks = 0
    pass_count = 0

    def snapshot_if_best():
        nonlocal best_grid, best_evaluation, stale_kicks
        if current_evaluation > best_evaluation:
            best_grid = [row[:] for row in state.grid]
            best_evaluation = current_evaluation
            stale_kicks = 0

    # Plain hill-climbing gets stuck at the first local optimum. Once stuck,
    # keep using the remaining budget by kicking a few random tiles and
    # climbing again (basin-hopping), always remembering the best grid seen
    # so a bad kick can never make the final result worse.
    while perf_counter() < deadline:
        random_source.shuffle(tile_locations)
        improved = False
        for row, column in tile_locations:
            if perf_counter() >= deadline:
                snapshot_if_best()
                state.grid = best_grid
                return best_evaluation
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

        random_source.shuffle(pair_order)
        for first_exit, second_exit in pair_order:
            if perf_counter() >= deadline:
                snapshot_if_best()
                state.grid = best_grid
                return best_evaluation
            if evaluator.path_of[first_exit].matched:
                result = drop_pair(state, evaluator, first_exit, second_exit)
            else:
                result = reroute_pair(
                    state, exits, exit_ids, evaluator, first_exit, second_exit, safe_budget,
                    random_source=random_source,
                    widen_strategy=widen_strategy,
                    max_widen_rounds=max_widen_rounds,
                    widen_pairs_per_round=widen_pairs_per_round,
                    deadline=deadline,
                )
            if result is not None and result.evaluation > current_evaluation:
                evaluator.commit_flips(result)
                current_evaluation = result.evaluation
                improved = True

        snapshot_if_best()
        pass_count += 1
        if max_passes is not None and pass_count >= max_passes:
            break
        if not improved:
            stale_kicks += 1
            if perf_counter() >= deadline:
                break
            if stale_kicks > KICK_PATIENCE:
                state.grid = [row[:] for row in best_grid]
                # The evaluator's cache tracks whatever was last committed,
                # which is now stale relative to this revert -- rebuild it
                # from the reverted grid rather than let it drift.
                evaluator = IncrementalEvaluator(
                    state, exits, exit_ids, initial_grid, target_matches
                )
                current_evaluation = best_evaluation
                stale_kicks = 0
            if random_source.random() < 0.5:
                kick_flips = kick_pairs(evaluator, state.pairs, random_source)
            else:
                kick_flips = kick_grid(tile_locations, random_source)
            current_evaluation = evaluator.apply_flips(kick_flips)

    snapshot_if_best()
    state.grid = best_grid
    return best_evaluation


# Tier 5 reinvestigation (scripts/investigate_attempt_selection.py): picking
# the construction attempt by its immediate post-construction evaluate_grid
# score is wrong in 7/9 real captured cases, leaving +6.55% of final score on
# the table vs. picking by each attempt's post-improve_grid outcome. This
# adds a cheap probe stage: run a short improve_grid on every constructed
# attempt, keep the best POLISHED grid, then resume polishing that winner to
# the real deadline. probe_seconds scales with the local-search reserve (same
# shape as target_local_search_seconds) so large-N cases, whose attempts are
# more expensive and more consequential, get deeper probes.
PROBE_SELECTION_MIN_SECONDS = 0.15
PROBE_SELECTION_MAX_SECONDS = 0.4
# Skip probing entirely for tiny attempt lists -- with only 1-2 candidates
# the construction-score pick is already close to free, and even a short
# probe per attempt is pure overhead.
PROBE_SELECTION_MIN_ATTEMPTS = 3


def target_probe_seconds(n):
    scale = (n - 3) / 17
    return PROBE_SELECTION_MIN_SECONDS + (
        PROBE_SELECTION_MAX_SECONDS - PROBE_SELECTION_MIN_SECONDS
    ) * scale


def _env_probe_selection():
    # Default ON for the 9.5s submission candidate: probe selection was only
    # net-positive locally in combination with the extra second of budget
    # (see the comment above SOLVE_DEADLINE_SECONDS). Set
    # HEXTILES_PROBE_SELECTION=0 locally to A/B against the old rule.
    value = os.environ.get("HEXTILES_PROBE_SELECTION", "1")
    return value.lower() not in ("", "0", "none", "off")


DEFAULT_PROBE_SELECTION = _env_probe_selection()


def solve(state):
    started_at = perf_counter()
    solve_deadline = started_at + SOLVE_DEADLINE_SECONDS
    initial_grid = [row[:] for row in state.grid]
    exits, exit_ids = build_exits(state)
    best_grid = initial_grid
    best_evaluation = (-1, -1, -1, 0)
    attempts = construction_attempts(state, exits)
    construction_durations = []
    search_reserve = target_local_search_seconds(state.n)
    probe_seconds = target_probe_seconds(state.n)
    probe_enabled = (
        DEFAULT_PROBE_SELECTION and len(attempts) >= PROBE_SELECTION_MIN_ATTEMPTS
    )
    for ordered_pairs, plan in attempts:
        now = perf_counter()
        if not can_start_construction_attempt(
            now, solve_deadline, search_reserve, construction_durations
        ):
            break

        attempt_started_at = now
        state.grid = [row[:] for row in initial_grid]
        candidate_grid = construct_grid(state, exits, exit_ids, ordered_pairs, plan)
        evaluation = evaluate_grid(state, exits, exit_ids, initial_grid)
        attempt_end = perf_counter()
        construction_durations.append(attempt_end - attempt_started_at)

        if probe_enabled:
            # Probe: a short improve_grid from this attempt's grid, so the
            # selection below sees a proxy for the post-search basin, not
            # just the raw construction snapshot. improve_grid restores
            # state.grid to the best grid it found (which can only be >= the
            # probe's own start), so state.grid is already the polished grid.
            probe_budget_left = solve_deadline - search_reserve - attempt_end
            this_probe = min(probe_seconds, max(probe_budget_left, 0.0))
            if this_probe > 0:
                evaluation = improve_grid(
                    state,
                    exits,
                    exit_ids,
                    initial_grid,
                    evaluation,
                    attempt_end + this_probe,
                    widen_strategy=DEFAULT_WIDEN_STRATEGY,
                    max_widen_rounds=DEFAULT_MAX_WIDEN_ROUNDS,
                    widen_pairs_per_round=DEFAULT_WIDEN_PAIRS_PER_ROUND,
                )
            candidate_grid = [row[:] for row in state.grid]

        if evaluation > best_evaluation:
            best_grid = candidate_grid
            best_evaluation = evaluation

    state.grid = [row[:] for row in best_grid]
    # Polish the winner to the solve deadline. Local search runs right up to
    # whatever deadline it is given (avg/max ~9.03s/9.06s observed locally
    # for a 9.0s deadline), so the margin to the 10s hard limit is set by
    # SOLVE_DEADLINE_SECONDS above, not here.
    best_evaluation = improve_grid(
        state,
        exits,
        exit_ids,
        initial_grid,
        best_evaluation,
        solve_deadline,
        widen_strategy=DEFAULT_WIDEN_STRATEGY,
        max_widen_rounds=DEFAULT_MAX_WIDEN_ROUNDS,
        widen_pairs_per_round=DEFAULT_WIDEN_PAIRS_PER_ROUND,
    )
    return moves_to_reach_grid(initial_grid, state.grid)
