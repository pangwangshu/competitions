"""E2 (Tier 5 reinvestigation): how much score headroom exists for upgrading
ALREADY-MATCHED pairs' paths at all? This is the decisive "is Tier 5/6 worth
doing" measurement:

- Tier 5's premise is that construction bakes in short/cheap paths that local
  search inherits. If, starting from the production solver's FINAL grid, even
  an offline oracle with ~unbounded compute per pair cannot find true-eval
  improvements by re-routing matched pairs through longer/bonus-richer
  corridors, then there is nothing for ANY Tier 5/6 variant to recover and
  the tier should be abandoned. If the headroom is large, the question
  becomes whether a cheap-enough online operator exists.

- The oracle is deliberately generous: corridor slack 8 around the pair's
  direct line (plus the pair's own tiles, always freed), rotation cost budget
  max(10, current length), hop cap min(length+30, 120), per-call deadline 3s,
  bounded conflict retries. Serial accept-only-if-true-evaluation-improves
  via IncrementalEvaluator.preview_flips -- the strongest monotone online
  acceptance rule possible, with full visibility of side effects on other
  pairs. This dominates any shippable Tier 6 variant, so its delta is an
  UPPER BOUND for the whole direction.

- Opportunity-cost yardstick: afterwards, the same final grid is handed back
  to plain improve_grid for exactly as long as the oracle took. If plain
  hill-climbing recovers as much as the oracle, the headroom isn't specific
  to route upgrading at all.

Usage: python3 scripts/investigate_upgrade_oracle.py case_file [case_file ...]
"""
import pathlib
import sys
from time import perf_counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hextiles.geometry import S, minimum_rotation, partner_at, rotation_distance  # noqa: E402
from hextiles.incremental import IncrementalEvaluator  # noqa: E402
from hextiles.model import read_input  # noqa: E402
from hextiles.solver import (  # noqa: E402
    MAX_CONFLICT_RETRIES,
    build_exits,
    evaluate_grid,
    hex_distance,
    improve_grid,
    move_to_neighbor,
    solve,
)

PRODUCTION_WIDEN = {"widen_strategy": "random", "max_widen_rounds": 2, "widen_pairs_per_round": 10}

ORACLE_CORRIDOR_SLACK = 8
ORACLE_HOP_MARGIN = 30
ORACLE_MAX_HOPS_CAP = 120
ORACLE_CALL_DEADLINE_SECONDS = 3.0
ORACLE_PASSES = 2


def corridor_tiles(state, exits, source_exit, target_exit, corridor_slack):
    source_position = exits[source_exit][:2]
    target_position = exits[target_exit][:2]
    direct_distance = hex_distance(source_position, target_position)
    allowed = set()
    for row in range(state.w):
        for column in range(state.w):
            if state.grid[row][column] < 0:
                continue
            detour = (
                hex_distance(source_position, (row, column))
                + hex_distance((row, column), target_position)
                - direct_distance
            )
            if detour <= corridor_slack:
                allowed.add((row, column))
    return allowed


def longest_path_dp(state, exits, exit_ids, free_tiles, blocked_tiles,
                    source_exit, target_exit, cost_budget, max_hops, deadline):
    """Bounded-hop-count layered DP over (node, hop) -- tier-5-lite's
    _connect_once_longest technique, reimplemented pure (returns the path
    instead of committing rotations) so the caller can route it through
    IncrementalEvaluator.preview_flips for true-eval acceptance.

    Tiles outside free_tiles are pass-through: exactly one forced chord at
    zero rotation cost (identical to claimed-tile handling in
    _connect_once/_connect_once_longest), so the pair's own current route is
    always reproducible at hop=record.length, cost 0 -- the DP can only ever
    find equal-or-longer routes.
    """
    regular_nodes = state.w * state.w * S
    start_row, start_column, start_edge = exits[source_exit]
    start_node = ((start_row * state.w + start_column) * S) + start_edge
    goal_node = regular_nodes + target_exit

    dp = [{start_node: 0}] + [{} for _ in range(max_hops)]
    back = [{} for _ in range(max_hops + 1)]

    best_hop = None
    for hop in range(max_hops):
        if perf_counter() >= deadline:
            break
        layer = dp[hop]
        if not layer:
            break
        next_layer = dp[hop + 1]
        next_back = back[hop + 1]
        for node, cost in layer.items():
            if node >= regular_nodes:
                continue
            entry_edge = node % S
            tile_index = node // S
            row, column = divmod(tile_index, state.w)
            if (row, column) not in free_tiles:
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
                next_cost = cost + rotation_cost
                if next_cost > cost_budget:
                    continue
                if next_cost < next_layer.get(next_node, cost_budget + 1):
                    next_layer[next_node] = next_cost
                    next_back[next_node] = (node, exit_edge)
        if goal_node in next_layer:
            best_hop = hop + 1

    if best_hop is None:
        return None

    path = []
    hop = best_hop
    node = goal_node
    while hop > 0:
        previous_node, exit_edge = back[hop][node]
        path.append((previous_node, exit_edge))
        node = previous_node
        hop -= 1
    path.reverse()
    return path


def path_to_flips(state, free_tiles, path):
    tile_orientations = {}
    for node, exit_edge in path:
        entry_edge = node % S
        row, column = divmod(node // S, state.w)
        if (row, column) not in free_tiles:
            continue
        if (row, column) in tile_orientations:
            if partner_at(entry_edge, tile_orientations[(row, column)]) != exit_edge:
                return None, (row, column)
            continue
        current_orientation = state.grid[row][column]
        tile_orientations[(row, column)] = min(
            (
                orientation
                for orientation in range(S)
                if partner_at(entry_edge, orientation) == exit_edge
            ),
            key=lambda orientation: rotation_distance(current_orientation, orientation),
        )
    return {
        tile: orientation
        for tile, orientation in tile_orientations.items()
        if orientation != state.grid[tile[0]][tile[1]]
    }, None


def oracle_upgrade_pair(state, exits, exit_ids, evaluator, first_exit, second_exit, deadline):
    record = evaluator.path_of[first_exit]
    if not (record.matched and record.partner == second_exit):
        return None
    free_tiles = corridor_tiles(state, exits, first_exit, second_exit, ORACLE_CORRIDOR_SLACK) | record.tiles
    cost_budget = max(10, record.length)
    max_hops = min(record.length + ORACLE_HOP_MARGIN, ORACLE_MAX_HOPS_CAP)

    blocked_tiles = set()
    for _ in range(MAX_CONFLICT_RETRIES):
        if perf_counter() >= deadline:
            return None
        path = longest_path_dp(
            state, exits, exit_ids, free_tiles, blocked_tiles,
            first_exit, second_exit, cost_budget, max_hops, deadline,
        )
        if path is None:
            return None
        flips, conflict_tile = path_to_flips(state, free_tiles, path)
        if conflict_tile is None:
            break
        blocked_tiles.add(conflict_tile)
    else:
        return None

    if not flips:
        return None
    return evaluator.preview_flips(flips)


def run_case(case_path):
    with open(case_path) as stream:
        state = read_input(stream)
    exits, exit_ids = build_exits(state)
    initial_grid = [row[:] for row in state.grid]

    solve(state)  # production pipeline; state.grid is now the final grid
    final_grid = [row[:] for row in state.grid]
    base_eval = evaluate_grid(state, exits, exit_ids, initial_grid)

    target_matches = {}
    for first_exit, second_exit in state.pairs:
        target_matches[first_exit] = second_exit
        target_matches[second_exit] = first_exit
    evaluator = IncrementalEvaluator(state, exits, exit_ids, initial_grid, target_matches)
    current_eval = base_eval

    oracle_started = perf_counter()
    upgrades = 0
    for _ in range(ORACLE_PASSES):
        improved = False
        for first_exit, second_exit in state.pairs:
            result = oracle_upgrade_pair(
                state, exits, exit_ids, evaluator, first_exit, second_exit,
                perf_counter() + ORACLE_CALL_DEADLINE_SECONDS,
            )
            if result is not None and result.evaluation > current_eval:
                evaluator.commit_flips(result)
                current_eval = result.evaluation
                upgrades += 1
                improved = True
        if not improved:
            break
    oracle_elapsed = perf_counter() - oracle_started
    oracle_eval = current_eval

    # Opportunity-cost yardstick: same final grid, same extra wall-clock,
    # plain improve_grid instead.
    state.grid = [row[:] for row in final_grid]
    polish_started = perf_counter()
    extra_eval = improve_grid(
        state, exits, exit_ids, initial_grid, base_eval,
        polish_started + oracle_elapsed, **PRODUCTION_WIDEN,
    )

    print(f"=== {pathlib.Path(case_path).name} N={state.n} M={state.m} B={state.b} "
          f"pairs={len(state.pairs)} ===")
    print(f"  production final:            score={base_eval[0]} matched={base_eval[1]} "
          f"path_score={base_eval[2]} moves={-base_eval[3]}")
    print(f"  oracle upgrades:             score={oracle_eval[0]} matched={oracle_eval[1]} "
          f"path_score={oracle_eval[2]} moves={-oracle_eval[3]} "
          f"(+{oracle_eval[0] - base_eval[0]}, {upgrades} upgrades, {oracle_elapsed:.1f}s)")
    print(f"  equal-time extra polish:     score={extra_eval[0]} matched={extra_eval[1]} "
          f"(+{extra_eval[0] - base_eval[0]})")
    return base_eval[0], oracle_eval[0], extra_eval[0]


def main():
    totals = [0, 0, 0]
    for case_path in sys.argv[1:]:
        base, oracle, extra = run_case(case_path)
        totals[0] += base
        totals[1] += oracle
        totals[2] += extra
    print(f"\n=== TOTAL over {len(sys.argv) - 1} cases: production {totals[0]}, "
          f"oracle {totals[1]} ({100.0 * (totals[1] - totals[0]) / max(totals[0], 1):+.2f}%), "
          f"equal-time polish {totals[2]} ({100.0 * (totals[2] - totals[0]) / max(totals[0], 1):+.2f}%) ===")


if __name__ == "__main__":
    main()
