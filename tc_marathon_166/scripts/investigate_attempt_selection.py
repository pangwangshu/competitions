"""E1 (Tier 5 reinvestigation): does solve()'s attempt selection -- pick the
construction attempt with the highest immediate-post-construction
evaluate_grid score -- actually pick the attempt that polishes best under
improve_grid?

feat/tier5's be6d50d root-caused Tier 5's e2e deficit to exactly this
pre-existing selection weakness (6/6 sampled cases picked the wrong attempt,
8.5%-33.6% of achievable final score left on the table), but measured it on
the OLD Tier 2.5 codebase. This script re-measures it on the CURRENT
tier-2.5-take-2 code (Tier 3 adaptive scheduler + widening random/2/10),
using REAL tester-captured cases (not synthetic make_random_state ones).

Method per case:
  1. Rebuild the exact attempt list construction_attempts() returns.
  2. Construct each attempt from the same initial grid; record evaluate_grid.
  3. Polish EVERY attempt's grid with improve_grid for
     target_local_search_seconds(n) (production's own local-search reserve)
     under production widen defaults (random/2/10).
  4. Report: construction-score pick vs oracle pick-by-final-score, and how
     much final score the production rule leaves on the table.

Usage: python3 scripts/investigate_attempt_selection.py case_file [case_file ...]
"""
import pathlib
import sys
from time import perf_counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hextiles.model import read_input  # noqa: E402
from hextiles.solver import (  # noqa: E402
    build_exits,
    construction_attempts,
    construct_grid,
    evaluate_grid,
    improve_grid,
    target_local_search_seconds,
)

PRODUCTION_WIDEN = {"widen_strategy": "random", "max_widen_rounds": 2, "widen_pairs_per_round": 10}


def run_case(case_path):
    with open(case_path) as stream:
        state = read_input(stream)
    exits, exit_ids = build_exits(state)
    initial_grid = [row[:] for row in state.grid]
    attempts = construction_attempts(state, exits)
    reserve = target_local_search_seconds(state.n)

    rows = []
    for index, (ordered_pairs, plan) in enumerate(attempts):
        state.grid = [row[:] for row in initial_grid]
        candidate_grid = construct_grid(state, exits, exit_ids, ordered_pairs, plan)
        construction_eval = evaluate_grid(state, exits, exit_ids, initial_grid)

        state.grid = [row[:] for row in candidate_grid]
        started = perf_counter()
        final_eval = improve_grid(
            state, exits, exit_ids, initial_grid, construction_eval,
            started + reserve, **PRODUCTION_WIDEN,
        )
        rows.append((index, construction_eval, final_eval))

    construction_pick = max(rows, key=lambda row: row[1])
    oracle_pick = max(rows, key=lambda row: row[2])
    gap = oracle_pick[2][0] - construction_pick[2][0]
    gap_pct = 100.0 * gap / max(oracle_pick[2][0], 1)

    print(f"\n=== {pathlib.Path(case_path).name} N={state.n} M={state.m} "
          f"B={state.b} pairs={len(state.pairs)} attempts={len(rows)} reserve={reserve:.2f}s ===")
    print(f"  {'attempt':>7} {'construction_score':>18} {'matched':>7} "
          f"{'final_score':>12} {'final_matched':>13}")
    for index, construction_eval, final_eval in rows:
        marker = ""
        if index == construction_pick[0]:
            marker += " <- construction pick"
        if index == oracle_pick[0]:
            marker += " <- ORACLE pick"
        print(f"  {index:>7} {construction_eval[0]:>18} {construction_eval[1]:>7} "
              f"{final_eval[0]:>12} {final_eval[1]:>13}{marker}")
    print(f"  selection gap: {gap} ({gap_pct:.2f}%) "
          f"{'WRONG PICK' if gap > 0 else 'correct pick'}")
    return construction_pick[2][0], oracle_pick[2][0]


def main():
    total_picked = 0
    total_oracle = 0
    wrong = 0
    for case_path in sys.argv[1:]:
        picked, oracle = run_case(case_path)
        total_picked += picked
        total_oracle += oracle
        if oracle > picked:
            wrong += 1
    print(f"\n=== TOTAL: production-pick {total_picked} vs oracle {total_oracle} "
          f"({100.0 * (total_oracle - total_picked) / max(total_oracle, 1):.2f}% left on table), "
          f"wrong pick in {wrong}/{len(sys.argv) - 1} cases ===")


if __name__ == "__main__":
    main()
