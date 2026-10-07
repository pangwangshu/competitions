"""Summarize a tester run into the CSVs behind the report's figures.

Run the tester with its save options first, so every case's input and the
solver's output land in one folder, alongside the tester's own score lines:

    mkdir -p runs/final
    PYTHONHASHSEED=0 java -jar tools/tester.jar -exec "python3 src/main.py" \
        -seed 1,250 -threads 4 -novis -printRuntime \
        -saveSolInput runs/final -saveSolOutput runs/final > runs/final/scores.txt

    python3 scripts/summarize_run.py runs/final report/data

For each seed this replays the output moves on the input grid, re-scores the
result independently of the tester (and checks the two agree), and records
whether each target pair was matched together with its exit-to-exit hex
distance. It also rebuilds the solver's first construction attempt from the
same input, to show what construction alone achieves before local search.
Writes:

    <out_dir>/seeds.csv              one row per seed
    <out_dir>/match_by_distance.csv  pairs and matched pairs per hex distance

An optional third argument names a folder of inputs captured with the
tester's -showOriginal flag (the hidden grid the generator shuffled). Each
seed then also gets the score of the reference solution "rotate every tile
back to the hidden grid", which matches every pair by construction:

    java -jar tools/tester.jar -exec "python3 scripts/read_only.py" \
        -seed 1,250 -novis -showOriginal -saveSolInput runs/hidden
    python3 scripts/summarize_run.py runs/final report/data runs/hidden
"""
import csv
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hextiles.incremental import trace_path  # noqa: E402
from hextiles.model import read_input  # noqa: E402
from hextiles.solver import (  # noqa: E402
    build_exits,
    construct_grid,
    construction_attempts,
    evaluate_grid,
    hex_distance,
)

SCORE_LINE = re.compile(r"Seed = (\d+), Score = ([-\d.]+), RunTime = (\d+) ms")


def read_tester_scores(path):
    scores = {}
    for line in path.read_text().splitlines():
        match = SCORE_LINE.search(line)
        if match:
            scores[int(match.group(1))] = (float(match.group(2)), int(match.group(3)))
    return scores


def summarize_case(input_path, output_path, hidden_path=None):
    with open(input_path) as stream:
        state = read_input(stream)
    initial_grid = [row[:] for row in state.grid]
    exits, exit_ids = build_exits(state)

    ordered_pairs, plan = construction_attempts(state, exits)[0]
    construct_grid(state, exits, exit_ids, ordered_pairs, plan)
    construction = evaluate_grid(state, exits, exit_ids, initial_grid)
    state.grid = [row[:] for row in initial_grid]

    lines = output_path.read_text().split("\n")
    move_count = int(lines[0])
    for line in lines[1:1 + move_count]:
        row, column, direction = map(int, line.split())
        state.grid[row][column] = (state.grid[row][column] + direction) % 6

    matched = 0
    total_path_score = 0
    top_path = (0, 0, 0)  # (path score, length, bonus tiles) of the best matched path
    pair_rows = []
    for first_exit, second_exit in state.pairs:
        end_exit, length, bonus_count, _ = trace_path(state, exits, exit_ids, first_exit)
        is_matched = end_exit == second_exit
        if is_matched:
            matched += 1
            path_score = length * (bonus_count + 1)
            total_path_score += path_score
            top_path = max(top_path, (path_score, length, bonus_count))
        distance = hex_distance(exits[first_exit], exits[second_exit])
        pair_rows.append((distance, is_matched))

    score = max(matched * (total_path_score - move_count * state.m), 0)
    case = {
        "n": state.n,
        "m": state.m,
        "b": state.b,
        "tiles": sum(cell >= 0 for row in state.grid for cell in row),
        "pairs": len(state.pairs),
        "matched": matched,
        "total_path_score": total_path_score,
        "top_path_score": top_path[0],
        "top_path_length": top_path[1],
        "top_path_bonus": top_path[2],
        "moves": move_count,
        "score": score,
        "construction_matched": construction[1],
        "construction_path_score": construction[2],
        "construction_score": construction[0],
    }

    if hidden_path is not None:
        with open(hidden_path) as stream:
            hidden = read_input(stream)
        assert (hidden.n, hidden.pairs, hidden.bonus) == (state.n, state.pairs, state.bonus)
        state.grid = hidden.grid
        hidden_score, hidden_matched, hidden_path_score, negative_moves = evaluate_grid(
            state, exits, exit_ids, initial_grid
        )
        case.update({
            "hidden_matched": hidden_matched,
            "hidden_path_score": hidden_path_score,
            "hidden_moves": -negative_moves,
            "hidden_score": hidden_score,
        })
    return case, pair_rows


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    run_dir = pathlib.Path(sys.argv[1])
    out_dir = pathlib.Path(sys.argv[2])
    hidden_dir = pathlib.Path(sys.argv[3]) if len(sys.argv) == 4 else None
    out_dir.mkdir(parents=True, exist_ok=True)

    tester_scores = read_tester_scores(run_dir / "scores.txt")
    seed_rows = []
    by_distance = {}
    mismatches = 0
    for seed in sorted(tester_scores):
        hidden_path = hidden_dir / f"{seed}.in" if hidden_dir else None
        case, pair_rows = summarize_case(run_dir / f"{seed}.in", run_dir / f"{seed}.out", hidden_path)
        tester_score, runtime_ms = tester_scores[seed]
        if case["score"] != tester_score:
            mismatches += 1
            print(f"seed {seed}: re-scored {case['score']} but tester reported {tester_score}")
        seed_rows.append({"seed": seed, **case, "runtime_ms": runtime_ms})
        for distance, is_matched in pair_rows:
            pairs, matched = by_distance.get(distance, (0, 0))
            by_distance[distance] = (pairs + 1, matched + int(is_matched))

    with open(out_dir / "seeds.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)

    with open(out_dir / "match_by_distance.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["distance", "pairs", "matched"])
        for distance in sorted(by_distance):
            writer.writerow([distance, *by_distance[distance]])

    total_pairs = sum(row["pairs"] for row in seed_rows)
    total_matched = sum(row["matched"] for row in seed_rows)
    print(f"{len(seed_rows)} seeds, {total_pairs} pairs, {total_matched} matched "
          f"({100 * total_matched / total_pairs:.1f}%), "
          f"score total {sum(row['score'] for row in seed_rows):,.0f}, "
          f"max runtime {max(row['runtime_ms'] for row in seed_rows)} ms, "
          f"{mismatches} score mismatches vs tester")


if __name__ == "__main__":
    main()
