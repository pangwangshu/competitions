#!/usr/bin/env python3
"""Compare each game's line bound with the largest product the solver played.

    python3 report/line_bound.py runs/grids report/data/anatomy_final.csv > report/data/line_bound.csv

The line bound of a board is the largest, over rows and columns, product of the
T largest word multipliers (DW = 2, TW = 3) in that line. One move places at
most T tiles, so no play's main word can have a larger word-multiplier product.
runs/grids/<seed>.txt holds "N T" and the N bonus rows. To write them, run the
tester with this file as the solver (it dumps the grid and ends the game):

    cd tools && java -jar tester.jar -seed 30001 -novis -noanimate \
        -exec "python3 ../report/line_bound.py --dump ../runs/grids/30001.txt"
"""
import csv
import math
import sys


def stdin_tokens():
    # Line by line: the tester keeps stdin open until the game ends.
    for line in sys.stdin:
        yield from line.split()


def dump(path):
    tokens = stdin_tokens()
    n, t, x, _penalty, count = (int(next(tokens)) for _ in range(5))
    for _ in range(count + t + x):
        next(tokens)
    rows = [" ".join(next(tokens) for _ in range(n)).replace(" ", "") for _ in range(n)]
    with open(path, "w") as f:
        f.write(f"{n} {t}\n" + "\n".join(rows) + "\n")
    print(-1, flush=True)


def line_bound(n, t, rows):
    mult = {"D": 2, "T": 3}
    lines = rows + ["".join(rows[r][c] for r in range(n)) for c in range(n)]
    return max(math.prod(sorted((mult.get(ch, 1) for ch in line), reverse=True)[:t]) for line in lines)


def main(grids, anatomy):
    out = csv.writer(sys.stdout)
    out.writerow(["seed", "N", "T", "line_bound", "board_best_product", "max_product", "score", "top1"])
    for row in csv.DictReader(open(anatomy)):
        head, *rows = open(f"{grids}/{row['seed']}.txt").read().split()
        n, t = int(head), int(rows.pop(0))
        assert t == int(row["T"]) and n == int(row["N"])
        out.writerow([row["seed"], n, t, line_bound(n, t, rows), row["board_best_product"],
                      row["max_product"], row["score"], row["top1"]])


if __name__ == "__main__":
    if sys.argv[1] == "--dump":
        dump(sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2])
