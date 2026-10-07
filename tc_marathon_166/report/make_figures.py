"""Regenerate the write-up's figures from the small files in report/data/.

    python report/make_figures.py

Needs matplotlib + numpy, plus the solver package in src/ (for the board
rendering). Every number plotted lives in report/data/: seeds.csv and
match_by_distance.csv come from scripts/summarize_run.py, submissions.csv and
changes.csv were transcribed from the development log, and board_seed*.in/out
are one tester case and the solver's answer to it.
"""

import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, PathPatch, Polygon  # noqa: E402
from matplotlib.path import Path  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from hextiles.geometry import CONNECTIONS, DC, DR, S, partner_at  # noqa: E402
from hextiles.model import read_input  # noqa: E402
from hextiles.solver import build_exits, evaluate_grid  # noqa: E402

BOARD_SEED = 199

# Reference palette (light surface), shared with the kaggriculture report.
# Text never wears a series color.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#f0efec"
S1 = "#2a78d6"  # blue
S2 = "#eb6834"  # orange
BONUS_FILL = "#fbe9c4"  # light tint of the palette's yellow slot

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.labelcolor": INK2,
})


def read_csv(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", os.path.relpath(path))


def dots(ax, x, y, color, size=46, marker="o", zorder=3, **kw):
    """>=8px marker with a 2px surface ring."""
    return ax.scatter(x, y, s=size, color=color, edgecolors=SURFACE,
                      linewidths=1.6, marker=marker, zorder=zorder, **kw)


# ---------------------------------------------------------------- Figure 1
SQRT3 = math.sqrt(3)


def tile_center(row, column):
    return column + row / 2, -row * SQRT3 / 2


def edge_midpoint(center, edge):
    dx = DC[edge] + DR[edge] / 2
    dy = -DR[edge] * SQRT3 / 2
    return center[0] + dx / 2, center[1] + dy / 2


def hex_corners(center):
    radius = 1 / SQRT3
    return [(center[0] + radius * math.cos(math.radians(30 + 60 * k)),
             center[1] + radius * math.sin(math.radians(30 + 60 * k))) for k in range(6)]


def classify_chords(state, exits, exit_ids):
    """Map (row, column, frozenset(edges)) -> 'top' | 'matched' for every chord
    on a matched path; anything absent is unmatched or part of a loop."""
    target = {}
    for a, b in state.pairs:
        target[a] = b
        target[b] = a
    paths = []
    for a, b in state.pairs:
        row, column, entry = exits[a]
        chords = []
        bonus = set()
        while True:
            exit_edge = partner_at(entry, state.grid[row][column])
            chords.append((row, column, frozenset((entry, exit_edge))))
            if (row, column) in state.bonus:
                bonus.add((row, column))
            nr, nc = row + DR[exit_edge], column + DC[exit_edge]
            if not (0 <= nr < state.w and 0 <= nc < state.w and state.grid[nr][nc] >= 0):
                end = exit_ids[(row, column, exit_edge)]
                break
            row, column, entry = nr, nc, (exit_edge + 3) % S
        if end == b:
            paths.append((len(chords) * (len(bonus) + 1), chords))
    paths.sort(key=lambda p: p[0], reverse=True)
    kind = {}
    for rank, (_, chords) in enumerate(paths):
        for chord in chords:
            kind.setdefault(chord, "top" if rank == 0 else "matched")
    return kind


def draw_board(ax, state, exits, exit_ids):
    kind = classify_chords(state, exits, exit_ids)
    styles = {None: (AXIS, 1.0, 1), "matched": (S1, 1.5, 2), "top": (S2, 2.1, 3)}
    for row in range(state.w):
        for column in range(state.w):
            if state.grid[row][column] < 0:
                continue
            center = tile_center(row, column)
            fill = BONUS_FILL if (row, column) in state.bonus else SURFACE
            ax.add_patch(Polygon(hex_corners(center), closed=True, facecolor=fill,
                                 edgecolor=GRID, linewidth=0.7, zorder=0))
            for first, second in CONNECTIONS:
                a = (first + state.grid[row][column]) % S
                b = (second + state.grid[row][column]) % S
                color, width, z = styles[kind.get((row, column, frozenset((a, b))))]
                path = Path([edge_midpoint(center, a), center, edge_midpoint(center, b)],
                            [Path.MOVETO, Path.CURVE3, Path.CURVE3])
                ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color,
                                       linewidth=width, capstyle="round", zorder=z))
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")


def fig_board():
    with open(os.path.join(DATA, f"board_seed{BOARD_SEED}.in")) as f:
        state = read_input(f)
    exits, exit_ids = build_exits(state)
    initial = [row[:] for row in state.grid]
    before = evaluate_grid(state, exits, exit_ids, initial)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
    draw_board(axes[0], state, exits, exit_ids)

    with open(os.path.join(DATA, f"board_seed{BOARD_SEED}.out")) as f:
        lines = f.read().split("\n")
    for line in lines[1:1 + int(lines[0])]:
        row, column, direction = map(int, line.split())
        state.grid[row][column] = (state.grid[row][column] + direction) % S
    after = evaluate_grid(state, exits, exit_ids, initial)
    draw_board(axes[1], state, exits, exit_ids)

    pairs = len(state.pairs)
    axes[0].set_title(f"Input: {before[1]} of {pairs} pairs connected, score {before[0]:,}",
                      fontsize=10.5, fontweight="normal", color=INK2, loc="center")
    axes[1].set_title(f"Output: {after[1]} of {pairs} connected, {-after[3]} moves, "
                      f"score {after[0]:,}", fontsize=10.5, fontweight="normal",
                      color=INK2, loc="center")
    fig.suptitle(f"One test case (seed {BOARD_SEED}: N = {state.n}, {state.b} bonus tiles, "
                 f"move cost M = {state.m}) before and after the solver",
                 x=0.06, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK)
    handles = [
        Line2D([], [], color=S2, lw=2.4, label="Highest-scoring path"),
        Line2D([], [], color=S1, lw=1.8, label="Other connected target pairs"),
        Line2D([], [], color=AXIS, lw=1.4, label="Unmatched paths and loops"),
        Patch(facecolor=BONUS_FILL, edgecolor=GRID, label="Bonus tile"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0),
               fontsize=9.5, handlelength=2.2)
    fig.subplots_adjust(wspace=0.04, top=0.84, bottom=0.07, left=0.02, right=0.98)
    save(fig, "fig1_board.png")


# ---------------------------------------------------------------- Figure 2
SHORT_LABELS = {
    1: "Minimal greedy", 2: "Multi-order + climb", 3: "Tier 0+1", 4: "Tier 2",
    5: "Tier 2 again", 6: "Widen spatial/2", 7: "Widen random/1", 8: "Same, 8.5 s",
    9: "Widen random/2", 10: "Route library", 11: "Take-2, 8.5 s", 12: "Take-2, 9.5 s",
    13: "random/1 again", 14: "+ probe, 9.5 s", 15: "+ multiprocess", 16: "Final",
}


def fig_submissions():
    rows = read_csv("submissions.csv")
    x = np.array([int(r["order"]) for r in rows])
    y = np.array([float(r["score"]) for r in rows])
    fig, ax = plt.subplots(figsize=(9.4, 4.6))

    days = []
    for r in rows:
        if not days or days[-1][0] != r["date"]:
            days.append([r["date"], int(r["order"]), int(r["order"])])
        days[-1][2] = int(r["order"])
    for k, (day, a, b) in enumerate(days):
        if k % 2 == 1:
            ax.axvspan(a - 0.5, b + 0.5, color=BAND, zorder=0, lw=0)
        ax.text((a + b) / 2, 63, "Aug " + day[-2:].lstrip("0"), ha="center",
                va="bottom", color=INK2, fontsize=8.5)

    lo = min(v for o, v in zip(x, y) if o >= 3 and v < 55)
    hi = max(v for o, v in zip(x, y) if o >= 3 and v < 55)
    ax.fill_between([2.6, 16.4], lo, hi, color=S1, alpha=0.08, lw=0, zorder=1)
    ax.text(9.6, lo - 1.2, f"13 of 14 readings from Tier 0+1 on fall in {lo:.1f}–{hi:.1f}",
            ha="center", va="top", color=INK2, fontsize=8.5)

    ax.plot(x, y, color=S1, lw=1, alpha=0.35, zorder=2)
    dots(ax, x[:-1], y[:-1], S1)
    dots(ax, x[-1:], y[-1:], S2, size=70, marker="D", zorder=4)

    def note(order, text, dx, dy, ha="left"):
        v = y[order - 1]
        ax.annotate(text, (order, v), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", color=INK, fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    note(1, "11.4", 10, 0)
    note(2, "35.6", 10, 0)
    note(8, "55.0: only change was the\ndeadline, 9.0 s → 8.5 s", -14, 16, ha="right")
    note(13, "49.7: the 55.0 build,\nsubmitted again", 0, -38, ha="center")
    note(16, "Final 52.4", 0, 20, ha="center")

    ax.set_xlim(0.4, 16.6)
    ax.set_ylim(0, 62)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[int(o)] for o in x], rotation=55, ha="right",
                       fontsize=8)
    ax.set_ylabel("Provisional score (0–100)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Leaderboard reading of every submission, in build order", pad=24)
    save(fig, "fig2_submissions.png")


# ---------------------------------------------------------------- Figure 3
def fig_changes():
    rows = read_csv("changes.csv")
    shipped = [r for r in rows if r["outcome"] == "shipped"]
    rejected = [r for r in rows if r["outcome"] == "rejected"]
    ordered = shipped + [None] + rejected
    labels, values, colors = [], [], []
    for r in ordered:
        if r is None:
            labels.append("")
            values.append(np.nan)
            colors.append(SURFACE)
            continue
        labels.append(r["change"])
        values.append(float(r["net_pct"]))
        colors.append(S1 if r["outcome"] == "shipped" else S2)
    ypos = np.arange(len(ordered))[::-1]

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.barh(ypos, values, height=0.62, color=colors, zorder=2)
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    for yy, r, v in zip(ypos, ordered, values):
        if r is None:
            continue
        ax.text(v + (0.6 if v >= 0 else -0.6), yy, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", color=INK, fontsize=8.5)
        ax.text(35.5, yy, f"n = {r['seeds']}", va="center", ha="right", color=MUTED,
                fontsize=8)
        if r["note"].startswith("shipped as a safety margin"):
            ax.text(2.2, yy, "kept as a margin (one leaderboard reading: +4.2)", va="center",
                    color=INK2, fontsize=8)
        if r["note"].startswith("48.98"):
            ax.text(v + 6.0, yy, "but 48.98 on the leaderboard", va="center",
                    color=INK2, fontsize=8)
    ticks = [(yy, label) for yy, label in zip(ypos, labels) if label]
    ax.set_yticks([yy for yy, _ in ticks])
    ax.set_yticklabels([label for _, label in ticks], fontsize=8.8)
    ax.set_xlim(-27, 36)
    ax.set_xlabel("Net change in summed raw score on the local tester (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=S1, label="Shipped"), Patch(color=S2, label="Rejected")],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=9)
    ax.set_title("Every substantive change, measured end to end before shipping", pad=26)
    save(fig, "fig3_changes.png")


# ---------------------------------------------------------------- Figure 5
BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 9), (10, 14), (15, 19), (20, 24),
           (25, 29), (30, 38)]


def fig_distance():
    rows = read_csv("match_by_distance.csv")
    pairs = {int(r["distance"]): int(r["pairs"]) for r in rows}
    matched = {int(r["distance"]): int(r["matched"]) for r in rows}
    total = sum(pairs.values())
    labels, rate, share = [], [], []
    for a, b in BUCKETS:
        p = sum(pairs.get(d, 0) for d in range(a, b + 1))
        m = sum(matched.get(d, 0) for d in range(a, b + 1))
        labels.append(str(a) if a == b else f"{a}–{b}")
        rate.append(100 * m / p)
        share.append(100 * p / total)
    weighted = (sum(d * matched[d] for d in matched) / sum(d * pairs[d] for d in pairs))
    overall = 100 * sum(matched.values()) / total

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True,
                                      gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.18})
    xs = np.arange(len(labels))
    top.plot(xs, rate, color=S1, lw=2, zorder=2)
    dots(top, xs, rate, S1)
    top.axhline(overall, color=MUTED, lw=1, zorder=1)
    top.text(len(xs) - 0.6, overall + 2, f"All pairs: {overall:.1f}%", ha="right",
             va="bottom", color=INK2, fontsize=8.5)
    for i in (0, 1, 7):
        top.annotate(f"{rate[i]:.0f}%", (xs[i], rate[i]), xytext=(9, 4),
                     textcoords="offset points", color=INK, fontsize=8.5)
    top.set_ylim(0, 100)
    top.set_ylabel("Pairs connected (%)")
    top.grid(axis="x", visible=False)
    top.set_title(f"Connection rate by exit-to-exit hex distance "
                  f"({total:,} target pairs, 250 seeds)", pad=10)

    bottom.bar(xs, share, width=0.62, color=S1, alpha=0.55, zorder=2)
    for i, v in enumerate(share):
        bottom.text(xs[i], v + 1.2, f"{v:.0f}%" if v >= 1 else f"{v:.1f}%",
                    ha="center", va="bottom", color=INK2, fontsize=8)
    bottom.set_ylim(0, 45)
    bottom.set_ylabel("Share of pairs (%)")
    bottom.set_xticks(xs)
    bottom.set_xticklabels(labels)
    bottom.set_xlabel("Hex distance between the pair's two exit tiles")
    bottom.grid(axis="x", visible=False)
    bottom.text(len(xs) - 0.6, 38,
                f"Weighted by distance, only {100 * weighted:.0f}% of demand is connected",
                ha="right", va="center", color=INK, fontsize=9)
    save(fig, "fig5_match_by_distance.png")


# ---------------------------------------------------------------- Figure 4
def fig_anatomy():
    rows = read_csv("seeds.csv")
    num = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    construction = num("construction_score")
    final = num("score")
    share = 100 * num("top_path_score") / num("total_path_score")
    tiles = num("tiles")
    top_length = num("top_path_length")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2),
                             gridspec_kw={"wspace": 0.32})

    ax = axes[0]
    keep = construction > 0
    dots(ax, construction[keep], final[keep], S1, size=24, zorder=3)
    lim = (10, 3e6)
    ax.plot(lim, lim, color=MUTED, lw=1, zorder=1)
    ax.text(2.2e5, 1.4e5, "no gain", color=INK2, fontsize=8.5, rotation=33)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ratio = np.median(final[keep] / construction[keep])
    ax.set_title(f"(a) Search multiplies the score\n(median ×{ratio:.1f})",
                 fontsize=11)
    ax.set_xlabel("Score after the first construction attempt")
    ax.set_ylabel("Final score")

    ax = axes[1]
    bins = np.arange(20, 101, 5)
    ax.hist(share, bins=bins, color=S1, alpha=0.85, rwidth=0.86, zorder=2)
    med = np.median(share)
    ax.axvline(med, color=INK2, lw=1, zorder=3)
    ax.text(med - 1.5, ax.get_ylim()[1] * 0.92, f"median {med:.0f}%", ha="right",
            color=INK, fontsize=8.5)
    ax.set_title("(b) One path carries most of the value", fontsize=11)
    ax.set_xlabel("Best path's share of total path score (%)")
    ax.set_ylabel("Seeds")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    dots(ax, tiles, top_length, S1, size=24, zorder=3)
    xs = np.array([0, 1200])
    ax.plot(xs, 3 * xs, color=MUTED, lw=1, zorder=1)
    ax.plot(xs, xs, color=AXIS, lw=1, zorder=1)
    ax.text(1215, 3500, "3 × tiles\n(every chord)", ha="left", va="center",
            color=INK2, fontsize=8.5, clip_on=False)
    ax.text(1215, 1200, "1 × tiles", ha="left", va="center", color=INK2, fontsize=8.5,
            clip_on=False)
    med_ratio = np.median(top_length / tiles)
    ax.set_title(f"(c) Its length vs. board size\n(median {med_ratio:.2f} × tiles)",
                 fontsize=11)
    ax.set_xlabel("Tiles on the board")
    ax.set_ylabel("Best path length (chords)")
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 3600)
    fig.suptitle("Anatomy of a solution, 250 seeds", x=0.07, y=1.06, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    save(fig, "fig4_anatomy.png")


if __name__ == "__main__":
    fig_board()
    fig_submissions()
    fig_changes()
    fig_anatomy()
    fig_distance()
