"""Regenerate the write-up's figures from the small files in report/data/.

    python report/make_figures.py

Needs matplotlib + numpy only. Every number plotted lives in report/data/:
lineage*.csv, anatomy_*.csv, per_tile_*.csv, product_*.csv, effort_final.csv,
multiplier_cells_final.csv and board_final.json come from the fresh 200-seed
run described in Appendix B; final5000.csv is the 5,000-game release run;
leaderboard.csv was parsed from the final standings page; submissions.csv,
changes.csv and budget_curve.csv were transcribed from the development log.
"""

import csv
import json
import math
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")

# Reference palette (light surface), shared with the MM166 and kaggriculture
# reports. Text never wears a series color.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#f0efec"
S1 = "#2a78d6"  # blue
S2 = "#eb6834"  # orange
S3 = "#1baf7a"  # aqua (below 3:1 on the surface: always direct-labelled)
REST = "#c3c2b7"  # context marks
LETTER_TINT = {"d": "#dbe8f8", "t": "#a9c9ef"}  # light steps of the blue ramp
WORD_TINT = {"D": "#fbdccd", "T": "#f6b394"}  # light steps of the orange ramp

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


def geo(c, r):
    return math.exp(statistics.mean(math.log((a + 1) / (b + 1)) for a, b in zip(c, r)))


def geo_ci(c, r, reps=2000, seed=167):
    logs = [math.log((a + 1) / (b + 1)) for a, b in zip(c, r)]
    rng = np.random.default_rng(seed)
    arr = np.array(logs)
    boots = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(reps)]
    return math.exp(np.mean(arr)), math.exp(np.percentile(boots, 2.5)), math.exp(np.percentile(boots, 97.5))


# ---------------------------------------------------------------- Figure 1
def fig_board():
    b = json.load(open(os.path.join(DATA, "board_final.json")))
    n, bonus, board = b["N"], b["bonus"], b["board"]
    plays = b["plays"]
    actions = [a for a in b["actions"] if a.startswith("PLACE")]
    top = max(range(len(plays)), key=lambda i: plays[i]["score"])
    top_action = actions[top].split()
    r0, c0, vert, word = int(top_action[1]), int(top_action[2]), top_action[3] == "V", top_action[4]
    top_cells = {(r0 + k, c0) if vert else (r0, c0 + k) for k in range(len(word))}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 6.0), gridspec_kw=dict(width_ratios=[1.05, 1]))
    ax.set_aspect("equal")
    ax.axis("off")
    for r in range(n):
        for c in range(n):
            ch = bonus[r * n + c]
            fill = LETTER_TINT.get(ch) or WORD_TINT.get(ch) or ("#e9e8e3" if ch == "S" else SURFACE)
            ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, facecolor=fill, edgecolor=GRID, linewidth=0.8))
            letter = board[r][c]
            if letter != ".":
                ax.add_patch(Rectangle((c + 0.08, n - 1 - r + 0.08), 0.84, 0.84, facecolor="#ffffff",
                                       edgecolor=AXIS, linewidth=0.6))
                ax.text(c + 0.5, n - 1 - r + 0.47, letter, ha="center", va="center", fontsize=8.5,
                        color=INK, fontweight="bold")
            elif ch != ".":
                label = {"d": "DL", "t": "TL", "D": "DW", "T": "TW", "S": "S"}[ch]
                ax.text(c + 0.5, n - 1 - r + 0.5, label, ha="center", va="center", fontsize=5.5, color=INK2)
    rr = [r for r, _ in top_cells]
    cc = [c for _, c in top_cells]
    ax.add_patch(Rectangle((min(cc), n - 1 - max(rr)), max(cc) - min(cc) + 1, max(rr) - min(rr) + 1,
                           facecolor="none", edgecolor=INK, linewidth=2.2))
    ax.set_xlim(-0.2, n + 0.2)
    ax.set_ylim(-1.6, n + 0.2)
    ax.set_title("(a) Final board, seed %d (N=%d, T=%d, X=%d, D=%d)" % (b["seed"], n, b["T"], b["X"], b["D"]))
    handles = [Patch(facecolor=LETTER_TINT["d"], edgecolor=GRID, label="DL"),
               Patch(facecolor=LETTER_TINT["t"], edgecolor=GRID, label="TL"),
               Patch(facecolor=WORD_TINT["D"], edgecolor=GRID, label="DW"),
               Patch(facecolor=WORD_TINT["T"], edgecolor=GRID, label="TW"),
               Patch(facecolor="none", edgecolor=INK, linewidth=2, label="highest-scoring play")]
    ax.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=8.5,
              handlelength=1.2, columnspacing=1.0)

    scores = [p["score"] for p in plays]
    cum = np.cumsum(scores)
    tiles = np.cumsum([p["tiles"] for p in plays])
    ax2.step(tiles, cum, where="post", color=S1, linewidth=2)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
    for i in order:
        dots(ax2, [tiles[i]], [cum[i]], S1)
        ax2.annotate("%s  +%d (×%d)" % (actions[i].split()[4], scores[i], plays[i]["product"]),
                     (tiles[i], cum[i]), xytext=(8, -12), textcoords="offset points", fontsize=8.5, color=INK2)
    total = b["score"]
    share = sum(scores[i] for i in order) / max(1, sum(scores))
    ax2.set_xlabel("tiles placed so far")
    ax2.set_ylabel("cumulative score")
    ax2.set_title("(b) Score over the game: three plays carry %d%%" % round(100 * share))
    ax2.text(0.02, 0.95, "final score %d, %d plays" % (total, len(plays)), transform=ax2.transAxes,
             color=INK2, fontsize=9, va="top")
    fig.tight_layout()
    save(fig, "fig1_board.png")


# ---------------------------------------------------------------- Figure 2
def box(ax, x, y, w, h, title, body, edge=AXIS, fill="#ffffff", title_size=9.6, footer=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.012",
                                facecolor=fill, edgecolor=edge, linewidth=1.3))
    ax.text(x + 0.008, y + h - 0.02, title, fontsize=title_size, fontweight="bold", color=INK, va="top")
    ax.text(x + 0.008, y + h - 0.058, body, fontsize=8.1, color=INK2, va="top", linespacing=1.35)
    if footer:
        ax.text(x + 0.008, y + 0.016, footer, fontsize=7.6, color=MUTED, va="bottom", linespacing=1.3)


def arrow(ax, a, b, color=MUTED, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, color=color, linewidth=1.3,
                                 linestyle=ls, connectionstyle="arc3,rad=%g" % rad, shrinkA=0, shrinkB=0))


def fig_architecture():
    fig = plt.figure(figsize=(13.2, 7.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.975, "One turn of the final solver (O-01C)", fontsize=13, fontweight="bold", color=INK, va="top")
    ax.text(0.02, 0.935, "Each stage narrows the candidate set and spends effort only on survivors; the parent move "
            "is always kept, so any stage can stop early and return a complete, legal action.",
            fontsize=9, color=INK2, va="top")
    y, h, w = 0.47, 0.38, 0.13
    xs = [0.02 + i * 0.163 for i in range(6)]
    stages = [
        ("1  Move generation", "Trie DFS from anchors\nwith cross-check masks\nand exact scoring.\n\nKeeps the top 512\nby immediate score\n(dedup by successor).", None),
        ("2  Lane prefilter", "key = score + μ·E[best\nlane play next turn]\nfrom the kept rack\nand visible refills.\n\n512 → 8…32 candidates\n(greedy best always in).", "reads: lane model\ncap: scheduler"),
        ("3  Two-ply value", "Q = score + E[best next]\n+ phase·α·V(board),\non common sampled\nrefills.\n\nSuccessive halving:\n32 → 16 → 8 → 4.", "cap: scheduler"),
        ("4  Rollout re-rank", "Top 4 + parent on 32\nshared tapes × 4 steps;\nthe continuation policy\nfishes (keeps letters\nfor a lane target).\n\nQ = ½·Q₂ + ½·Q_roll", "reads: lane model\ncap: scheduler"),
        ("5  Decision layer", "Triggered when no word,\nQ < 15, or a fishing\ntarget beats the play:\nPLACE vs ≤ 8 discard\nsubsets vs STOP on the\nsame tapes, 2-SE margin.", "reads: lane model\ncap: decision pool"),
        ("Action", "PLACE r c dir word\nDISCARD letters\n-1 (stop)\n\nKnown refills are\npredicted and checked\nagainst the feedback.", None),
    ]
    for i, (t, body, foot) in enumerate(stages):
        edge = S1 if 1 <= i <= 4 else AXIS
        box(ax, xs[i], y, w, h, t, body, edge=edge, footer=foot)
        if i:
            arrow(ax, (xs[i - 1] + w + 0.004, y + h / 2), (xs[i] - 0.004, y + h / 2), color=INK2)
    counts = ["≤ 512", "≤ 32", "4", "1", "1"]
    for i, ctext in enumerate(counts):
        ax.text((xs[i] + w + xs[i + 1]) / 2, y + h / 2 + 0.018, ctext, ha="center", fontsize=7.6, color=MUTED)

    yb, hb = 0.13, 0.26
    box(ax, xs[0], yb, w, hb, "Input",
        "N, T, X, D, dictionary,\nhand, X preview tiles,\nbonus grid. Feedback\nafter each action:\nelapsed ms, new hand\nand preview.", edge=AXIS)
    arrow(ax, (xs[0] + w / 2, yb + hb + 0.004), (xs[0] + w / 2, y - 0.006), color=INK2)
    box(ax, xs[1], yb, xs[2] + w - xs[1], hb, "Lane model (shared)",
        "Windows of ≤ T empty cells whose main word reuses a board\nletter and whose word-multiplier product ≥ 4. Exact word\nlists per window, cached by signature; refill odds of the\nmissing tiles (exact for ≤ 3); events merged by score.", edge=S2)
    box(ax, xs[3], yb, xs[4] + w - xs[3], hb, "Effort scheduler",
        "Effort = deterministic counts × fitted costs (generator nodes,\ncross cells, lane words, ...) in reference-machine ms.\nPlan 3000 ms + decision pool 1000 ms; turn cap = 3 × what\nis left ÷ expected turns. The wall clock is only a guard\n(7.5 s budget, measured slow-down factor).", edge=S2)
    box(ax, xs[5], yb, w, hb, "Shared state",
        "Rules (trie, bonus grid,\nletter frequencies)\nshared read-only; the\nmutable State is a small\ncopyable struct with\napply / undo.", edge=AXIS)
    ax.text(0.02, 0.06, "Blue: per-turn search stages.  Orange: shared models; each stage's footer says which it reads "
            "and where its effort cap comes from.", fontsize=8.5, color=INK2)
    save(fig, "fig2_architecture.png")


# ---------------------------------------------------------------- Figure 3
LINEAGE = ["B0", "P1-01", "P1-02", "P2-03", "INT-01", "O-01A", "O-01C"]
LABEL = {"B0": "B0\ngreedy", "P1-01": "P1-01\n2-ply", "P1-02": "P1-02\nlanes",
         "P2-03": "P2-03\nscheduler", "INT-01": "INT-01\nrollouts", "O-01A": "O-01A\nfishing",
         "O-01C": "O-01C\nfinal"}


def fig_lineage():
    seeds = read_csv("lineage_seeds.csv")
    b0 = [float(r["B0"]) for r in seeds]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    xs = np.arange(len(LINEAGE))
    vals, lo, hi = [], [], []
    for v in LINEAGE:
        g, a, b = geo_ci([float(r[v]) for r in seeds], b0)
        vals.append(g)
        lo.append(g - a)
        hi.append(b - g)
    ax.bar(xs, [v - 1 for v in vals], bottom=1, width=0.56, color=S1, zorder=2)
    ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=3)
    for x, v, up in zip(xs, vals, hi):
        ax.text(x, v + up + 0.02, "×%.2f" % v, ha="center", fontsize=8.8, color=INK)
    ax.set_xticks(xs, [LABEL[v] for v in LINEAGE], fontsize=8.5)
    ax.set_ylabel("score relative to B0 (geometric mean)")
    ax.set_ylim(0.9, max(vals) + 0.25)
    ax.axhline(1, color=AXIS, linewidth=1)
    ax.set_title("(a) Local: each build on 200 fresh seeds")
    ax.grid(axis="x", visible=False)

    subs = read_csv("submissions.csv")
    names = [r["build"].split(" ")[0] for r in subs]
    names[-1] = "final"
    prov = [float(r["provisional"]) for r in subs]
    xs2 = np.arange(len(subs))
    ax2.plot(xs2, prov, color=S1, linewidth=2, zorder=2)
    dots(ax2, xs2, prov, S1)
    for x, p in zip(xs2, prov):
        ax2.text(x, p + 0.55, "%.2f" % p, ha="center", fontsize=8.5, color=INK)
    ax2.set_xticks(xs2, names, fontsize=8.5)
    ax2.set_ylim(24, 41)
    ax2.set_ylabel("provisional score (0–100, 100 cases)")
    ax2.set_title("(b) Online: provisional reading of each submission")
    ax2.annotate("local +2.9%,\nonline −2.4%", (4.5, 36.4), xytext=(3.4, 30.3), fontsize=8.5, color=INK2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig3_lineage.png")


# ---------------------------------------------------------------- Figure 4
def fig_changes():
    rows = read_csv("changes.csv")
    rows = [r for r in rows if r["G"]]
    rows.sort(key=lambda r: (r["status"] == "rejected", -float(r["G"])))
    fig, ax = plt.subplots(figsize=(10.4, 7.6))
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        g = float(r["G"])
        color = S1 if r["status"] == "shipped" else S3 if r["status"].startswith("merged") else REST
        ax.barh(y, 100 * (g - 1), height=0.6, color=color, zorder=2)
        if r["ci_lo"]:
            ax.errorbar(100 * (g - 1), y, xerr=[[100 * (g - float(r["ci_lo"]))], [100 * (float(r["ci_hi"]) - g)]],
                        fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
        x_text = max(0.0, 100 * (max(g, float(r["ci_hi"]) if r["ci_hi"] else g) - 1)) + 0.6
        note = "  (superseded by P1-02)" if r["id"] == "R-wide" else ""
        ax.text(x_text, y, "%+.1f%%  n=%s%s" % (100 * (g - 1), r["n"], note), va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys, ["%s  %s" % (r["id"], r["change"][:62] + ("…" if len(r["change"]) > 62 else ""))
                       for r in rows], fontsize=8)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("measured effect on local score, % (geometric mean vs the stated reference)")
    ax.set_xlim(-6, 31)
    ax.set_title("Every substantive change, measured end to end")
    ax.legend(handles=[Patch(color=S1, label="shipped"), Patch(color=S3, label="shipped inside INT-01"),
                       Patch(color=REST, label="rejected")], loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig4_changes.png")


# ---------------------------------------------------------------- Figure 5
def fig_anatomy():
    per = read_csv("per_tile_final.csv")
    an = read_csv("anatomy_final.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))
    ax = axes[0]
    tiles = np.array([float(r["tiles"]) for r in per])
    pts = np.array([float(r["points"]) for r in per])
    labels = ["<5", "5–10", "10–20", "20–50", "50–100", "100–300", "≥300"]
    xs = np.arange(len(per))
    ax.bar(xs - 0.2, 100 * tiles / tiles.sum(), width=0.38, color=REST, label="share of tiles placed", zorder=2)
    ax.bar(xs + 0.2, 100 * pts / pts.sum(), width=0.38, color=S1, label="share of points", zorder=2)
    ax.set_xticks(xs, labels, fontsize=8.5)
    ax.set_xlabel("points per tile of the play")
    ax.set_ylabel("%")
    ax.legend(fontsize=8.5, loc="upper right")
    low = 100 * tiles[:2].sum() / tiles.sum()
    lowp = 100 * pts[:2].sum() / pts.sum()
    ax.set_title("(a) %d%% of tiles earn %d%% of points" % (round(low), round(lowp)))
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    q = np.array([[float(r["q%d" % k]) for k in range(1, 5)] for r in an])
    qt = np.array([[float(r["tiles_q%d" % k]) for k in range(1, 5)] for r in an])
    share_s = 100 * q.sum(0) / q.sum()
    share_t = 100 * qt.sum(0) / qt.sum()
    ppt = q.sum(0) / np.maximum(1, qt.sum(0))
    xs = np.arange(4)
    ax.bar(xs - 0.2, share_t, width=0.38, color=REST, label="share of tiles used", zorder=2)
    ax.bar(xs + 0.2, share_s, width=0.38, color=S1, label="share of points", zorder=2)
    for x, v, p in zip(xs, share_s, ppt):
        ax.text(x + 0.2, v + 1, "%.0f pt/tile" % p, ha="center", fontsize=8, color=INK2)
    ax.set_xticks(xs, ["1st", "2nd", "3rd", "4th"])
    ax.set_xlabel("quarter of the deck drawn")
    ax.set_ylabel("%")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("(b) Value collapses as the board fills")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    score = np.array([float(r["score"]) for r in an])
    tot = np.array([max(1.0, float(r["q1"]) + float(r["q2"]) + float(r["q3"]) + float(r["q4"])) for r in an])
    for key, color, name in (("top1", S2, "best play"), ("top3", S1, "best 3 plays")):
        v = np.sort(100 * np.array([float(r[key]) for r in an]) / tot)
        ax.plot(v, np.linspace(0, 100, len(v)), color=color, linewidth=2)
        med = np.median(v)
        ax.text(med + 2, 52 if key == "top1" else 40, "%s: median %d%%" % (name, round(med)), color=INK2, fontsize=8.8)
    ax.set_xlabel("share of the game's points")
    ax.set_ylabel("% of games (cumulative)")
    ax.set_xlim(0, 100)
    ax.set_title("(c) A few plays carry each game")
    fig.tight_layout()
    save(fig, "fig5_anatomy.png")
    return score


# ---------------------------------------------------------------- Figure 6
def fig_opportunity():
    an = read_csv("anatomy_final.csv")
    cells = read_csv("multiplier_cells_final.csv")
    rare = read_csv("rare_letters_final.csv")
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(14.6, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    best = np.array([float(r["board_best_product"]) for r in an])
    got = np.array([float(r["max_product"]) for r in an])
    rng = np.random.default_rng(1)
    jit = lambda v: v * np.exp(rng.normal(0, 0.06, len(v)))
    dots(ax, jit(best), jit(got), S1, size=26, alpha=0.85)
    lim = [1, max(best.max(), got.max()) * 1.6]
    ax.plot(lim, lim, color=AXIS, linewidth=1.2, zorder=1)
    ax.text(lim[1] / 1.6, lim[1] / 1.15, "achieved = available", fontsize=8.5, color=INK2, ha="right")
    ax.set_xscale("log", base=3)
    ax.set_yscale("log", base=3)
    ticks = [t for t in [1, 3, 9, 27, 81, 243, 729, 2187] if t <= lim[1]]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("best word-multiplier product over T consecutive cells")
    ax.set_ylabel("largest product of any play in the game")
    share = 100 * np.mean(got >= best)
    ax.set_title("(a) The best window is cashed (%d%% of games)" % round(share))

    prod = np.array([int(r["play_product"]) for r in cells])
    cnt = np.array([float(r["word_multiplier_cells"]) for r in cells])
    groups = [("×2–3", prod <= 3), ("×4–8", (prod >= 4) & (prod <= 8)),
              ("×9–26", (prod >= 9) & (prod <= 26)), ("≥×27", prod >= 27)]
    vals = [100 * cnt[m].sum() / cnt.sum() for _, m in groups]
    xs = np.arange(len(groups))
    ax2.bar(xs, vals, width=0.56, color=[REST, S1, S1, S1], zorder=2)
    for x, v in zip(xs, vals):
        ax2.text(x, v + 0.8, "%.0f%%" % v, ha="center", fontsize=9, color=INK)
    ax2.set_xticks(xs, [g for g, _ in groups])
    ax2.set_xlabel("product of the play that used the cell")
    ax2.set_ylabel("% of DW/TW cells used")
    ax2.set_title("(b) %.0f%% of DW/TW cells are spent alone" % vals[0])
    ax2.grid(axis="x", visible=False)

    order = ["discarded", "x1", "x2-8", "x9+"]
    names = {"discarded": "discarded", "x1": "×1 play", "x2-8": "×2–8 play", "x9+": "≥×9 play"}
    tot = sum(float(r["tiles"]) for r in rare)
    vals3 = [100 * float(next(r["tiles"] for r in rare if r["destination"] == k)) / tot for k in order]
    ax3.bar(xs, vals3, width=0.56, color=[REST, REST, S1, S1], zorder=2)
    for x, v in zip(xs, vals3):
        ax3.text(x, v + 0.8, "%.0f%%" % v, ha="center", fontsize=9, color=INK)
    ax3.set_xticks(xs, [names[k] for k in order])
    ax3.set_xlabel("where J, K, Q, X, Z tiles ended up")
    ax3.set_ylabel("% of those tiles")
    ax3.set_title("(c) Only %.0f%% of J/K/Q/X/Z meet a ≥×9 play" % vals3[-1])
    ax3.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig6_opportunity.png")


# ---------------------------------------------------------------- Figure 7
def fig_budget():
    curve = read_csv("budget_curve.csv")
    lin = read_csv("lineage.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4))
    p = [r for r in curve if r["build"] == "P2-03"]
    x = [float(r["budget_ms"]) for r in p]
    y = [float(r["G"]) for r in p]
    ax.plot(x, y, color=S1, linewidth=2)
    dots(ax, x, y, S1)
    ax.axvline(800, color=AXIS, linestyle="--", linewidth=1)
    ax.text(850, 1.02, "shipped\n800 ms", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([200, 400, 800, 1500, 3000, 7500], ["200", "400", "800", "1500", "3000", "7500"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("effort budget per game (reference-machine ms, log)")
    ax.set_ylabel("score vs P1-02 (geometric mean)")
    ax.set_title("(a) Two-ply search saturates at 1–1.5 s")

    # Work-budget probes never engaged the wall-clock guard, so the parallel run
    # is exact for them; O-01C and the wall-clock probes come from the serial rerun.
    seeds = read_csv("lineage_seeds.csv")
    serial = {r["seed"]: r for r in read_csv("serial_probes.csv")}
    lin_rt = {r["version"]: float(r["runtime_p50_ms"]) for r in lin}
    col = {"O-01C": "O-01C", "O-01C-B5000": "O-01C-B5000", "O-01C-B3000": "O-01C-B3000"}
    base = [float(serial[r["seed"]]["O-01C"]) for r in seeds]

    def score_of(v):
        if v in col:
            return [float(serial[r["seed"]][col[v]]) for r in seeds]
        return [float(r[v]) for r in seeds]

    probes = [("O-01C-W700", "work 700"), ("O-01C-W1000", "work 1000"), ("O-01C-W1500", "work 1500"),
              ("O-01C-W2000", "work 2000"), ("O-01C", "O-01C"), ("O-01C-B3000", "wall 3.0 s"),
              ("O-01C-B5000", "wall 5.0 s")]
    rt = {v: (statistics.median(float(serial[r["seed"]][v + "_ms"]) for r in seeds) if v in col else lin_rt[v])
          for v, _ in probes}
    xs = np.arange(len(probes))
    vals, lo, hi = [], [], []
    for v, _ in probes:
        g, a, b = geo_ci(score_of(v), base)
        vals.append(g)
        lo.append(g - a)
        hi.append(b - g)
    ax2.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1.1, capsize=3, zorder=2)
    dots(ax2, xs, vals, S1, size=52)
    ax2.axhline(1, color=AXIS, linewidth=1)
    ax2.set_xticks(xs, ["%s\np50 %.1f s" % (lbl, rt[name] / 1000) for name, lbl in probes], fontsize=8)
    ax2.set_ylim(0.86, 1.04)
    ax2.set_ylabel("score vs O-01C (geometric mean)")
    ax2.set_title("(b) Every time probe of the final submission costs score")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig7_budget.png")


# ---------------------------------------------------------------- Figure 8
def fig_effort():
    rows = read_csv("effort_final.csv")
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for (lo_n, hi_n), color, name in (((10, 14), S1, "N = 10–14"), ((15, 20), S2, "N = 15–20")):
        sel = [r for r in rows if lo_n <= int(r["N"]) <= hi_n]
        games = len({r["seed"] for r in sel})
        per = defaultdict(float)
        for r in sel:
            k = min(9, int(float(r["progress"]) * 10))
            per[k] += float(r["work_ms"])
        y = [per[k] / games for k in range(10)]
        ax.plot((bins[:-1] + bins[1:]) / 2 * 100, y, color=color, linewidth=2)
        dots(ax, (bins[:-1] + bins[1:]) / 2 * 100, y, color, size=30)
        if lo_n > 10:
            ax.text(47, y[4] + 35, name, color=INK2, fontsize=9)
        else:
            ax.text(26, y[2] - 100, name, color=INK2, fontsize=9)
    ax.set_xlabel("% of the deck drawn")
    ax.set_ylabel("effort spent per game (ms, per 10% of deck)")
    ax.set_title("Where the effort goes: front-loaded by design")
    fig.tight_layout()
    save(fig, "fig8_effort.png")


# ---------------------------------------------------------------- Figure 9
def fig_leaderboard():
    rows = [r for r in read_csv("leaderboard.csv") if r["final_score"] not in ("", "N/A")]
    rows = [r for r in rows if float(r["final_score"]) > 0]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ranks = [int(r["final_rank"]) for r in rows]
    fin = [float(r["final_score"]) for r in rows]
    colors = [S2 if r["handle"] == "wash" else S1 if int(r["final_rank"]) <= 5 else REST for r in rows]
    ax.bar(ranks, fin, width=0.8, color=colors, zorder=2)
    ax.text(48, 86, "\n".join("%s. %s  %.2f" % (r["final_rank"], r["handle"], float(r["final_score"]))
                               for r in rows[:5]), fontsize=8.3, color=INK2, va="top", linespacing=1.4)
    me = next(r for r in rows if r["handle"] == "wash")
    ax.annotate("this solver: %.2f, rank %s" % (float(me["final_score"]), me["final_rank"]),
                (int(me["final_rank"]), float(me["final_score"])), xytext=(46, 52), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_xlabel("final rank")
    ax.set_ylabel("final score (5,000 cases)")
    ax.set_title("(a) Final standings: a top-5 cluster, then a gap")
    ax.grid(axis="x", visible=False)

    prov = [(float(r["provisional_score"]), float(r["final_score"]), r["handle"]) for r in rows if r["provisional_score"]]
    dots(ax2, [p for p, f, h in prov if h != "wash"], [f - p for p, f, h in prov if h != "wash"], REST, size=34)
    me_p = next(p for p in prov if p[2] == "wash")
    dots(ax2, [me_p[0]], [me_p[1] - me_p[0]], S2, size=60)
    ax2.text(me_p[0] + 1.5, me_p[1] - me_p[0] + 0.25, "this solver", fontsize=9, color=INK)
    ax2.axhline(0, color=AXIS, linewidth=1)
    ax2.set_xlabel("provisional score (100 cases)")
    ax2.set_ylabel("final − provisional")
    ax2.set_title("(b) Almost every score rose in final testing")
    fig.tight_layout()
    save(fig, "fig9_leaderboard.png")


# ---------------------------------------------------------------- Figure 10
def fig_parameters():
    rows = read_csv("final5000.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4))
    edges = np.linspace(0.1, 0.5, 6)
    mids = (edges[:-1] + edges[1:]) / 2
    for (lo_n, hi_n), color, name in (((10, 13), S1, "N = 10–13"), ((14, 17), S2, "N = 14–17"),
                                      ((18, 20), S3, "N = 18–20")):
        ys = []
        for a, b in zip(edges[:-1], edges[1:]):
            v = [float(r["O01C"]) for r in rows if lo_n <= int(r["N"]) <= hi_n and a <= float(r["B"]) < b + 1e-9]
            ys.append(statistics.median(v))
        ax.plot(mids, ys, color=color, linewidth=2)
        dots(ax, mids, ys, color, size=32)
        ax.text(mids[-1] + 0.008, ys[-1], name, color=INK2, fontsize=9, va="center")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 0.56)
    ax.set_xlabel("bonus ratio B (hidden generator parameter)")
    ax.set_ylabel("median final score (log)")
    ax.set_title("(a) Score grows with multiplier density")

    xs = np.arange(6)
    vals, lo, hi = [], [], []
    for x in range(6):
        sel = [r for r in rows if int(r["X"]) == x]
        g, a, b = geo_ci([float(r["O01C"]) for r in sel], [float(r["INT01"]) for r in sel], reps=1000)
        vals.append(g)
        lo.append(g - a)
        hi.append(b - g)
    ax2.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1.1, capsize=3, zorder=2)
    dots(ax2, xs, vals, S1, size=52)
    ax2.axhline(1, color=AXIS, linewidth=1)
    ax2.set_xticks(xs, [str(x) for x in xs])
    ax2.set_xlabel("lookahead tiles X")
    ax2.set_ylabel("O-01C vs INT-01 (geometric mean)")
    ax2.set_title("(b) Fishing pays more when refills are visible")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig10_parameters.png")


# ---------------------------------------------------------------- Figure 11
def fig_redesign():
    fig = plt.figure(figsize=(13.2, 6.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.97, "A plan-centric redesign: targets as first-class objects", fontsize=13,
            fontweight="bold", color=INK, va="top")
    ax.text(0.02, 0.93, "The unit of decision becomes a jackpot target (lane + word family), not a single move. "
            "Moves are judged by what they do to the target portfolio and to the board's remaining multipliers.",
            fontsize=9, color=INK2, va="top")
    w, h = 0.21, 0.25
    top = [(0.02, "Target enumerator", "Every line segment through ≥ 2 word\nmultipliers that can be anchored; word\nfamilies that fit, with exact score\n(letter mults + cross words)."),
           (0.27, "Completion model", "P(rack + refills complete word within\nk turns | keep plan), exact draw DP over\nletter frequencies + visible preview;\ncost in tiles and discard penalties."),
           (0.52, "Portfolio planner", "Choose ≤ 2 active targets maximising\nexpected points per tile of the deck left;\nreserve their cells (no burning);\nplan the setup moves that anchor them."),
           (0.77, "Move selector", "Setup / fish (keep + discard) / cash-in /\nfiller. Filler moves must not burn\nreserved multipliers and should create\nanchors. Rollouts only to break ties.")]
    for i, (x, t, b) in enumerate(top):
        box(ax, x, 0.55, w, h, t, b, edge=S1)
        if i:
            arrow(ax, (top[i - 1][0] + w + 0.003, 0.55 + h / 2), (x - 0.004, 0.55 + h / 2), color=INK2)
    bottom = [(0.02, "Fast in-process simulator", "Game + tile sampler in C++, millions of\nself-play games: the training and\nevaluation loop from day one."),
              (0.27, "Learned values", "Leave values per rack multiset, shadow\nprices per multiplier cell, target\nhit-rate tables; fitted by self-play."),
              (0.52, "Full-time budget", "Spend ~9 s: Monte Carlo over sampled\ndecks for plan choice (when to cash,\nwhich target to abandon)."),
              (0.77, "Ceiling experiments", "Perfect-information oracle on local\nseeds bounds what planning can gain\nbefore building it.")]
    for x, t, b in bottom:
        box(ax, x, 0.17, w, 0.24, t, b, edge=S2)
    for x, _, _ in bottom:
        arrow(ax, (x + w / 2, 0.41), (x + w / 2, 0.548), color=S2, ls="--")
    ax.text(0.02, 0.10, "Blue: per-turn decision path.  Orange: infrastructure that feeds it. Inferred from our "
            "measurements and the score's structure, not from other competitors' code.", fontsize=8.5, color=INK2)
    save(fig, "fig11_redesign.png")


FIGURES = [fig_board, fig_architecture, fig_lineage, fig_changes, fig_anatomy, fig_opportunity, fig_budget,
           fig_effort, fig_leaderboard, fig_parameters, fig_redesign]

if __name__ == "__main__":
    import sys
    wanted = set(sys.argv[1:])
    for f in FIGURES:
        if not wanted or f.__name__ in wanted:
            f()
