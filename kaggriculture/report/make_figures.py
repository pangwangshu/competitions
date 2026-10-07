"""Regenerate the paper's figures from the small CSVs in report/data/.

    python report/make_figures.py

Needs matplotlib + numpy only. Every number plotted lives in report/data/,
so the figures stay reproducible after the raw analysis artifacts are gone.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")

# Reference palette (light surface). Text never wears a series color.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#f0efec"
S1 = "#2a78d6"  # blue
S2 = "#eb6834"  # orange
S3 = "#1baf7a"  # aqua (sub-3:1 contrast: always direct-labelled)

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
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", os.path.relpath(path))


def dots(ax, x, y, color, size=46, marker="o", zorder=3, **kw):
    """>=8px marker with a 2px surface ring."""
    return ax.scatter(x, y, s=size, color=color, edgecolors=SURFACE,
                      linewidths=1.6, marker=marker, zorder=zorder, **kw)


# ---------------------------------------------------------------- Figure 1
def fig_rating():
    rows = read_csv("submissions.csv")
    x = np.arange(1, len(rows) + 1)
    y = np.array([float(r["public_score"]) for r in rows])
    fig, ax = plt.subplots(figsize=(9.2, 4.2))

    phases = []
    for i, r in enumerate(rows):
        if not phases or phases[-1][0] != r["phase"]:
            phases.append([r["phase"], i + 1, i + 1])
        phases[-1][2] = i + 1
    names = {"I rules": "I  Heuristic rules",
             "II hierarchical RL": "II  Hierarchical RL",
             "III execution": "III",
             "IV instrumented": "IV  Instrumented mechanism search"}
    for k, (ph, a, b) in enumerate(phases):
        if k % 2 == 0:
            ax.axvspan(a - 0.5, b + 0.5, color=BAND, zorder=0, lw=0)
        ax.text((a + b) / 2, 822, names[ph], ha="center", va="bottom",
                color=INK2, fontsize=8.5)

    ax.axhline(766.7, color=MUTED, lw=1, zorder=1)
    ax.text(len(rows) + 0.6, 766.7, "Leaderboard\nmedian 767", va="center",
            ha="left", color=INK2, fontsize=8.5)
    ax.plot(x, y, color=S1, lw=1, alpha=0.35, zorder=2)
    dots(ax, x, y, S1)

    ax.set_xlim(0.3, len(rows) + 0.7)
    ax.set_ylim(250, 850)
    ax.set_xticks(x)
    short = {"v13 (resubmit)": "v13r", "v14a (opening)": "v14a", "v18a (combo)": "v18a"}
    ax.set_xticklabels([short.get(r["label"], r["label"]) for r in rows], rotation=90,
                       fontsize=7.5)
    ax.set_ylabel("Kaggle rating (as of 2026-10-03)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Rating of every submission, in submission order", pad=22)
    save(fig, "fig2_rating_trajectory.png")


# ---------------------------------------------------------------- Figure 3
def fig_dispatch():
    rows = read_csv("dispatcher_sweep.csv")
    order = ["1", "2", "4", "8", "16", "32", "64", "pure_distance"]
    tick = ["1", "2", "4", "8", "16", "32", "64", "distance\nonly"]
    tun = {r["weight"]: r for r in rows if r["split"] == "tuning"}
    held = {r["weight"]: r for r in rows if r["split"] == "heldout"}
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    xs = np.arange(len(order))
    ty = [float(tun[w]["paired_margin"]) for w in order]
    ax.plot(xs[:-1], ty[:-1], color=S1, lw=2, solid_capstyle="round", zorder=2)
    dots(ax, xs, ty, S1, label="Tuning seeds (n=320 per point)")
    hx = [order.index(w) for w in held]
    hy = [float(held[w]["paired_margin"]) for w in held]
    dots(ax, hx, hy, S2, size=70, marker="D", zorder=4,
         label="Held-out seeds (n=480)")
    for w in held:
        i = order.index(w)
        v = float(held[w]["paired_margin"])
        ax.annotate(f"${v:,.0f}\nt = {float(held[w]['t']):.1f}", (i, v),
                    xytext=(12, -4), textcoords="offset points", va="center",
                    color=INK, fontsize=8.5)
    ax.annotate("w = 1: priorities never inverted\n+$758, t = 0.9", (0, ty[0]),
                xytext=(-8, 62), textcoords="offset points", va="bottom",
                color=INK2, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                shrinkA=2, shrinkB=5))
    ax.set_xticks(xs)
    ax.set_xticklabels(tick)
    ax.set_xlabel("Distance weight w in matching cost  w·travel − priority")
    ax.set_ylabel("Paired final margin vs old dispatcher ($/game)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_ylim(0, 12000)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Dispatcher: how far may a unit walk to serve a higher priority?")
    save(fig, "fig3_dispatch_weight.png")


# ---------------------------------------------------------------- Figure 4
def fig_resolution():
    # Paired per-game sd of Δmargin measured on the same 151-tape panel (R5)
    # and on the pinned-randomness paired panel. Games needed so that the 95% CI
    # half-width equals the effect: n = (1.96 sd / effect)^2.
    series = [
        ("Late-season change, tape panel (sd $1.7k)", 1739, S1),
        ("Early change, pinned paired panel (sd $6.6k)", 6600, S3),
        ("Early change, tape panel (sd $23.3k)", 23309, S2),
    ]
    eff = np.geomspace(250, 20000, 200)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for name, sd, c in series:
        n = (1.96 * sd / eff) ** 2
        ax.plot(eff, n, color=c, lw=2, label=name, solid_capstyle="round")
    ax.axhline(151, color=MUTED, lw=1)
    ax.text(21500, 151, "151 real-ladder\ntapes available", va="center",
            ha="left", color=INK2, fontsize=8.5)
    for e, sd, c in [(3000, 23309, S2), (1000, 1739, S1)]:
        n = (1.96 * sd / e) ** 2
        dots(ax, [e], [n], c, zorder=4)
        ax.annotate(f"{n:,.0f} games", (e, n), xytext=(8, 6),
                    textcoords="offset points", color=INK, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(250, 20000)
    ax.set_ylim(0.5, 1e5)
    ax.set_xticks([250, 500, 1000, 2000, 5000, 10000, 20000])
    ax.set_xticklabels(["$250", "$500", "$1k", "$2k", "$5k", "$10k", "$20k"])
    ax.set_yticks([1, 10, 100, 1000, 10000, 100000])
    ax.set_yticklabels(["1", "10", "100", "1,000", "10,000", "100,000"])
    ax.set_xlabel("True effect on final margin ($/game)")
    ax.set_ylabel("Paired games to resolve it (95% CI)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("Why early-game changes were unmeasurable on a fixed panel")
    save(fig, "fig4_measurement_resolution.png")


# ---------------------------------------------------------------- Figure 5
def fig_flippable():
    games = read_csv("online_games_v24_v25.csv")
    losses = np.sort([-float(g["margin"]) for g in games if g["res"] == "L"])
    n = len(losses)
    xs = np.linspace(0, 60000, 601)
    ys = [100.0 * np.mean(losses < x) for x in xs]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(xs, ys, color=S1, lw=2, solid_capstyle="round")
    ax.fill_between(xs, ys, color=S1, alpha=0.10, lw=0)
    for x in (3000, 10000, 20000):
        yv = 100.0 * np.mean(losses < x)
        dots(ax, [x], [yv], S1, zorder=4)
        ax.annotate(f"+${x // 1000}k flips {yv:.0f}%", (x, yv),
                    xytext=(26, -3) if x == 3000 else (8, -12), va="center" if x == 3000 else "baseline",
                    textcoords="offset points", color=INK, fontsize=8.5)
    ax.axvspan(500, 4000, color=BAND, zorder=0, lw=0)
    ax.text(2250, 92, "typical size of\none adopted change", ha="center",
            va="top", color=INK2, fontsize=8.5)
    ax.set_xlim(0, 60000)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"+${v / 1000:.0f}k"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Uniform improvement in my final money ($/game)")
    ax.set_ylabel("Share of online losses overturned")
    ax.set_title(f"Online losses of the final pair (n = {n}) were mostly blowouts")
    save(fig, "fig6_flippable_losses.png")


# ---------------------------------------------------------------- Figure 6
def fig_effects():
    rows = [r for r in read_csv("adopted_effects.csv") if r["version"] != "v11"]
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ys = np.arange(len(rows))
    for y, r in zip(ys, rows):
        lo, hi, d = float(r["ci_lo"]), float(r["ci_hi"]), float(r["delta"])
        ax.plot([lo, hi], [y, y], color=S1, lw=2, solid_capstyle="round", zorder=2)
        dots(ax, [d], [y], S1, zorder=3)
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['version']}  {r['change']}  ({r['panel'].lower()})"
                        for r in rows], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-500, 10000)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:+,.0f}" if v else "0"))
    ax.set_xlabel("Paired Δ final margin vs predecessor ($/game, 95% CI)")
    ax.set_title("Every adopted change, measured offline")
    save(fig, "fig5_adopted_effects.png")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig_rating()
    fig_dispatch()
    fig_resolution()
    fig_flippable()
    fig_effects()
