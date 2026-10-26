"""Write-up figures 2-5, rendered from the small CSVs in report/data/.

  uv run python report/make_figures.py

Each CSV holds exactly the numbers a figure plots, with a `source` or `source_key` column naming the experiment record it
was taken from (the registry of those records is experiments/results.csv). Figure 1 (eda_overview.png) is written by
scripts/eda.py from the competition metadata, which is not redistributed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "report" / "data"
FIGS = ROOT / "report" / "figures"

# Reference palette (light mode): categorical slots 1-3, recessive ink and grid.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": INK_2, "ytick.color": INK_2,
    "font.size": 9.5, "legend.frameon": False, "axes.axisbelow": True,
})

# S5 comparisons on fold 0: label, analysis key, what it tests.
ABLATIONS = [
    ("Mirror aug., 14 epochs (T1)", "T1-14ep vs R0-14ep"),
    ("Mirror aug., 10 epochs", "T1-EP10 vs EP10"),
    ("10 vs 14 epochs", "EP10 vs R0-14ep"),
    ("ConvNeXt-Small vs Tiny", "small-EP10 vs EP10"),
    ("Training resolution 288", "res288-EP10 vs EP10"),
    ("2 Tiny + 2 Small vs 4 Tiny", "2 tiny EP10 + 2 small vs 4 tiny"),
    ("2 Tiny + 2 res288 vs 4 Tiny", "2 tiny EP10 + 2 res288 vs 4 tiny"),
]
# Public reader-only probes: (members, label, results.csv id).
PUBLIC = [
    (3, "author ckpts", "S3-probe"),
    (3, "own Tiny", "S6-probe-tiny3"),
    (6, "own Tiny + Small", "S6-probe-tiny3-small3"),
    (9, "author + own", "S6-probe-author3-tiny3-small3"),
    (10, "C* (5 folds)", "S6-final-own"),
]
# T4 distillation on fold 0 under I*2: arm label, per-seed eval dirs, matched leaderboard probe.
T4_ARMS = [
    ("v0 labels (control)", ["S5-T1-istar2", "S5-T1-seed1-istar2"], "S5-probe-E-t1x2"),
    ("distilled labels_t4s", ["S5-T4S6-istar2", "S5-T4S6-seed1-istar2"], "S5-probe-F-t4s6x2"),
]


def ablation_data():
    return pd.read_csv(DATA / "s5_ablations.csv")


def fig_ablations(df):
    metrics = df.metric.unique()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    labels = [a for a, _ in ABLATIONS]
    y = range(len(labels))[::-1]
    for ax, metric in zip(axes, metrics):
        d = df[df.metric == metric].set_index("comparison").reindex(labels)
        ax.axvline(0, color=INK_2, lw=1)
        for yi, (_, r) in zip(y, d.iterrows()):
            if pd.isna(r.delta):
                ax.annotate("not computed", (0, yi), xytext=(6, 0), textcoords="offset points", va="center",
                            color=INK_2, fontsize=8)
                continue
            clear = r.ci_low > 0 or r.ci_high < 0
            ax.plot([r.ci_low, r.ci_high], [yi, yi], color=BLUE, lw=2, solid_capstyle="round")
            ax.plot(r.delta, yi, "o", ms=8 if clear else 6.5, mfc=BLUE if clear else SURFACE, mec=BLUE, mew=2)
        ax.set_yticks(list(y), labels)
        ax.set_title(metric)
        ax.set_xlabel("Δ macro AUC, 2-seed ensembles (95% paired bootstrap)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    fig.text(0.01, 0.01, "Filled dot: interval excludes 0. Fold 0, ConvNeXt reader, inference I*2 not applied "
             "(in-training evaluation).", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def ensemble_data():
    return pd.read_csv(DATA / "ensemble_offline_fold0.csv"), pd.read_csv(DATA / "ensemble_public.csv")


def fig_ensemble(offline, public):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for col, color, name in (("oof", BLUE, "OOF"), ("gold", ORANGE, "Gold58")):
        a.fill_between(offline.members, offline[col] - offline[col + "_sd"], offline[col] + offline[col + "_sd"],
                       color=color, alpha=0.15, lw=0)
        a.plot(offline.members, offline[col], "-o", color=color, lw=2, ms=7, mec=SURFACE, mew=2, label=name)
        last = offline.iloc[-1]
        a.annotate(name, (last.members, last[col]), xytext=(6, 0), textcoords="offset points", va="center",
                   color=INK_2)
    a.set_title("Fold 0, offline: random subsets of 9 models")
    a.set_xlabel("models in the ensemble")
    a.set_ylabel("macro AUC")
    a.set_xticks(offline.members)
    a.legend(loc="lower right")
    own = public[public.label != "author ckpts"]
    author = public[public.label == "author ckpts"].iloc[0]
    for col, color, name in (("public", BLUE, "public (30%)"), ("private", ORANGE, "private (70%)")):
        b.plot(own.members, own[col], "-o", color=color, lw=2, ms=7, mec=SURFACE, mew=2, label=f"own models, {name}")
        b.plot(author.members, author[col], "s", color=color, ms=7, mfc=SURFACE, mew=2)
        for _, r in own.iterrows():
            b.annotate(f"{r[col]:.3f}", (r.members, r[col]), xytext=(0, 7), ha="center", textcoords="offset points",
                       color=INK_2, fontsize=8)
    b.annotate("open square: author ckpts x3", (author.members, author.public), xytext=(10, 0), va="center",
               textcoords="offset points", color=INK_2, fontsize=8)
    b.set_title("Leaderboard: own readers, I*2 inference")
    b.set_xlabel("models in the ensemble")
    b.set_ylabel("macro AUC (3 decimals)")
    b.set_xticks([3, 6, 9, 10])
    b.set_xlim(2, 11)
    b.legend(loc="center right")
    fig.tight_layout()
    return fig


def t4_data():
    df = pd.read_csv(DATA / "t4_distillation.csv")
    df["seed"] = df.seed.fillna(-1).astype(int)  # -1: a leaderboard probe, which has no per-seed value
    return df


def fig_t4(df):
    metrics = ["OOF fold 0", "Gold58", "Public", "Private"]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.0))
    control, student = (a for a, _, _ in T4_ARMS)
    for ax, metric in zip(axes, metrics):
        d = df[df.metric == metric]
        for seed in d.seed.unique():
            pair = d[d.seed == seed].set_index("arm").auc
            ax.plot([0, 1], [pair[control], pair[student]], "-o", color=BLUE, lw=1.5, ms=7, mec=SURFACE, mew=2)
            for x, arm in ((0, control), (1, student)):
                ax.annotate(f"{pair[arm]:.4f}" if metric in metrics[:2] else f"{pair[arm]:.3f}", (x, pair[arm]),
                            xytext=(-8 if x == 0 else 8, 0), ha="right" if x == 0 else "left", va="center",
                            textcoords="offset points", color=INK_2, fontsize=8)
        ax.set_title(metric)
        ax.set_xticks([0, 1], ["v0", "distilled"])
        ax.set_xlim(-0.8, 1.8)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("macro AUC")
    fig.text(0.01, 0.01, "Fold-0 ConvNeXt-Tiny, mirror aug., 14 epochs, I*2. OOF/Gold58: one line per seed. "
             "Public/private: 2-seed probes E and F.", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def gpu_data():
    return pd.read_csv(DATA / "gpu_hours_by_stage.csv")


def fig_gpu(df):
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    d = df[df.gpu_hours > 0].iloc[::-1]
    ax.barh(d.stage, d.gpu_hours, color=BLUE, height=0.6)
    for s, h in zip(d.stage, d.gpu_hours):
        ax.annotate(f"{h:.1f} h", (h, s), xytext=(4, 0), textcoords="offset points", va="center", color=INK_2)
    ax.set_title(f"Kaggle GPU-quota hours by stage (total {df.gpu_hours.sum():.1f} h)")
    ax.set_xlabel("quota hours, a 2 x T4 session charged once (summed from experiments/results.csv)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    offline, public = ensemble_data()
    abl, gpu, t4 = ablation_data(), gpu_data(), t4_data()
    for name, fig in (("s5_ablations", fig_ablations(abl)), ("ensemble_curve", fig_ensemble(offline, public)),
                      ("gpu_hours", fig_gpu(gpu)), ("t4_distillation", fig_t4(t4))):
        fig.savefig(FIGS / f"{name}.png", dpi=160)
        plt.close(fig)
        print("wrote", FIGS / f"{name}.png")


if __name__ == "__main__":
    main()
