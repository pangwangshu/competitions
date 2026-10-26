"""Exploratory analysis of the competition metadata CSVs (no DICOMs needed).

Writes a text summary to report/eda_summary.txt and figures to report/figures/.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsna_knee.constants import LABELS, PLANES, RAW_DIR, REPORTS_DIR

FIG_DIR = REPORTS_DIR / "figures"


def detect_lang_guess(report: str) -> str:
    """Cheap heuristic: non-ASCII + a few marker words per language."""
    text = report.lower()
    markers = {
        "es": ["rodilla", "menisco", "derrame", "ruptura", "técnica"],
        "de": [" knei", "meniskus", "erguss", "bande", "kontusion", "fraktur"],
        "fr": ["genou", "ménisque", "épanchement", "rupture", "osseuse"],
        "pt": ["joelho", "menisco", "derrame", "fratura"],
        "it": ["ginocchio", "menisco", "versamento", "frattura"],
    }
    for lang, words in markers.items():
        if any(w in text for w in words):
            return lang
    return "en/other"


def main() -> None:
    train = pd.read_csv(RAW_DIR / "train.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(RAW_DIR / "train_series.csv")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=== STUDIES ===")
    log(f"studies: {len(train)}   series: {len(series)}   "
        f"series/study: {len(series) / len(train):.2f}")

    log("\n=== LABELS ===")
    n_labeled = int(train[LABELS].notna().to_numpy().any(axis=1).sum())
    log(f"studies with gold labels: {n_labeled} / {len(train)} "
        f"(rest are fully UNLABELED — blank != negative!)")
    gold = train[train[LABELS].notna().to_numpy().any(axis=1)]
    label_stats = []
    for col in LABELS:
        pos = int(gold[col].sum())
        label_stats.append(
            {"label": col, "positives": pos, "prevalence_in_gold": pos / len(gold)}
        )
    stats = pd.DataFrame(label_stats).sort_values("prevalence_in_gold", ascending=False)
    log(stats.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    n_pos = gold[LABELS].sum(axis=1)
    log("\npositives per gold study:")
    log(n_pos.value_counts().sort_index().to_string())
    log(f"gold studies with >=1 positive: {(n_pos > 0).sum()}/{len(gold)}")

    log("\n=== REPORTS ===")
    log(f"studies with report text: {train['Report'].notna().mean():.1%}")
    log(f"report length (chars): median {train['Report'].str.len().median():.0f}, "
        f"p95 {train['Report'].str.len().quantile(0.95):.0f}")
    lang = train["Report"].fillna("").map(detect_lang_guess)
    log("\nlanguage guess:")
    log(lang.value_counts().to_string())

    log("\n=== SERIES ===")
    log("planes:")
    log(series["Anatomical_Plane"].value_counts(dropna=False).to_string())
    log("fluid-sensitive / fat-suppression combos:")
    combo = series.groupby(["Fluid_Sensitive", "Fat_Suppression"]).size()
    log(combo.to_string())
    per_study = series.groupby("StudyInstanceUID").size()
    log(f"\nseries per study: min {per_study.min()}, median {per_study.median():.0f}, "
        f"max {per_study.max()}")
    log("\nplane combos per study (top 8):")
    combo_per_study = (
        series.groupby("StudyInstanceUID")["Anatomical_Plane"]
        .apply(lambda s: "+".join(sorted(s)))
    )
    log(combo_per_study.value_counts().head(8).to_string())

    # ---- figures ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    stats.set_index("label")["prevalence_in_gold"].sort_values().plot.barh(
        ax=axes[0], title="Label prevalence (58 gold studies)"
    )
    n_pos.value_counts().sort_index().plot.bar(
        ax=axes[1], title="Positives per gold study"
    )
    lang.value_counts().plot.bar(ax=axes[2], title="Report language (guess)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_overview.png", dpi=150)
    print(f"\nfigure saved: {FIG_DIR / 'eda_overview.png'}")

    out = REPORTS_DIR / "eda_summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"summary saved: {out}")


if __name__ == "__main__":
    main()
