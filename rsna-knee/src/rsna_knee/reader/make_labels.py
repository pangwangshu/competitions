"""Build the training label table: soft targets from report-derived label sources + CV folds.

Source: goodpjw2008/rsna-knee-2-5d-convnext-reader.

Usage: python -m rsna_knee.reader.make_labels <version>
  v0 = mean of four public LLM label tables
  v1 = v0 sources + our own graded LLM extraction (needs llm_grades_*.jsonl)
Gold studies (official labels) keep fold = -1 and are never used for training in CV.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .paths import DATA, EXT, WORK
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
KEYS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa",
        "pf_oa", "effusion", "synovitis", "bakers_cyst", "contusion", "fracture"]

# grade -> probability that an image annotator calls the finding positive.
# Index 0 is "not addressed" (-1), then grades 0..3. Annotators only counted
# high-grade ligament injury, moderate/large effusion and Baker's cyst, and >50% cartilage loss.
GRADE_P = {
    "ACL":              [0.10, 0.03, 0.20, 0.70, 0.97],
    "MCL":              [0.08, 0.03, 0.20, 0.70, 0.95],
    "Medial Meniscus":  [0.15, 0.04, 0.25, 0.92, 0.97],
    "Lateral Meniscus": [0.12, 0.04, 0.25, 0.92, 0.97],
    "Medial OA":        [0.06, 0.03, 0.20, 0.75, 0.95],
    "Lateral OA":       [0.05, 0.03, 0.20, 0.75, 0.95],
    "PF OA":            [0.10, 0.03, 0.25, 0.75, 0.95],
    "Effusion":         [0.25, 0.05, 0.35, 0.85, 0.97],
    "Synovitis":        [0.30, 0.10, 0.55, 0.85, 0.95],
    "Baker's":          [0.05, 0.02, 0.30, 0.80, 0.95],
    "Contusion":        [0.15, 0.05, 0.40, 0.90, 0.95],
    "Fracture":         [0.12, 0.05, 0.40, 0.90, 0.97],
}


def public_sources(ids):
    def read(owner, slug, filename):
        # Support both current and legacy Kaggle mounts and the author's local layout.
        candidates = [Path(EXT) / f"{owner}_{slug}" / filename,
                      Path(EXT) / "datasets" / owner / slug / filename,
                      Path(EXT) / slug / filename]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing {owner}/{slug}/{filename}; tried {candidates}")
        return pd.read_csv(path)

    src = {}
    src["steven"] = read("stevenleehans", "rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")
    src["pilkwang"] = read("pilkwang", "rsna-knee-llm-labels", "report_labels_v2.csv")
    src["lixin"] = read("lixin73", "rsna-knee-llm-report-labels-sol56", "labels_llm_gpt56sol.csv")
    q = read("laymond", "rsna-knee-abnormality-qwen3-8b-weak-labels", "qwen_knee_weak_labels.csv")
    src["qwen"] = q[["StudyInstanceUID"] + [f"{c}__label" for c in LABELS]].set_axis(["StudyInstanceUID"] + LABELS, axis=1)
    return {k: v.set_index("StudyInstanceUID")[LABELS].astype(float).reindex(ids) for k, v in src.items()}


def own_grades(ids, path):
    rows = [json.loads(line) for line in open(path)]
    g = pd.DataFrame(rows).set_index("study")[KEYS].set_axis(LABELS, axis=1).reindex(ids)
    p = pd.DataFrame({c: g[c].map(lambda v: np.nan if pd.isna(v) else GRADE_P[c][int(v) + 1]) for c in LABELS}, index=ids)
    # a silent report on synovitis is more likely positive when a sizeable effusion is described
    eff = g["Effusion"]
    silent = g["Synovitis"] == -1
    p.loc[silent & (eff == 2), "Synovitis"] = 0.45
    p.loc[silent & (eff == 3), "Synovitis"] = 0.60
    p.loc[silent & (eff == 0), "Synovitis"] = 0.15
    return g, p


def report_group(text):
    return re.sub(r"\W+", "", str(text).lower())


def stable_group_folds(groups, n_splits=5):
    """GroupKFold(n_splits) fold per sample, with group-size ties broken stably.

    Same greedy rule as scikit-learn (largest group to the lightest fold, groups
    in np.unique order). Nearly every report group has one study, so the folds
    hinge on tie order: older scikit-learn used numpy's default argsort, whose
    order differs between CPUs, while newer versions sort stably. The author's
    checkpoints were trained on the stable folds (S2 fold check), so the stable
    rule is written out here instead of depending on the installed version.
    """
    unique, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)
    load, fold_of = np.zeros(n_splits), np.zeros(len(unique), int)
    for g in np.argsort(sizes, kind="stable")[::-1]:
        fold_of[g] = np.argmin(load)
        load[fold_of[g]] += sizes[g]
    return fold_of[inverse]


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "v0"
    tr = pd.read_csv(f"{DATA}/train.csv")
    ids = tr.StudyInstanceUID
    gold = tr.set_index("StudyInstanceUID")[LABELS]
    is_gold = gold.notna().all(axis=1)

    src = public_sources(ids)
    if version != "v0":
        _, src["own"] = own_grades(ids, sys.argv[2])
    weights = {k: 1.0 for k in src}
    if "own" in src:
        weights["own"] = 2.0
    num = sum(src[k].fillna(0) * weights[k] for k in src)
    den = sum(src[k].notna() * weights[k] for k in src)
    soft = (num / den).clip(0.01, 0.99)

    print("AUC on the 58 gold studies (label source vs official label):")
    table = {}
    for k, v in {**src, "consensus": soft}.items():
        table[k] = [roc_auc_score(gold.loc[is_gold, c], v.loc[is_gold, c].fillna(0.5)) for c in LABELS]
    t = pd.DataFrame(table, index=LABELS)
    t.loc["MEAN"] = t.mean()
    print(t.round(3).to_string())

    out = soft.copy()
    out.insert(0, "is_gold", is_gold.astype(int))
    out["fold"] = -1
    weak = ~is_gold.values
    groups = tr.Report.map(report_group).values[weak]
    out.loc[weak, "fold"] = stable_group_folds(groups)
    for c in LABELS:  # official labels for the gold rows, kept separately from the soft targets
        out[f"gold_{c}"] = gold[c]
    Path(WORK).mkdir(parents=True, exist_ok=True)
    t.to_csv(f"{WORK}/labels_{version}_auc.csv", index_label="label")
    out.to_csv(f"{WORK}/labels_{version}.csv")
    print("saved", f"{WORK}/labels_{version}.csv", out.shape, "| fold sizes", out.fold.value_counts().sort_index().to_dict())
    print("mean soft label:", out[LABELS].mean().round(2).to_dict())


if __name__ == "__main__":
    main()
