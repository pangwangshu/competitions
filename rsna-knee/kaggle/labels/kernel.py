"""Reproduce the author's v0 consensus and report-grouped five folds."""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from rsna_knee.reader import make_labels
from rsna_knee.reader.cache import environment, write_json
from rsna_knee.reader.paths import DATA, EXT, WORK

sys.argv = ["make_labels.py", "v0"]
make_labels.main()
labels = pd.read_csv(Path(WORK) / "labels_v0.csv")
auc = pd.read_csv(Path(WORK) / "labels_v0_auc.csv").set_index("label")
sources = []
for owner, slug, filename in [
    ("stevenleehans", "rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv"),
    ("pilkwang", "rsna-knee-llm-labels", "report_labels_v2.csv"),
    ("lixin73", "rsna-knee-llm-report-labels-sol56", "labels_llm_gpt56sol.csv"),
    ("laymond", "rsna-knee-abnormality-qwen3-8b-weak-labels", "qwen_knee_weak_labels.csv"),
]:
    candidates = [Path(EXT) / "datasets" / owner / slug / filename,
                  Path(EXT) / slug / filename, Path(EXT) / f"{owner}_{slug}" / filename]
    sources.append(next(p for p in candidates if p.is_file()))
gold = labels.is_gold.eq(1)
weak = labels.loc[~gold]
groups = pd.read_csv(Path(DATA) / "train.csv").set_index("StudyInstanceUID").Report.map(make_labels.report_group)
group_folds = weak.set_index("StudyInstanceUID").assign(group=groups).groupby("group").fold.nunique()
checks = {"rows_4407": len(labels) == 4407, "unique_studies": labels.StudyInstanceUID.is_unique,
          "gold_58_fold_minus1": int(gold.sum()) == 58 and bool(labels.loc[gold, "fold"].eq(-1).all()),
          "weak_4349": len(weak) == 4349,
          "five_balanced_folds": set(weak.fold) == set(range(5)) and bool(weak.fold.value_counts().between(865, 875).all()),
          "complete_soft_labels": bool(labels[make_labels.LABELS].notna().all().all()),
          "duplicate_reports_stay_together": bool(group_folds.eq(1).all()),
          "consensus_auc_in_range": abs(float(auc.loc["MEAN", "consensus"]) - 0.896) <= 0.005}
report = {"checks": checks, "fold_counts": {str(k): int(v) for k, v in labels.fold.value_counts().sort_index().items()},
          "consensus_macro_auc": float(auc.loc["MEAN", "consensus"]), "environment": environment(),
          "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in sources],
          "input_sha256": {name: hashlib.sha256((Path(DATA) / name).read_bytes()).hexdigest()
                           for name in ["train.csv", "train_series.csv", "test.csv", "test_series.csv"]},
          "report_groups_sha256": hashlib.sha256(json.dumps(sorted(
              (str(study), str(group)) for study, group in groups.items()), ensure_ascii=False).encode()).hexdigest()}
write_json(Path(WORK) / "labels_report.json", report)
print(json.dumps(report, indent=2))
# The plan allows an AUC mismatch, with the difference recorded for S2.
assert all(v for k, v in checks.items() if k != "consensus_auc_in_range"), checks
