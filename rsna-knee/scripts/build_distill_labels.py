"""Distillation labels for T4: 0.5 * v0 + 0.5 * Platt-calibrated teacher.

  python scripts/build_distill_labels.py --teacher experiments/S5-T4/teacher_probs.csv \\
      --out experiments/S5-T4/labels_t4.csv --teacher-name "..."

Per finding, the teacher's logit is rescaled as sigmoid(a * logit(t) + b), with (a, b) fitted by
soft-target BCE against the v0 labels on the weak studies only (BFGS); the gold studies are never
used for fitting. Studies the teacher could not score keep their v0 labels. The procedure is the one
that produced experiments/S5-T4/labels_t4.csv (sha256 b0b8b540...) on 2026-10-10, moved into a script.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rsna_knee.evaluation import targets  # noqa: E402
from rsna_knee.metrics import macro_auc, paired_bootstrap  # noqa: E402
from rsna_knee.reader.knee import LABELS  # noqa: E402

V0 = ROOT / "experiments" / "S1" / "labels" / "labels_v0.csv"


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_platt(x, y):
    """(a, b) minimising soft-target BCE of sigmoid(a * x + b) against y."""
    def loss(ab):
        z = ab[0] * x + ab[1]
        p = 1 / (1 + np.exp(-z))
        return np.mean(np.logaddexp(0, z) - y * z), np.array([np.mean((p - y) * x), np.mean(p - y)])

    res = minimize(loss, [1.0, 0.0], jac=True, method="BFGS")
    if not res.success or res.x[0] <= 0:
        raise RuntimeError(f"Platt fit failed: {res.message}, a={res.x[0]}")
    return float(res.x[0]), float(res.x[1])


def build(lab, teacher):
    """Return (labels frame, Platt params, calibrated teacher, rows that fell back to v0)."""
    t = teacher.reindex(lab.index)[LABELS]
    have = t.notna().all(axis=1)
    v0 = lab[LABELS]
    weak = lab.index[(lab.is_gold == 0) & have]
    cal, params = v0.copy(), {}
    for c in LABELS:
        a, b = fit_platt(logit(t.loc[weak, c].to_numpy()), v0.loc[weak, c].to_numpy())
        params[c] = [a, b]
        cal.loc[have, c] = 1 / (1 + np.exp(-(a * logit(t.loc[have, c].to_numpy()) + b)))
    out = lab.copy()
    out[LABELS] = (0.5 * v0 + 0.5 * cal).clip(0.01, 0.99)
    out.loc[~have, LABELS] = v0.loc[~have]
    return out, params, cal, lab.index[~have]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=V0, help="original v0 labels and fixed fold table")
    ap.add_argument("--teacher", type=Path, required=True, help="csv indexed by StudyInstanceUID with the 12 label columns")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--teacher-name", required=True)
    args = ap.parse_args()
    lab = pd.read_csv(args.labels, index_col=0)
    teacher = pd.read_csv(args.teacher, index_col=0)
    out, params, cal, missing = build(lab, teacher)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out)
    g = lab.index[lab.is_gold == 1]
    yg = targets(lab, g, "gold")
    g_have = [s for s in g if s not in set(missing)]
    r = paired_bootstrap(yg, lab.loc[g, LABELS].to_numpy(), out.loc[g, LABELS].to_numpy(), n=2000)
    report = {
        "mix": "0.5 * v0 + 0.5 * platt(teacher)", "platt_logit_scale_shift": params,
        "fit_on": f"{int(((lab.is_gold == 0) & ~lab.index.isin(missing)).sum()):,} weak studies, soft-target BCE vs v0 (BFGS)",
        "teacher": args.teacher_name,
        "teacher_gold_auc": macro_auc(targets(lab, g_have, "gold"), teacher.loc[g_have, LABELS].to_numpy()) if g_have else None,
        "labels_gold_auc": macro_auc(yg, out.loc[g, LABELS].to_numpy()),
        "labels_v0_gold_auc": macro_auc(yg, lab.loc[g, LABELS].to_numpy()),
        "labels_vs_v0_gold_delta": [r["delta_observed"], r["ci_low"], r["ci_high"]],
        "teacher_missing_rows": len(missing), "teacher_missing_gold_rows": len(g) - len(g_have),
        "labels_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "platt_logit_scale_shift"}, indent=1))


if __name__ == "__main__":
    main()
