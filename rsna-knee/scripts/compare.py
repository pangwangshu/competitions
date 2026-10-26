"""Compare two evaluation runs with the paired-bootstrap decision rule.

Usage: python scripts/compare.py <eval_dir_A> <eval_dir_B> [--labels-changed] [--out result.json]
A is the control, B the variant; Δ = AUC(B) - AUC(A). Each dir holds preds_oof.csv
and/or preds_gold.csv from rsna_knee.evaluation. Rule (report/writeup.md, §5.3):
  adopt    Δ_oof 95% CI lower bound > 0 and Δ_gold >= -0.01
  pending  Δ_oof bootstrap mean > 0 but the CI crosses 0
  reject   everything else
With --labels-changed (training labels differ, so M_oof is biased) or when either run
lacks OOF predictions, only Δ_gold is reported and the decision is left to gold + LB.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsna_knee.evaluation import targets, write_json  # noqa: E402
from rsna_knee.metrics import paired_bootstrap  # noqa: E402
from rsna_knee.reader.knee import LABELS  # noqa: E402

DEFAULT_LABELS = Path(__file__).resolve().parents[1] / "experiments" / "S1" / "labels" / "labels_v0.csv"
GOLD_FLOOR = -0.01


def paired(labels, a_path, b_path, kind, n, seed):
    a, b = pd.read_csv(a_path, index_col=0), pd.read_csv(b_path, index_col=0)
    common = a.index.intersection(b.index).sort_values()
    if not len(common):
        raise ValueError(f"No common studies between {a_path} and {b_path}")
    result = paired_bootstrap(targets(labels, common, kind), a.loc[common, LABELS].to_numpy(),
                              b.loc[common, LABELS].to_numpy(), n=n, seed=seed)
    result.update(studies=len(common), only_in_a=len(a.index.difference(common)),
                  only_in_b=len(b.index.difference(common)))
    return result


def decide(oof, gold, labels_changed=False):
    if labels_changed or oof is None:
        return "gold_only"
    if oof["ci_low"] > 0 and (gold is None or gold["delta_observed"] >= GOLD_FLOOR):
        return "adopt"
    if oof["delta_mean"] > 0 and oof["ci_low"] <= 0 < oof["ci_high"]:
        return "pending"
    return "reject"


def compare(a, b, labels_path=DEFAULT_LABELS, labels_changed=False, n=2000, seed=0):
    labels = pd.read_csv(labels_path, index_col=0)
    a, b = Path(a), Path(b)
    result = {"control": str(a), "variant": str(b), "labels_changed": labels_changed, "n_boot": n, "seed": seed}
    for kind, name in [("weak", "oof"), ("gold", "gold")]:
        pa, pb = a / f"preds_{name}.csv", b / f"preds_{name}.csv"
        result[name] = paired(labels, pa, pb, kind, n, seed) if pa.exists() and pb.exists() else None
    if result["gold"] is None:
        raise ValueError("Both runs need preds_gold.csv")
    result["verdict"] = decide(result["oof"], result["gold"], labels_changed)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--labels-changed", action="store_true")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = compare(args.a, args.b, args.labels, args.labels_changed, args.n)
    for name in ("oof", "gold"):
        r = result[name]
        if r is not None:
            print(f"{name:4s} Δ={r['delta_observed']:+.4f} mean={r['delta_mean']:+.4f} "
                  f"CI=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] P(Δ>0)={r['p_positive']:.3f} n={r['studies']}")
    print(f"verdict: {result['verdict']}")
    if args.out:
        write_json(args.out, result)


if __name__ == "__main__":
    main()
