"""Collect out-of-fold predictions of one or more runs and score them.

Source: goodpjw2008/rsna-knee-2-5d-convnext-reader.

Usage: python -m rsna_knee.reader.oof <labels_version> <run> [<run> ...]
Writes runs/<run>/oof.csv (probabilities; one row per weak-labelled study from its held-out fold,
gold studies averaged over the folds) and prints per-class AUC for each run and for their mean.
"""
import glob
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from .knee import LABELS
from .paths import RUNS, WORK


def load_run(run):
    parts = [pd.read_csv(f, index_col=0) for f in sorted(glob.glob(f"{RUNS}/{run}/oof_fold[0-9].csv"))]
    df = pd.concat(parts)
    prob = 1 / (1 + np.exp(-df[LABELS]))
    val = prob[df.set == "val"]
    gold = prob[df.set == "gold"].groupby(level=0).mean()
    out = pd.concat([val, gold])
    out.to_csv(f"{RUNS}/{run}/oof.csv")
    return out, len(parts)


def score(pred, lab):
    weak = lab[lab.is_gold == 0].index.intersection(pred.index)
    gold = lab[lab.is_gold == 1].index.intersection(pred.index)
    w = [roc_auc_score(lab.loc[weak, c] > 0.5, pred.loc[weak, c]) for c in LABELS]
    g = [roc_auc_score(lab.loc[gold, f"gold_{c}"], pred.loc[gold, c]) for c in LABELS]
    return w, g, len(weak)


def main():
    lab = pd.read_csv(f"{WORK}/{sys.argv[1]}.csv", index_col=0)
    table, preds = {}, []
    for run in sys.argv[2:]:
        p, nf = load_run(run)
        w, g, n = score(p, lab)
        table[f"{run}({nf}f) weak"], table[f"{run} gold"] = w, g
        preds.append(p)
        print(f"{run}: {nf} folds, {n} weak-labelled OOF studies")
    if len(preds) > 1:
        common = preds[0].index
        for p in preds[1:]:
            common = common.intersection(p.index)
        ens = sum(p.loc[common].apply(rankdata) / len(common) for p in preds) / len(preds)
        ens = pd.DataFrame(ens, index=common, columns=LABELS)
        table["rank-mean weak"], table["rank-mean gold"], _ = score(ens, lab)
    t = pd.DataFrame(table, index=LABELS)
    t.loc["MEAN"] = t.mean()
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
