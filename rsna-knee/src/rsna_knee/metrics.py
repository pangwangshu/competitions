"""Macro ROC AUC, paired study bootstrap, and per-label rank ensembles.

Inputs to AUC functions must already contain binary evaluation labels. Weak
labels must be thresholded explicitly by the caller (the reader uses > 0.5).
Bootstrap uncertainty concerns the supplied predictions and evaluation sample;
it does not measure training randomness or correct for experiment selection.
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def _arrays(y, *predictions):
    for prediction in predictions:
        if hasattr(y, "columns") and hasattr(prediction, "columns"):
            if not y.index.equals(prediction.index) or not y.columns.equals(prediction.columns):
                raise ValueError("Study IDs and label columns must be aligned before scoring")
    y = np.asarray(y)
    predictions = tuple(np.asarray(p, dtype=float) for p in predictions)
    if y.ndim != 2 or not all(p.shape == y.shape for p in predictions) or 0 in y.shape:
        raise ValueError("Labels and predictions must have the same nonempty [study, label] shape")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Evaluation labels must be finite binary values (0 or 1)")
    if not all(np.isfinite(p).all() for p in predictions):
        raise ValueError("Predictions must be finite")
    return y.astype(np.int8), *predictions


def per_label_auc(y, p):
    """Return label AUCs, using NaN for labels containing only one class."""
    y, p = _arrays(y, p)
    return np.array([
        roc_auc_score(y[:, i], p[:, i]) if 0 < y[:, i].sum() < len(y) else np.nan
        for i in range(y.shape[1])
    ])


def macro_auc(y, p):
    """Mean of evaluable label AUCs; NaN if no label has both classes."""
    aucs = per_label_auc(y, p)
    return float(np.nanmean(aucs)) if np.isfinite(aucs).any() else float("nan")


def _weighted_auc(y, p, weights):
    """AUC for many bootstrap draws, treating ties as half concordant.

    Integer weights represent repeated studies. Grouping tied scores produces
    exactly the same AUC as materializing those studies and calling sklearn.
    """
    order = np.argsort(p, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    sorted_weights = weights[:, order]
    positives = np.add.reduceat(sorted_weights * y[order], starts, axis=1)
    negatives = np.add.reduceat(sorted_weights * (1 - y[order]), starts, axis=1)
    numerator = (positives * (np.cumsum(negatives, axis=1) - negatives / 2)).sum(axis=1)
    denominator = positives.sum(axis=1) * negatives.sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)


def _bootstrap_macro(y, predictions, n, seed, groups):
    """Macro AUC of each prediction on the same valid draws: ([draw, model], meta)."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    eligible = np.flatnonzero((y.sum(axis=0) > 0) & (y.sum(axis=0) < len(y)))
    if not len(eligible):
        raise ValueError("Bootstrap needs at least one label with both classes")
    if groups is None:
        inverse = np.arange(len(y))
    else:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(y):
            raise ValueError("groups must contain one group ID per study")
        _, inverse = np.unique(groups, return_inverse=True)
    n_units = int(inverse.max()) + 1
    rng = np.random.default_rng(seed)
    macros = []
    for start in range(0, n, 64):
        draws = rng.multinomial(n_units, np.full(n_units, 1 / n_units), size=min(64, n - start))
        weights = draws[:, inverse]
        aucs = [np.column_stack([_weighted_auc(y[:, i], p[:, i], weights) for i in eligible]) for p in predictions]
        valid = np.all([np.isfinite(a).all(axis=1) for a in aucs], axis=0)
        macros.append(np.column_stack([a[valid].mean(axis=1) for a in aucs]))
    macros = np.concatenate(macros)
    if not len(macros):
        raise ValueError("All bootstrap draws lacked a class; use a larger evaluation sample")
    meta = {"n_requested": int(n), "n_valid": len(macros), "n_invalid": int(n - len(macros)), "seed": int(seed),
            "sampling_unit": "study" if groups is None else "group", "n_units": n_units,
            "eligible_labels": eligible.tolist()}
    return macros, meta


def bootstrap_auc(y, p, n=2000, seed=0, *, groups=None):
    """Return macro AUC with a percentile 95% CI over resampled studies (or groups).

    The CI covers labels evaluable in the full sample, as in paired_bootstrap.
    """
    y, p = _arrays(y, p)
    macros, meta = _bootstrap_macro(y, [p], n, seed, groups)
    low, high = np.quantile(macros[:, 0], [0.025, 0.975])
    return {"auc": macro_auc(y, p), "boot_mean": float(macros[:, 0].mean()),
            "ci_low": float(low), "ci_high": float(high), **meta}


def paired_bootstrap(y, p_a, p_b, n=2000, seed=0, *, groups=None):
    """Return paired Δ = AUC(B) - AUC(A), a percentile CI, and P(Δ > 0).

    By default studies are sampled with replacement. Pass one group ID per
    study to resample complete groups instead. The same draw is used for both
    models and all labels. Labels constant in the original sample are excluded;
    draws missing a class in any remaining label are discarded to keep the
    macro average comparable. n_valid/n_invalid report that loss explicitly.
    """
    y, p_a, p_b = _arrays(y, p_a, p_b)
    macros, meta = _bootstrap_macro(y, [p_a, p_b], n, seed, groups)
    differences = macros[:, 1] - macros[:, 0]
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "delta_observed": macro_auc(y, p_b) - macro_auc(y, p_a),
        "delta_mean": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_positive": float((differences > 0).mean()),
        **meta,
    }


def rank_mean(preds):
    """Mean normalized ranks per model and label, shape [study, label].

    The input is [model, study, label]. DataFrames must have matching study IDs
    and columns; NumPy inputs must already be aligned. Ties use average ranks.
    """
    preds = list(preds)
    if not preds:
        raise ValueError("At least one prediction array is required")
    for p in preds[1:]:
        if hasattr(preds[0], "columns") and hasattr(p, "columns"):
            if not preds[0].index.equals(p.index) or not preds[0].columns.equals(p.columns):
                raise ValueError("Study IDs and label columns must be aligned before ensembling")
    arrays = [np.asarray(p, dtype=float) for p in preds]
    if arrays[0].ndim != 2 or 0 in arrays[0].shape or not all(a.shape == arrays[0].shape for a in arrays):
        raise ValueError("Prediction arrays must have the same nonempty [study, label] shape")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Predictions must be finite")
    return np.mean([rankdata(a, axis=0, method="average") / len(a) for a in arrays], axis=0)
