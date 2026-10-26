import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from rsna_knee.metrics import _weighted_auc, macro_auc, paired_bootstrap, per_label_auc, rank_mean


def test_macro_auc_excludes_single_class_labels():
    y = np.array([[0, 1], [1, 1], [0, 1], [1, 1]])
    p = np.array([[0.1, 0.1], [0.9, 0.2], [0.2, 0.8], [0.8, 0.9]])
    assert macro_auc(y, p) == 1
    assert np.isnan(per_label_auc(y, p)[1])
    assert np.isnan(macro_auc(np.ones_like(y), p))


def test_macro_auc_matches_sklearn():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, (58, 12))
    p = rng.random(y.shape)
    expected = np.mean([roc_auc_score(y[:, i], p[:, i]) for i in range(12)])
    assert macro_auc(y, p) == pytest.approx(expected)


def test_evaluation_requires_binary_finite_aligned_labels():
    with pytest.raises(ValueError, match="binary"):
        macro_auc([[0.1], [0.9]], [[0.1], [0.9]])
    with pytest.raises(ValueError, match="finite"):
        macro_auc([[0], [1]], [[np.nan], [1]])
    y = pd.DataFrame({"ACL": [0, 1]}, index=["a", "b"])
    with pytest.raises(ValueError, match="aligned"):
        macro_auc(y, y.iloc[::-1])


def test_identical_predictions_have_zero_paired_delta():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (58, 12))
    p = rng.random(y.shape)
    result = paired_bootstrap(y, p, p, n=2000)
    assert result["delta_observed"] == result["delta_mean"] == result["ci_low"] == result["ci_high"] == 0
    assert result["p_positive"] == 0
    assert result["n_valid"] + result["n_invalid"] == 2000


def test_bootstrap_preserves_pairing_and_delta_direction():
    y = np.tile([[0], [1]], (50, 1))
    good = y.astype(float)
    result = paired_bootstrap(y, 1 - good, good, n=200, seed=17)
    assert result["delta_observed"] == result["delta_mean"] == result["ci_low"] == result["ci_high"] == 1
    assert result["p_positive"] == 1
    assert result == paired_bootstrap(y, 1 - good, good, n=200, seed=17)


def test_weighted_auc_matches_explicit_resampling_with_ties():
    y = np.array([0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.3, 0.3, 0.6, 0.8, 0.8])
    weights = np.random.default_rng(23).multinomial(6, np.full(6, 1 / 6), size=100)
    actual = _weighted_auc(y, p, weights)
    for row, auc in zip(weights, actual):
        indices = np.repeat(np.arange(len(y)), row)
        if len(np.unique(y[indices])) == 1:
            assert np.isnan(auc)
        else:
            assert auc == pytest.approx(roc_auc_score(y[indices], p[indices]))


def test_degenerate_draws_are_reported_and_macro_labels_stay_fixed():
    y = np.array([[0, 1], [1, 1]])
    result = paired_bootstrap(y, y, y, n=100, seed=1)
    assert result["eligible_labels"] == [0]
    assert result["n_invalid"] > 0
    assert result["n_valid"] + result["n_invalid"] == 100
    with pytest.raises(ValueError, match="both classes"):
        paired_bootstrap(np.ones((2, 2)), y, y)


def test_group_bootstrap_resamples_complete_groups():
    y = np.tile([[0], [1]], (3, 1))
    result = paired_bootstrap(y, 1 - y, y, n=100, groups=["a", "a", "b", "b", "c", "c"])
    assert result["sampling_unit"] == "group" and result["n_units"] == 3
    assert result["n_invalid"] == 0 and result["ci_low"] == 1


def test_rank_mean_ranks_each_label_and_handles_ties():
    a = np.array([[1, 10], [1, 30], [3, 20]])
    b = np.array([[2, 30], [1, 20], [3, 10]])
    expected = (np.array([[1.5, 1], [1.5, 3], [3, 2]]) + np.array([[2, 3], [1, 2], [3, 1]])) / 6
    np.testing.assert_allclose(rank_mean([a, b]), expected)
    with pytest.raises(ValueError):
        rank_mean([a, b[:2]])
