import numpy as np
import pandas as pd
import pytest

from rsna_knee.reader.knee import LABELS
from scripts.compare import compare, decide


def result(low, high, mean=None, delta=None):
    mean = (low + high) / 2 if mean is None else mean
    return {"ci_low": low, "ci_high": high, "delta_mean": mean, "delta_observed": mean if delta is None else delta}


def test_decision_rule_matches_report():
    assert decide(result(0.001, 0.01), result(-0.05, 0.03, delta=-0.009)) == "adopt"
    assert decide(result(0.001, 0.01), result(-0.05, 0.03, delta=-0.011)) == "reject"
    assert decide(result(-0.001, 0.01), result(-0.05, 0.03)) == "pending"
    assert decide(result(-0.01, 0.002, mean=-0.004), result(0.0, 0.03)) == "reject"
    assert decide(None, result(0.0, 0.03)) == "gold_only"
    assert decide(result(0.001, 0.01), result(0.0, 0.03), labels_changed=True) == "gold_only"


def test_compare_runs_on_common_studies(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    studies = [f"s{i}" for i in range(n)]
    labels = pd.DataFrame(rng.random((n, len(LABELS))), index=pd.Index(studies, name="StudyInstanceUID"),
                          columns=LABELS)
    labels["is_gold"] = [1] * 40 + [0] * 160
    for c in LABELS:
        labels[f"gold_{c}"] = rng.integers(0, 2, n)
    labels.to_csv(tmp_path / "labels.csv")
    for name, noise in [("a", 2.0), ("b", 0.2)]:
        d = tmp_path / name
        d.mkdir()
        weak = labels[labels.is_gold == 0]
        gold = labels[labels.is_gold == 1]
        (weak[LABELS] + rng.normal(0, noise, weak[LABELS].shape)).iloc[: 150 if name == "a" else 160].to_csv(
            d / "preds_oof.csv")
        gold_target = gold[[f"gold_{c}" for c in LABELS]].to_numpy()
        pd.DataFrame(gold_target + rng.normal(0, noise, gold_target.shape), index=gold.index,
                     columns=LABELS).to_csv(d / "preds_gold.csv")
    out = compare(tmp_path / "a", tmp_path / "b", tmp_path / "labels.csv", n=200)
    assert out["oof"]["studies"] == 150 and out["oof"]["only_in_b"] == 10
    assert out["verdict"] == "adopt" and out["oof"]["ci_low"] > 0
    with pytest.raises(ValueError, match="preds_gold"):
        (tmp_path / "b" / "preds_gold.csv").unlink()
        compare(tmp_path / "a", tmp_path / "b", tmp_path / "labels.csv", n=50)
