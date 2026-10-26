import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.reader.knee import LABELS

spec = importlib.util.spec_from_file_location(
    "build_distill_labels", Path(__file__).resolve().parents[1] / "scripts" / "build_distill_labels.py")
bdl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bdl)


def frame(n, gold, seed):
    rng = np.random.default_rng(seed)
    lab = pd.DataFrame(rng.uniform(0.05, 0.95, (n, len(LABELS))), columns=LABELS,
                       index=[f"s{i}" for i in range(n)])
    lab.insert(0, "is_gold", [1] * gold + [0] * (n - gold))
    lab["fold"] = [-1] * gold + list(np.arange(n - gold) % 5)
    return lab


def test_platt_recovers_known_scale_and_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2, 20000)
    y = 1 / (1 + np.exp(-(1.7 * x - 0.4)))
    a, b = bdl.fit_platt(x, y)
    assert abs(a - 1.7) < 1e-3 and abs(b + 0.4) < 1e-3


def test_studies_without_teacher_keep_v0_and_gold_is_not_fitted():
    lab = frame(200, 10, 1)
    noise = np.random.default_rng(2).normal(0, 0.1, (len(lab), len(LABELS)))
    teacher = (lab[LABELS] + noise).clip(0.01, 0.99)  # a teacher that agrees with v0 up to noise
    missing = ["s3", "s50", "s150"]
    out, params, _, fell_back = bdl.build(lab, teacher.drop(index=missing))
    assert sorted(fell_back) == sorted(missing)
    pd.testing.assert_frame_equal(out.loc[missing, LABELS], lab.loc[missing, LABELS])
    assert (out.fold == lab.fold).all() and (out.is_gold == lab.is_gold).all()
    # Changing the teacher on gold rows must not change the fitted parameters.
    t2 = teacher.drop(index=missing).copy()
    t2.loc[[s for s in lab.index[:10] if s not in missing], LABELS] = 0.5
    _, params2, _, _ = bdl.build(lab, t2)
    assert params == params2
    assert out[LABELS].min().min() >= 0.01 and out[LABELS].max().max() <= 0.99
