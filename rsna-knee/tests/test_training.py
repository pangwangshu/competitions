import os

import pandas as pd
import pytest

from rsna_knee import training
from tests.test_evaluation import _write_fixture

LOG = [(12.0, '{"name": "x"}'), (30.0, "train 3479  val 870  gold 58"),
       (95.0, "  ep 0 step 100/434 loss 0.4100 0.61s/it"),
       (400.0, "ep  0 loss 0.4012 time 300s | gold 0.8123 | val 0.8456 [0.9 0.8]"),
       (760.0, "ep  1 loss 0.3500 time 290s | gold 0.8500")]


def test_parse_log_separates_setup_training_and_evaluation():
    parsed = training.parse_log(LOG)
    assert parsed["setup_seconds"] == 30.0 and parsed["train_val_gold"] == [3479, 870, 58]
    first, second = parsed["epochs"]
    assert first["train_seconds"] == 300 and first["eval_seconds"] == pytest.approx(70.0)
    assert first["val"] == 0.8456 and second["val"] is None
    assert second["eval_seconds"] == pytest.approx(760 - 400 - 290)
    assert parsed["step_reports"] == [(95.0, 100, 0.61)]


def test_flags_render_train_arguments():
    assert training.flags({"fold": 0, "amp": "fp16"}) == ["--fold", "0", "--amp", "fp16"]


def test_prepare_work_links_cache_and_checks_pins(tmp_path):
    data, labels, train_meta, test_meta, caches, ckpts = _write_fixture(tmp_path)
    work = tmp_path / "work"
    n = training.prepare_work(work, tmp_path / "labels.csv", train_meta, caches[:1],
                              meta_sha256=training.meta_digest(train_meta))
    assert n == len(train_meta) and (work / "labels_v0.csv").exists()
    link = work / "cache384_train" / f"{train_meta.series[0]}.npy"
    assert os.path.islink(link) and pd.read_csv(work / "cache384_train_meta.csv").equals(train_meta)
    with pytest.raises(ValueError, match="labels"):
        training.prepare_work(work, tmp_path / "labels.csv", train_meta, caches[:1], labels_sha256="0" * 64)


def test_parse_log_accepts_epochs_without_evaluation():
    parsed = training.parse_log([(5.0, "train 10  val 5  gold 3"), (100.0, "ep  0 loss 0.5000 time 90s"),
                                 (300.0, "ep  1 loss 0.4000 time 95s | gold 0.8000 | val 0.7000 [0.7]")])
    first, second = parsed["epochs"]
    assert first["gold"] is None and first["eval_seconds"] == pytest.approx(5.0)
    assert second["gold"] == 0.8 and second["eval_seconds"] == pytest.approx(105.0)
