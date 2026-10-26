import json

import numpy as np
import pandas as pd
import pytest
import torch

from rsna_knee import evaluation
from rsna_knee.reader.knee import LABELS, KneeNet

CKPTS = [{"file": f"f{k}.pt", "fold": k} for k in range(3)]


def test_config_rejects_unknown_keys_and_shared_folds():
    assert evaluation.validate_config({"ckpts": CKPTS})["infer"]["k_eval"] == 12
    with pytest.raises(ValueError, match="Unknown config"):
        evaluation.validate_config({"ckpts": CKPTS, "tta": "mirror"})
    with pytest.raises(ValueError, match="Unknown infer"):
        evaluation.validate_config({"ckpts": CKPTS, "infer": {"fov": 0.9}})
    with pytest.raises(ValueError, match="tta must be"):
        evaluation.validate_config({"ckpts": CKPTS, "infer": {"tta": "flip"}})
    with pytest.raises(ValueError, match="at most one"):
        evaluation.validate_config({"ckpts": [{"file": "a", "fold": 0}, {"file": "b", "fold": 0}]})


def test_ensembles_keep_study_order():
    logits = np.array([[[0.0, 2.0], [1.0, -1.0]], [[2.0, 0.0], [3.0, -3.0]]])
    prob = evaluation.ensemble(logits, "prob")
    np.testing.assert_allclose(prob, (1 / (1 + np.exp(-logits))).mean(0))
    np.testing.assert_allclose(evaluation.ensemble(logits, "logit"), 1 / (1 + np.exp(-logits.mean(0))))
    np.testing.assert_allclose(evaluation.ensemble(logits, "rank"), [[0.5, 1.0], [1.0, 0.5]])


def _labels(n_per_fold=40, seed=0):
    rng = np.random.default_rng(seed)
    folds = np.repeat(np.arange(5), n_per_fold)
    studies = [f"s{i:04d}" for i in range(len(folds))]
    soft = rng.random((len(folds), len(LABELS)))
    frame = pd.DataFrame(soft, index=pd.Index(studies, name="StudyInstanceUID"), columns=LABELS)
    frame.insert(0, "is_gold", 0)
    frame["fold"] = folds
    return frame


def _cross_logits(labels, held_noise=3.0, trained_noise=0.3, seed=0):
    """Logits that fit trained folds closely and held-out folds poorly."""
    rng = np.random.default_rng(seed)
    soft = labels[LABELS].to_numpy()
    rows = []
    for ckpt in range(3):
        noise = np.where((labels.fold.to_numpy() == ckpt)[:, None], held_noise, trained_noise)
        logit = np.log(soft / (1 - soft)) + rng.normal(0, 1, soft.shape) * noise
        frame = pd.DataFrame(logit, columns=LABELS)
        frame.insert(0, "ckpt_fold", ckpt)
        frame.insert(0, "ckpt", ckpt)
        frame.insert(0, "StudyInstanceUID", labels.index)
        frame.insert(0, "source", "cache")
        frame.insert(0, "task", "weak")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_oof_uses_only_the_held_out_checkpoint():
    labels = _labels()
    logits = _cross_logits(labels)
    oof = evaluation.oof_predictions(logits, labels)
    assert len(oof) == 3 * 40 and set(oof.fold) == {0, 1, 2}
    expected = logits[(logits.ckpt == 1)].set_index("StudyInstanceUID").loc[oof.index[oof.fold == 1], LABELS]
    np.testing.assert_allclose(oof.loc[oof.fold == 1, LABELS], 1 / (1 + np.exp(-expected)))


def test_fold_check_detects_matching_and_shuffled_folds():
    labels = _labels()
    logits = _cross_logits(labels)
    check = evaluation.fold_check(logits, labels)
    for ckpt in "012":
        assert check["held_out_vs_trained"][ckpt]["gap"] > 0.05
        assert check["membership"][ckpt]["held_out_ckpt_has_highest_loss"] > 0.9
        assert check["same_studies"][ckpt]["gap"] > 0.05
    shuffled = labels.assign(fold=np.random.default_rng(1).permutation(labels.fold.to_numpy()))
    check = evaluation.fold_check(logits, shuffled)
    for ckpt in "012":
        assert abs(check["held_out_vs_trained"][ckpt]["gap"]) < 0.05
        assert check["membership"][ckpt]["held_out_ckpt_has_highest_loss"] < 0.6


def _write_fixture(tmp_path):
    """Two folds of weak studies, gold and test studies, one 8-slice series per study."""
    rng = np.random.default_rng(0)
    data, cache_train, cache_test = tmp_path / "data", tmp_path / "cache_train", tmp_path / "cache_test"
    for d in (data, cache_train, cache_test):
        d.mkdir()
    rows, series_rows = [], {"train": [], "test": []}
    meta = {"train": [], "test": []}
    studies = [(f"w{i}", "train", i % 2, 0) for i in range(8)] + [(f"g{i}", "train", -1, 1) for i in range(6)]
    studies += [(f"t{i}", "test", None, None) for i in range(2)]
    for study, split, fold, gold in studies:
        series = f"{study}.s"
        volume = rng.integers(0, 255, (8, 64, 64), dtype=np.uint8)
        np.save((cache_train if split == "train" else cache_test) / f"{series}.npy", volume)
        series_rows[split].append({"StudyInstanceUID": study, "SeriesInstanceUID": series,
                                   "Fat_Suppression": 1, "Anatomical_Plane": "Sagittal"})
        meta[split].append({"series": series, "study": study, "n": 8})
        if split == "train":
            row = {"StudyInstanceUID": study, "is_gold": gold, "fold": fold}
            row.update({c: float(rng.random()) for c in LABELS})
            row.update({f"gold_{c}": int(i % 2 == j % 2) for j, c in enumerate(LABELS) for i in [len(rows)]})
            rows.append(row)
    labels = pd.DataFrame(rows)
    gold = labels.is_gold == 1
    for j, c in enumerate(LABELS):  # both classes in every gold label
        labels.loc[gold, f"gold_{c}"] = [0, 1, 0, 1, 1, 0]
    labels.to_csv(tmp_path / "labels.csv", index=False)
    for split in ("train", "test"):
        pd.DataFrame(series_rows[split]).to_csv(data / f"{split}_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ["t0", "t1"]}).to_csv(data / "test.csv", index=False)
    torch.manual_seed(0)
    ckpts = []
    for k in range(2):
        path = tmp_path / f"f{k}.pt"
        torch.save({"model": KneeNet("resnet10t", pretrained=False).state_dict(),
                    "args": {"backbone": "resnet10t", "res": 32}}, path)
        ckpts.append(path)
    return data, labels, pd.DataFrame(meta["train"]), pd.DataFrame(meta["test"]), [cache_train, cache_test], ckpts


def test_evaluate_end_to_end_on_cpu(tmp_path):
    data, labels, train_meta, test_meta, caches, ckpts = _write_fixture(tmp_path)
    config = {"ckpts": [{"file": "f0.pt", "fold": 0}, {"file": "f1.pt", "fold": 1}], "sets": ["cross", "gold", "test"],
              "infer": {"k_eval": 4, "batch_size": 3}, "bootstrap": 50,
              "meta_sha256": evaluation.meta_digest(train_meta)}
    out = tmp_path / "out"
    metrics = evaluation.evaluate(config, data=data, labels_path=tmp_path / "labels.csv", train_meta=train_meta,
                                  test_meta=test_meta, cache_dirs=caches, ckpt_paths=ckpts, out=out, device="cpu")
    logits = pd.read_csv(out / "logits.csv")
    assert logits.groupby("task").size().to_dict() == {"weak": 16, "gold": 12, "test": 4}
    oof = pd.read_csv(out / "preds_oof.csv", index_col=0)
    assert sorted(oof.index) == [f"w{i}" for i in range(8)]
    # Batching and the cross pass must not change a study's logits: rescore one study alone.
    alone = evaluation.evaluate({**config, "sets": ["gold"], "limit": 1}, data=data,
                                labels_path=tmp_path / "labels.csv", train_meta=train_meta, test_meta=test_meta,
                                cache_dirs=caches, ckpt_paths=ckpts, out=tmp_path / "alone", device="cpu")
    one = pd.read_csv(tmp_path / "alone" / "logits.csv")
    both = logits[(logits.task == "gold") & (logits.StudyInstanceUID == "g0")]
    np.testing.assert_allclose(one[LABELS].to_numpy(), both[LABELS].to_numpy(), rtol=1e-4, atol=1e-5)
    assert metrics["gold"]["studies"] == 6 and len(metrics["gold"]["ci"]) == 2
    assert set(json.loads((out / "timing.json").read_text())) >= {"weak", "gold", "test", "workers"}
    with pytest.raises(ValueError, match="metadata differs"):
        evaluation.evaluate({**config, "meta_sha256": "0" * 64}, data=data, labels_path=tmp_path / "labels.csv",
                            train_meta=train_meta, test_meta=test_meta, cache_dirs=caches, ckpt_paths=ckpts,
                            out=tmp_path / "bad", device="cpu")


def test_mirror_volume_flips_patient_left_right_and_is_an_involution():
    from rsna_knee.reader.knee import mirror_volume

    vol = np.arange(4 * 3 * 5).reshape(4, 3, 5)
    np.testing.assert_array_equal(mirror_volume(vol, 0), vol[::-1])  # sagittal: slice order
    np.testing.assert_array_equal(mirror_volume(vol, 4), vol[:, :, ::-1])  # coronal: columns
    np.testing.assert_array_equal(mirror_volume(vol, 2), vol[:, :, ::-1])  # axial: columns
    for slot in range(6):
        np.testing.assert_array_equal(mirror_volume(mirror_volume(vol, slot), slot), vol)


def test_base_view_reproduces_study_dataset(tmp_path):
    from rsna_knee.reader.knee import StudyDataset

    data, labels, train_meta, test_meta, caches, ckpts = _write_fixture(tmp_path)
    se = pd.read_csv(data / "train_series.csv")
    table = evaluation.build_study_table(se, dict(zip(train_meta.series, train_meta.n)))
    studies = sorted(table)
    paths = evaluation.cache_paths(caches)
    view = evaluation.ViewDataset(studies, table, 5, evaluation.CacheVolumes(paths))
    base = StudyDataset(studies, table, None, None, 5, loader=lambda s: np.load(paths[s], mmap_mode="r"))
    for i in range(len(studies)):
        for a, b in zip(view[i], base[i][:3]):
            assert a.dtype == b.dtype and torch.equal(a, b)
    mirrored = evaluation.ViewDataset(studies, table, 5, evaluation.CacheVolumes(paths), mirror=True)
    x, mask, pos = mirrored[0]
    assert torch.equal(mask, view[0][1]) and not torch.equal(x, view[0][0])


def test_base_view_reproduces_dicom_dataset():
    from pathlib import Path

    from rsna_knee.reader.infer import DicomStudyDataset, dicom_table

    from rsna_knee.reader.paths import DATA

    root = Path(DATA)
    if not (root / "test_series").is_dir():
        pytest.skip("local competition test DICOMs are not available")
    studies = pd.read_csv(root / "test.csv").StudyInstanceUID.tolist()[:1]
    table = dicom_table(pd.read_csv(root / "test_series.csv"), str(root / "test_series"), studies)
    view = evaluation.ViewDataset(studies, table, 4, evaluation.DicomVolumes(str(root / "test_series")))[0]
    base = DicomStudyDataset(studies, table, str(root / "test_series"), 4)[0]
    for a, b in zip(view, base):
        assert torch.equal(a, b)


def test_task_views_add_passes_only_where_candidates_exist():
    table = {"a": [["a1", "a2", "a3"], [], [], [], [], []], "b": [["b1"], [], [], [], [], []]}
    task = {"table": table}
    infer = {"series": "all", "tta": "mirror"}
    views = evaluation.task_views(task, infer, ["a", "b"])
    assert [(p, m, v) for p, m, v in views] == [
        (0, False, ["a", "b"]), (0, True, ["a", "b"]), (1, False, ["a"]), (1, True, ["a"]),
        (2, False, ["a"]), (2, True, ["a"])]
    assert evaluation.task_views(task, {"series": "first", "tta": "none"}, ["a", "b"]) == [(0, False, ["a", "b"])]


def test_multi_view_logits_average_probabilities(tmp_path):
    data, labels, train_meta, test_meta, caches, ckpts = _write_fixture(tmp_path)
    base = {"ckpts": [{"file": "f0.pt", "fold": 0}, {"file": "f1.pt", "fold": 1}], "sets": ["gold"],
            "infer": {"k_eval": 4, "batch_size": 4}, "bootstrap": 20}
    kw = dict(data=data, labels_path=tmp_path / "labels.csv", train_meta=train_meta, test_meta=test_meta,
              cache_dirs=caches, ckpt_paths=ckpts, device="cpu")
    evaluation.evaluate(base, out=tmp_path / "plain", **kw)
    evaluation.evaluate({**base, "infer": {**base["infer"], "series": "all"}}, out=tmp_path / "all", **kw)
    evaluation.evaluate({**base, "infer": {**base["infer"], "tta": "mirror"}}, out=tmp_path / "tta", **kw)
    plain, every, tta = (pd.read_csv(tmp_path / d / "logits.csv") for d in ("plain", "all", "tta"))
    # One candidate per slot: series="all" is the plain single view.
    np.testing.assert_array_equal(plain[LABELS].to_numpy(), every[LABELS].to_numpy())
    assert np.isfinite(tta[LABELS].to_numpy()).all() and not np.allclose(tta[LABELS], plain[LABELS])
    timing = json.loads((tmp_path / "tta" / "timing.json").read_text())
    assert [v[:2] for v in timing["gold"]["per_rank"][0]["views"]] == [[0, False], [0, True]]


def test_submission_follows_sample_order(tmp_path):
    data, labels, train_meta, test_meta, caches, ckpts = _write_fixture(tmp_path)
    # Test DICOMs are not in the fixture: stand in cached volumes for the DICOM decoder.
    paths = evaluation.cache_paths(caches)
    original = evaluation.DicomVolumes.__call__
    evaluation.DicomVolumes.__call__ = lambda self, study, series: np.load(paths[series])
    try:
        for t in ("t0", "t1"):
            series_dir = data / "test_series" / t / f"{t}.s"
            series_dir.mkdir(parents=True)
            for i in range(8):
                (series_dir / f"{i}.dcm").touch()
        pd.DataFrame({"StudyInstanceUID": ["t1", "t0"], **{c: 0.5 for c in LABELS}}).to_csv(
            data / "sample_submission.csv", index=False)
        config = {"ckpts": [{"file": "f0.pt"}, {"file": "f1.pt"}], "infer": {"k_eval": 4}}
        pred = evaluation.submission(config, data=data, ckpt_paths=ckpts, out=tmp_path / "sub", device="cpu")
    finally:
        evaluation.DicomVolumes.__call__ = original
    sub = pd.read_csv(tmp_path / "sub" / "submission.csv")
    assert sub.StudyInstanceUID.tolist() == ["t1", "t0"] and list(sub.columns) == ["StudyInstanceUID", *LABELS]
    assert ((sub[LABELS] >= 0) & (sub[LABELS] <= 1)).all().all() and len(pred) == 2


def test_batch_without_readable_series_scores_one_half(tmp_path):
    data, labels, train_meta, test_meta, caches, ckpts = _write_fixture(tmp_path)
    config = {"ckpts": [{"file": "f0.pt"}], "sets": ["test"], "bootstrap": 10, "infer": {"k_eval": 4}}
    evaluation.evaluate(config, data=data, labels_path=tmp_path / "labels.csv", train_meta=train_meta,
                        test_meta=test_meta.iloc[:0], cache_dirs=caches, ckpt_paths=ckpts[:1],
                        out=tmp_path / "empty", device="cpu")
    preds = pd.read_csv(tmp_path / "empty" / "preds_test.csv", index_col=0)
    assert (preds[LABELS] == 0.5).all().all()


def test_gpu_mirror_equals_volume_mirror(tmp_path):
    data, labels, train_meta, test_meta, caches, ckpts = _write_fixture(tmp_path)
    se = pd.read_csv(data / "train_series.csv")
    table = evaluation.build_study_table(se, dict(zip(train_meta.series, train_meta.n)))
    studies = sorted(table)
    volumes = evaluation.CacheVolumes(evaluation.cache_paths(caches))
    # Window centres are symmetric for the k used in practice (12, 24) at every slice count
    # 3..64; banker's rounding of x.5 centres breaks symmetry for some small k (e.g. 5).
    for k in (12, 24):
        for i in range(len(studies)):
            x, mask, pos = (t.unsqueeze(0) for t in evaluation.ViewDataset(studies, table, k, volumes)[i])
            want = [t.unsqueeze(0) for t in evaluation.ViewDataset(studies, table, k, volumes, mirror=True)[i]]
            got = evaluation.mirror_batch(x, mask, pos)
            # Same tokens up to order: compare each window as a (content, position) pair, sorted by position.
            for s in range(x.shape[1]):
                if not mask[0, s]:
                    continue
                order_w, order_g = torch.argsort(want[2][0, s]), torch.argsort(got[2][0, s])
                assert torch.equal(want[0][0, s][order_w], got[0][0, s][order_g])
                assert torch.allclose(want[2][0, s][order_w], got[2][0, s][order_g])


def test_gpu_and_volume_mirror_give_the_same_predictions(tmp_path):
    data, labels, train_meta, test_meta, caches, ckpts = _write_fixture(tmp_path)
    kw = dict(data=data, labels_path=tmp_path / "labels.csv", train_meta=train_meta, test_meta=test_meta,
              cache_dirs=caches, ckpt_paths=ckpts, device="cpu")
    base = {"ckpts": [{"file": "f0.pt"}, {"file": "f1.pt"}], "sets": ["gold"], "bootstrap": 10,
            "infer": {"k_eval": 12, "tta": "mirror"}}
    evaluation.evaluate(base, out=tmp_path / "volume", **kw)
    evaluation.evaluate({**base, "infer": {**base["infer"], "mirror_mode": "gpu"}}, out=tmp_path / "gpu", **kw)
    a, b = (pd.read_csv(tmp_path / d / "logits.csv")[LABELS].to_numpy() for d in ("volume", "gpu"))
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


def test_window_centres_are_mirror_symmetric_for_used_k():
    from rsna_knee.reader.knee import window_centres

    for k in (12, 24):
        for n in range(3, 65):
            c = window_centres(n, k, False)
            assert sorted(c) == sorted(n - 1 - c)
