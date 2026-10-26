import hashlib

import numpy as np
import pandas as pd

from rsna_knee.reader import cache, preprocess


def test_default_cache_workers_preserve_inference_physical_spacing(monkeypatch):
    original = preprocess.MM_PER_PX
    monkeypatch.setattr(preprocess, "SIZE", preprocess.SIZE)
    monkeypatch.setattr(preprocess, "MM_PER_PX", original)
    cache._init_worker(384)
    assert preprocess.SIZE == 384 and preprocess.MM_PER_PX == original


def test_partition_keeps_a_study_together_and_covers_every_series():
    studies = pd.Series([f"study-{i // 6}" for i in range(4407 * 6)])
    assigned = studies.map(lambda study: cache.shard_of(study, 8))
    assert set(assigned) == set(range(8))
    assert assigned.groupby(studies).nunique().eq(1).all()
    assert cache.shard_of("abc", 8) == int(hashlib.sha256(b"abc").hexdigest(), 16) % 8


def test_file_order_cannot_change_volume_with_tied_positions(monkeypatch):
    monkeypatch.setattr(preprocess, "SIZE", 8)
    images = {"a.dcm": np.arange(64, dtype=np.float32).reshape(8, 8),
              "b.dcm": np.arange(64, dtype=np.float32).reshape(8, 8) + 10,
              "c.dcm": np.arange(64, dtype=np.float32).reshape(8, 8) + 20}
    from pathlib import Path
    monkeypatch.setattr(preprocess, "_read", lambda path: [(images[Path(path).name], None, None, [0.4, 0.4], 0)])
    monkeypatch.setattr(preprocess.os, "listdir", lambda _: ["c.dcm", "a.dcm", "b.dcm"])
    first, info = preprocess.load_series("unused")
    monkeypatch.setattr(preprocess.os, "listdir", lambda _: ["b.dcm", "c.dcm", "a.dcm"])
    second, _ = preprocess.load_series("unused")
    np.testing.assert_array_equal(first, second)
    assert info["n"] == 3


def test_cache_formats_round_trip_and_unreadable_series_are_recorded(tmp_path, monkeypatch):
    volume = np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8)
    monkeypatch.setattr(preprocess, "load_series", lambda *_: (volume, {"n": 3, "n_raw": 4, "n_bad": 1}))
    rows = []
    for fmt in ["npy", "npz"]:
        row = cache._task(("unused", str(tmp_path), "train", "study", "series", "Sagittal", fmt, False))
        assert row["error"] == "" and row["n_bad"] == 1
        path = tmp_path / f"series.{fmt}"
        if fmt == "npy":
            restored = np.load(path, allow_pickle=False)
        else:
            with np.load(path, allow_pickle=False) as archive:
                restored = archive["volume"]
        np.testing.assert_array_equal(restored, volume)
        rows.append(row)
    assert rows[0]["bytes"] == (tmp_path / "series.npy").stat().st_size
    monkeypatch.setattr(preprocess, "load_series", lambda *_: (_ for _ in ()).throw(FileNotFoundError("missing")))
    row = cache._task(("unused", str(tmp_path), "train", "study", "missing", "Sagittal", "npy", False))
    assert row["error"].startswith("FileNotFoundError:")
    assert not (tmp_path / "missing.npy").exists()
