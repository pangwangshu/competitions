import json

import numpy as np
import pandas as pd
import pytest

from scripts.s1 import choose_plan, verify_test, verify_shard_provenance


def test_plan_prefers_original_npy_and_uses_all_measured_series(tmp_path):
    probe = tmp_path / "probe"
    data = tmp_path / "data"
    probe.mkdir()
    data.mkdir()
    pd.DataFrame({"StudyInstanceUID": [f"study-{i}" for i in range(24371)]}).to_csv(data / "train_series.csv", index=False)
    (probe / "probe.json").write_text(json.dumps({"train_series_count": 24371, "sample_count": 200,
                                                 "train_wall_seconds": 400}))
    pd.DataFrame({"error": [""] * 200, "n": [32] * 200, "bytes": [5_000_000] * 200,
                  "npz_bytes": [2_000_000] * 200}).to_csv(probe / "probe_series.csv", index=False)
    pd.DataFrame({"bytes": [5_000_000] * 15}).to_csv(probe / "probe_test.csv", index=False)
    result = choose_plan(probe, data)
    assert result["selection"]["format"] == "npy"
    assert result["selection"]["shards"] == 7
    assert sum(result["selection"]["shard_series_counts"].values()) == 24371
    # An undersharded raw cache cannot meet the output cap; compression can.
    result = choose_plan(probe, data, forced_shards=3)
    assert result["selection"]["format"] == "npz"


def test_byte_verification_rejects_cloud_volume_changes(tmp_path, monkeypatch):
    from scripts import s1
    monkeypatch.setattr(s1, "EXPERIMENT", tmp_path)
    local, cloud = tmp_path / "local", tmp_path / "cloud"
    local.mkdir()
    cloud.mkdir()
    frame = pd.DataFrame({"series": [f"series-{i}" for i in range(15)],
                          "study": [f"study-{i // 5}" for i in range(15)], "n": 32,
                          "bytes": 128 + 32 * 384 * 384, "sha256": [str(i) for i in range(15)]})
    frame.to_csv(local / "files_test.csv", index=False)
    frame.to_csv(cloud / "probe_test.csv", index=False)
    assert verify_test(local, cloud)["equal_count"] == 15
    frame.loc[0, "sha256"] = "modified"
    frame.to_csv(cloud / "probe_test.csv", index=False)
    with pytest.raises(ValueError, match="byte mismatch"):
        verify_test(local, cloud)


@pytest.mark.parametrize("change,error", [("source", "source provenance"), ("kernel", "kernel identity"),
                                           ("version", "version identity"), ("privacy", "private CPU")])
def test_provenance_rejects_mixed_source_or_kernel_outputs(change, error):
    selected = {"shards": 7, "format": "npy", "size": 384}
    publication = {"is_private": True, "version": 4, "manifest": {"archive_sha256": "source-v4"}}
    receipt = {"kernel": "wangshu/rsna-knee-cache-shard-0", "version": 1, "is_private": True,
               "enable_gpu": False, "code_archive_sha256": "source-v4", "code_dataset_version": 4,
               "config": {"shard": 0, "shards": 7, "fmt": "npy", "size": 384}}
    output = {"kernel": receipt["kernel"], "version": 1}
    status = {**output, "status": "COMPLETE"}
    verify_shard_provenance(0, selected, publication, receipt, output, status)
    if change == "source":
        receipt["code_archive_sha256"] = "older-source"
    elif change == "kernel":
        output["kernel"] = "wangshu/rsna-knee-cache-shard-1"
    elif change == "version":
        output["version"] = 2
    else:
        receipt["is_private"] = False
    with pytest.raises(ValueError, match=error):
        verify_shard_provenance(0, selected, publication, receipt, output, status)


def test_stable_group_folds_match_stable_sklearn_and_keep_groups_together():
    from sklearn.model_selection import GroupKFold

    from rsna_knee.reader.make_labels import stable_group_folds

    rng = np.random.default_rng(0)
    groups = rng.integers(0, 300, 1000).astype(str)
    folds = stable_group_folds(groups)
    assert pd.Series(folds).groupby(groups).nunique().eq(1).all()
    expected = np.zeros(len(groups), int)
    for k, (_, test) in enumerate(GroupKFold(5).split(groups, groups=groups)):
        expected[test] = k
    # Holds for scikit-learn versions that sort group sizes stably (1.9.1 here).
    np.testing.assert_array_equal(folds, expected)
