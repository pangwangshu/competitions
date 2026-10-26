import csv
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.log_experiment import record


def test_registry_is_idempotent_and_updates_require_replace(tmp_path):
    metrics = tmp_path / "trial" / "metrics.json"
    metrics.parent.mkdir()
    values = {"date": "2026-10-08", "config_diff": "k=16 | TTA", "kernel": "wangshu/trial",
              "kernel_version": 2, "m_oof": {"auc": 0.9, "ci": [0.88, 0.92]},
              "gpu_hours": 0.5, "conclusion": "pending"}
    metrics.write_text(json.dumps(values))
    output = tmp_path / "experiments"
    record(metrics, output)
    record(metrics, output)
    with (output / "results.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 1
    assert "k=16 \\| TTA" in (output / "log.md").read_text()
    values["conclusion"] = "adopt"
    metrics.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="--replace"):
        record(metrics, output)
    record(metrics, output, replace=True)
    assert "adopt" in (output / "log.md").read_text()


def test_invalid_metrics_do_not_create_a_registry(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"date": "2026-10-08", "config_diff": "test", "conclusion": "passed",
                                  "gpu_hours": 0, "m_gold": float("nan")}))
    output = tmp_path / "experiments"
    with pytest.raises(ValueError, match="Invalid metric"):
        record(metrics, output)
    assert not output.exists()


def test_concurrent_records_do_not_lose_experiments(tmp_path):
    paths = []
    for i in range(12):
        metrics = tmp_path / f"trial-{i}" / "metrics.json"
        metrics.parent.mkdir()
        metrics.write_text(json.dumps({"date": "2026-10-08", "config_diff": f"k={i}",
                                       "gpu_hours": 0, "conclusion": "passed"}))
        paths.append(metrics)
    output = tmp_path / "experiments"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda path: record(path, output), paths))
    with (output / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12 and len({r["id"] for r in rows}) == 12
