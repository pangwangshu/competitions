"""Choose S1 cache shards from measured data and verify the resulting artifacts."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.reader.cache import META_COLUMNS, process_series, shard_of
from rsna_knee.reader.make_labels import LABELS, report_group
from rsna_knee.metrics import macro_auc
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kx import ROOT, read_json, write_json

EXPERIMENT = ROOT / "experiments" / "S1"
SHARD_SOURCE = '''"""Build one deterministic S1 cache shard using all available CPU cores."""
import json
import os

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"

from rsna_knee.reader.cache import build_shard
from rsna_knee.reader.paths import DATA, WORK

config = json.loads(os.environ["RK_CONFIG_JSON"])
build_shard(DATA, WORK, **config)
'''


def choose_plan(probe_dir, data, *, forced_shards=None):
    report = read_json(probe_dir / "probe.json")
    sample = pd.read_csv(probe_dir / "probe_series.csv")
    series = pd.read_csv(data / "train_series.csv")
    assert len(series) == report["train_series_count"]
    valid = sample[sample.error.fillna("").eq("") & sample.n.ge(3)]
    if len(valid) < 0.99 * len(sample):
        raise ValueError("Probe has too many failed series; diagnose before planning")
    rng = np.random.RandomState(0)
    candidates = []
    for fmt, byte_column in [("npy", "bytes"), ("npz", "npz_bytes")]:
        mean = float(valid[byte_column].mean())
        boot = rng.choice(valid[byte_column].values, size=(2000, len(valid)), replace=True).mean(axis=1)
        upper = float(np.quantile(boot, 0.975))
        test_bytes = pd.read_csv(probe_dir / "probe_test.csv").bytes.sum()
        # Use the upper size estimate in N=ceil(total/18GB), retaining headroom.
        total = upper * len(series) + test_bytes
        shards = forced_shards or math.ceil(total / 18_000_000_000)
        counts = series.StudyInstanceUID.map(lambda study: shard_of(study, shards)).value_counts().sort_index()
        sizes = counts * upper
        sizes.loc[0] += test_bytes
        # Probe includes BOTH saves and hashing, giving a conservative runtime for either format.
        seconds_per_series = report["train_wall_seconds"] / report["sample_count"]
        hours = counts * seconds_per_series / 3600
        candidate = {"format": fmt, "size": 384, "mean_bytes": mean, "upper_95_mean_bytes": upper,
                     "estimated_total_bytes": mean * len(series) + test_bytes,
                     "upper_95_total_bytes": total, "shards": shards,
                     "estimated_max_shard_bytes": float(sizes.max()),
                     "estimated_max_shard_hours": float(hours.max()),
                     "passes": bool(sizes.max() <= 19_000_000_000 and hours.max() <= 10),
                     "shard_series_counts": {str(k): int(v) for k, v in counts.items()},
                     "shard_upper_bytes": {str(k): float(v) for k, v in sizes.items()}}
        candidates.append(candidate)
    selected = next((c for c in candidates if c["passes"]), None)
    if selected is None:
        raise ValueError("Neither 384px format fits; measure 288px or rerun with more shards")
    return {"selection": selected, "candidates": candidates, "probe_archive_sha256":
            hashlib.sha256((probe_dir / "probe.json").read_bytes()).hexdigest(),
            "partition": "int(SHA256(StudyInstanceUID), 16) % N", "gb_definition": "decimal, 1GB=10^9 bytes",
            "size_estimation": "bootstrap 2000 resamples, 97.5th percentile mean, ceil(total/18GB)",
            "runtime_estimation": "measured CPU-parallel probe wall time, including both saves and hashing"}


def provision(plan):
    selected = plan["selection"]
    for shard in range(selected["shards"]):
        directory = ROOT / "kaggle" / f"cache-shard-{shard}"
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "kernel.json", {"id": f"wangshu/rsna-knee-cache-shard-{shard}",
                   "title": f"RSNA Knee Cache Shard {shard}", "code_file": "kernel.py", "is_private": True,
                   "enable_gpu": False, "enable_internet": False, "dataset_sources": []})
        write_json(directory / "config.json", {"shard": shard, "shards": selected["shards"],
                                              "fmt": selected["format"], "size": selected["size"]})
        (directory / "kernel.py").write_text(SHARD_SOURCE)


def verify_test(local_dir, cloud_dir):
    local = pd.read_csv(local_dir / "files_test.csv").set_index("series")
    cloud_csv = cloud_dir / "files_test.csv"
    if not cloud_csv.exists():
        cloud_csv = cloud_dir / "probe_test.csv"
    cloud = pd.read_csv(cloud_csv).set_index("series")
    if set(local.index) != set(cloud.index) or len(local) != 15:
        raise ValueError("Expected the same 15 public test series")
    comparison = local[["study", "n", "bytes", "sha256"]].join(
        cloud[["bytes", "sha256"]].rename(columns={"bytes": "cloud_bytes", "sha256": "cloud_sha256"}))
    comparison["equal"] = comparison.sha256.eq(comparison.cloud_sha256) & comparison.bytes.eq(comparison.cloud_bytes)
    comparison.to_csv(EXPERIMENT / "test_byte_comparison.csv")
    result = {"series_count": 15, "study_count": int(comparison.study.nunique()),
              "equal_count": int(comparison.equal.sum()), "passed": bool(comparison.equal.all())}
    # Compare downloaded files directly too, when they are available (probe output).
    downloaded = cloud_dir / "cache_test"
    if downloaded.is_dir():
        selected = read_json(EXPERIMENT / "cache_plan.json")["selection"] if (EXPERIMENT / "cache_plan.json").exists() else {"format": "npy"}
        for series in comparison.index:
            expected = comparison.loc[series, "cloud_sha256"]
            if hashlib.sha256((downloaded / f"{series}.{selected['format']}").read_bytes()).hexdigest() != expected:
                raise ValueError(f"Downloaded test file changed: {series}")
        result["downloaded_files_checked"] = 15
    write_json(EXPERIMENT / "test_verification.json", result)
    if not result["passed"]:
        raise ValueError(f"Cloud/local byte mismatch: {result}")
    return result


def verify_all(data):
    selected = read_json(EXPERIMENT / "cache_plan.json")["selection"]
    publication = read_json(EXPERIMENT / "code-publication.json")
    local_dir = ROOT / "runs" / "s1-local-test"
    if selected["format"] != "npy" or selected["size"] != 384 or not (local_dir / "files_test.csv").exists():
        rows, _ = process_series(data, local_dir / "cache_test", "test", pd.read_csv(data / "test_series.csv"),
                                 fmt=selected["format"], size=selected["size"], workers=4)
        rows.to_csv(local_dir / "files_test.csv", index=False)
    frames, reports = [], []
    for shard in range(selected["shards"]):
        directory = EXPERIMENT / "shards" / str(shard)
        output = read_json(directory / "kx-output.json")
        status = read_json(directory / "status.json")
        receipt = read_json(directory / "kernel-publication.json")
        verify_shard_provenance(shard, selected, publication, receipt, output, status)
        if status["status"] != "COMPLETE" or status["version"] != output["version"]:
            raise ValueError(f"shard {shard} is not COMPLETE at the downloaded version")
        report = read_json(directory / f"shard{shard}.json")
        if (report["shard"], report["shards"], report["format"], report["size"]) != (
                shard, selected["shards"], selected["format"], selected["size"]):
            raise ValueError(f"shard {shard} configuration mismatch")
        if report["output_bytes"] > 19_000_000_000 or report["wall_seconds"] > 36000:
            raise ValueError(f"shard {shard} exceeds output/runtime limits")
        meta = pd.read_csv(directory / f"meta_shard{shard}.csv")
        if meta.columns.tolist() != META_COLUMNS:
            raise ValueError(f"shard {shard} metadata schema mismatch")
        files = pd.read_csv(directory / f"files_shard{shard}.csv")
        if not files.series.is_unique or len(files) != len(meta):
            raise ValueError(f"shard {shard} file inventory mismatch")
        valid = files.error.fillna("").eq("") & files.n.ge(3)
        if not (files.loc[valid, "bytes"].gt(0).all() and
                files.loc[valid, "sha256"].str.fullmatch(r"[0-9a-f]{64}").fillna(False).all()):
            raise ValueError(f"shard {shard} successful cache lacks bytes/SHA256")
        if not meta.study.map(lambda study: shard_of(study, selected["shards"])).eq(shard).all():
            raise ValueError(f"shard {shard} contains studies assigned elsewhere")
        meta = meta.merge(files[["series", "bytes", "sha256"]], on="series", validate="one_to_one")
        meta = meta.assign(split="train", shard=shard, kernel=output["kernel"], kernel_version=output["version"],
                           format=selected["format"], size=selected["size"])
        frames.append(meta)
        reports.append(report)
    train = pd.concat(frames, ignore_index=True)
    expected = pd.read_csv(data / "train_series.csv").rename(columns={"SeriesInstanceUID": "series", "StudyInstanceUID": "study"})
    if len(train) != 24371 or not train.series.is_unique or set(train.series) != set(expected.series):
        raise ValueError("Manifest must contain all 24,371 training series exactly once")
    identity = train.merge(expected[["series", "study"]], on="series", suffixes=("", "_expected"), validate="one_to_one")
    if not identity.study.eq(identity.study_expected).all():
        raise ValueError("Manifest series/study association mismatch")
    failed = train.error.fillna("").ne("") | train.n.fillna(0).lt(3)
    train.loc[failed].to_csv(EXPERIMENT / "failures.csv", index=False)
    train[META_COLUMNS].to_csv(EXPERIMENT / "cache384_train_meta.csv", index=False)
    directory = EXPERIMENT / "shards" / "0"
    test = pd.read_csv(directory / "files_test.csv")
    expected_test = pd.read_csv(data / "test_series.csv")
    if len(test) != 15 or set(test.series) != set(expected_test.SeriesInstanceUID):
        raise ValueError("Expected all 15 test series in shard 0")
    if not (test.error.fillna("").eq("") & test.n.ge(3)).all():
        raise ValueError("Test cache contains failed series")
    test = test.assign(split="test", shard=0, kernel=train.iloc[0].kernel if selected["shards"] == 1 else
                       "wangshu/rsna-knee-cache-shard-0", kernel_version=read_json(directory / "kx-output.json")["version"],
                       format=selected["format"], size=selected["size"])
    columns = ["split", *META_COLUMNS, "shard", "kernel", "kernel_version", "format", "size", "bytes", "sha256"]
    pd.concat([train[columns], test[columns]], ignore_index=True).to_csv(EXPERIMENT / "cache_manifest.csv", index=False)
    result = {"train_series": len(train), "test_series": len(test), "failed_series": int(failed.sum()),
              "failure_rate": float(failed.mean()), "failure_rate_passed": bool(failed.mean() <= 0.005),
              "shards_complete": len(reports), "shard_reports": reports,
              "test": verify_test(ROOT / "runs" / "s1-local-test", directory),
              "labels": verify_labels(data, EXPERIMENT / "labels")}
    label_checks = result["labels"]["actual_checks"]
    result["passed"] = result["failure_rate_passed"] and result["test"]["passed"] and all(
        v for k, v in label_checks.items() if k != "consensus_auc_in_range")
    write_json(EXPERIMENT / "verification.json", result)
    if not result["passed"]:
        raise ValueError("S1 verification failed; inspect verification.json")
    return result


def verify_shard_provenance(shard, selected, publication, receipt, output, status):
    expected = f"wangshu/rsna-knee-cache-shard-{shard}"
    if not (receipt["kernel"] == output["kernel"] == status["kernel"] == expected):
        raise ValueError(f"shard {shard} kernel identity mismatch")
    if not (receipt["version"] == output["version"] == status["version"]):
        raise ValueError(f"shard {shard} version identity mismatch")
    if receipt["is_private"] is not True or receipt["enable_gpu"] is not False or publication["is_private"] is not True:
        raise ValueError(f"shard {shard} must have private CPU provenance")
    if receipt["code_archive_sha256"] != publication["manifest"]["archive_sha256"] or receipt["code_dataset_version"] != publication["version"]:
        raise ValueError(f"shard {shard} source provenance mismatch")
    if receipt["config"] != {"shard": shard, "shards": selected["shards"], "fmt": selected["format"], "size": selected["size"]}:
        raise ValueError(f"shard {shard} published config mismatch")


def verify_labels(data, directory):
    report = read_json(directory / "labels_report.json")
    labels = pd.read_csv(directory / "labels_v0.csv", float_precision="round_trip").set_index("StudyInstanceUID")
    training = pd.read_csv(data / "train.csv", float_precision="round_trip").set_index("StudyInstanceUID")
    if len(labels) != 4407 or not labels.index.is_unique or set(labels.index) != set(training.index):
        raise ValueError("Label CSV must contain all 4,407 official study IDs exactly once")
    labels = labels.reindex(training.index)
    gold = training[LABELS].notna().all(axis=1)
    weak = labels.loc[~gold]
    groups = training.Report.map(report_group)
    group_folds = weak.assign(group=groups).groupby("group").fold.nunique()
    auc = macro_auc(training.loc[gold, LABELS].to_numpy(), labels.loc[gold, LABELS].to_numpy())
    checks = {"rows_4407": len(labels) == 4407, "unique_studies": labels.index.is_unique,
              "gold_58_fold_minus1": int(gold.sum()) == 58 and bool(labels.loc[gold, "fold"].eq(-1).all()),
              "is_gold_matches_official": bool(labels.is_gold.eq(gold.astype(int)).all()),
              "weak_4349": len(weak) == 4349,
              "five_balanced_folds": set(weak.fold) == set(range(5)) and bool(weak.fold.value_counts().between(865, 875).all()),
              "complete_soft_labels": bool(np.isfinite(labels[LABELS].to_numpy()).all()),
              "gold_targets_match_official": bool(np.array_equal(labels.loc[gold, [f"gold_{c}" for c in LABELS]].to_numpy(), training.loc[gold, LABELS].to_numpy())),
              "duplicate_reports_stay_together": bool(group_folds.eq(1).all()),
              "consensus_auc_in_range": abs(auc - 0.896) <= 0.005}
    if not all(v for k, v in checks.items() if k != "consensus_auc_in_range"):
        raise ValueError(f"Downloaded label CSV verification failed: {checks}")
    for name, expected_sha in report.get("input_sha256", {}).items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != expected_sha:
            raise ValueError(f"Local/cloud competition CSV mismatch: {name}")
    return {**report, "actual_checks": checks, "actual_consensus_macro_auc": auc}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--probe-dir", type=Path, default=EXPERIMENT / "probe")
    plan.add_argument("--shards", type=int, help="Override shard count after a timeout or oversize failure")
    test = commands.add_parser("verify-test")
    test.add_argument("--local-dir", type=Path, default=ROOT / "runs" / "s1-local-test")
    test.add_argument("--cloud-dir", type=Path, default=EXPERIMENT / "probe")
    commands.add_parser("verify")
    args = parser.parse_args()
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    if args.command == "plan":
        result = choose_plan(args.probe_dir, args.data, forced_shards=args.shards)
        write_json(EXPERIMENT / "cache_plan.json", result)
        provision(result)
    elif args.command == "verify-test":
        result = verify_test(args.local_dir, args.cloud_dir)
    else:
        result = verify_all(args.data)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
