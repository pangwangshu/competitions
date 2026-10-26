"""Measured, study-sharded canonical caches for S1 (CPU only)."""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import preprocess

META_COLUMNS = ["series", "study", "n", "n_raw", "n_bad", "plane", "slice_gap", "error"]


def shard_of(study, shards):
    """Python's built-in hash is process-randomized; SHA256 is reproducible."""
    if shards < 1:
        raise ValueError("shards must be positive")
    return int(hashlib.sha256(str(study).encode()).hexdigest(), 16) % shards


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def environment():
    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "pandas", "pydicom", "scikit-learn"]}
    versions["opencv"] = cv2.__version__
    meminfo = Path("/proc/meminfo")
    return {"cpu_count": os.cpu_count(), "python": platform.python_version(),
            "versions": versions, "memory": meminfo.read_text() if meminfo.exists() else None,
            "disk": subprocess.check_output(["df", "-h"], text=True)}


def _init_worker(size):
    cv2.setNumThreads(1)
    preprocess.SIZE = size
    # Retain the reader's exact 384px float constant; division rounds differently.
    preprocess.MM_PER_PX = 0.4 if size == 384 else 153.6 / size


def _task(args):
    data, output, split, study, series, plane, fmt, probe = args
    row = {"series": series, "study": study, "error": ""}
    start = time.perf_counter()
    try:
        vol, info = preprocess.load_series(Path(data) / f"{split}_series" / study / series, plane)
        row.update(info)
        row["n"] = len(vol)
        row["load_seconds"] = time.perf_counter() - start
        destination = Path(output) / f"{series}.{fmt}"
        temp = destination.with_name(destination.name + ".tmp")
        save_start = time.perf_counter()
        with temp.open("wb") as handle:
            if fmt == "npy":
                np.save(handle, vol, allow_pickle=False)
            else:
                np.savez_compressed(handle, volume=vol)
        os.replace(temp, destination)
        row["save_seconds"] = time.perf_counter() - save_start
        row["bytes"] = destination.stat().st_size
        row["sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        if probe:
            compressed = destination.with_suffix(".npz")
            save_start = time.perf_counter()
            np.savez_compressed(compressed, volume=vol)
            row["npz_save_seconds"] = time.perf_counter() - save_start
            row["npz_bytes"] = compressed.stat().st_size
            destination.unlink()
            compressed.unlink()
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"[:300]
    row["seconds"] = time.perf_counter() - start
    return row


def process_series(data, output, split, series, *, fmt="npy", size=384, workers=None, probe=False):
    if fmt not in {"npy", "npz"} or size not in {384, 288}:
        raise ValueError("Supported caches: 384/288 pixels, npy/npz")
    workers = workers or os.cpu_count() or 1
    Path(output).mkdir(parents=True, exist_ok=True)
    tasks = [(str(data), str(output), split, str(r.StudyInstanceUID), str(r.SeriesInstanceUID),
              r.Anatomical_Plane, fmt, probe) for r in series.itertuples(index=False)]
    rows = []
    start = time.perf_counter()
    with Pool(workers, initializer=_init_worker, initargs=(size,)) as pool:
        for row in pool.imap_unordered(_task, tasks, chunksize=1):
            rows.append(row)
            if len(rows) % 100 == 0 or len(rows) == len(tasks):
                print(f"{split}: {len(rows)}/{len(tasks)} in {time.perf_counter() - start:.1f}s", flush=True)
    frame = pd.DataFrame(rows).sort_values("series").reset_index(drop=True)
    # Keep the author's exact metadata schema even when there are no errors/gaps.
    for column in META_COLUMNS:
        if column not in frame:
            frame[column] = np.nan if column != "error" else ""
    return frame, time.perf_counter() - start


def run_probe(data, output, count=200, workers=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = environment()
    print(json.dumps(env, indent=2), flush=True)
    train = pd.read_csv(Path(data) / "train_series.csv")
    test = pd.read_csv(Path(data) / "test_series.csv")
    sample = train.sample(count, random_state=0)
    rows, elapsed = process_series(data, output / "scratch", "train", sample, workers=workers, probe=True)
    rows.to_csv(output / "probe_series.csv", index=False)
    shutil.rmtree(output / "scratch")
    tests, test_elapsed = process_series(data, output / "cache_test", "test", test, workers=workers)
    tests.to_csv(output / "probe_test.csv", index=False)
    valid = rows[rows.error.fillna("").eq("")]
    report = {"environment": env, "size": 384, "mm_per_px": 0.4, "random_state": 0,
              "sample_count": count, "train_series_count": len(train), "test_series_count": len(test),
              "workers": workers or os.cpu_count() or 1, "train_wall_seconds": elapsed,
              "test_wall_seconds": test_elapsed, "failed_count": int((rows.error.fillna("").ne("") | rows.n.fillna(0).lt(3)).sum()),
              "mean_npy_bytes": float(valid.bytes.mean()), "mean_npz_bytes": float(valid.npz_bytes.mean()),
              "compression_ratio": float(valid.npz_bytes.sum() / valid.bytes.sum()),
              "mean_load_seconds": float(valid.load_seconds.mean()),
              "mean_npy_save_seconds": float(valid.save_seconds.mean()),
              "mean_npz_save_seconds": float(valid.npz_save_seconds.mean())}
    write_json(output / "probe.json", report)
    print(json.dumps(report, indent=2), flush=True)


def build_shard(data, output, shard, shards, fmt="npy", size=384, workers=None):
    if not 0 <= shard < shards:
        raise ValueError("shard must be in [0, shards)")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(Path(data) / "train_series.csv")
    selected = train[train.StudyInstanceUID.map(lambda s: shard_of(s, shards)).eq(shard)]
    env = environment()
    write_json(output / f"environment_shard{shard}.json", env)
    rows, elapsed = process_series(data, output / "cache_train", "train", selected,
                                   fmt=fmt, size=size, workers=workers)
    rows[META_COLUMNS].to_csv(output / f"meta_shard{shard}.csv", index=False)
    rows.to_csv(output / f"files_shard{shard}.csv", index=False)
    test_rows = None
    if shard == 0:
        test = pd.read_csv(Path(data) / "test_series.csv")
        test_rows, test_elapsed = process_series(data, output / "cache_test", "test", test,
                                                fmt=fmt, size=size, workers=workers)
        test_rows[META_COLUMNS].to_csv(output / "meta_test.csv", index=False)
        test_rows.to_csv(output / "files_test.csv", index=False)
        elapsed += test_elapsed
    failed = rows.error.fillna("").ne("") | rows.n.fillna(0).lt(3)
    rows.loc[failed].to_csv(output / f"failures_shard{shard}.csv", index=False)
    total = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    report = {"shard": shard, "shards": shards, "partition": "int(SHA256(StudyInstanceUID), 16) % N",
              "format": fmt, "size": size, "mm_per_px": 0.4 if size == 384 else 153.6 / size,
              "series_count": len(rows), "study_count": int(selected.StudyInstanceUID.nunique()),
              "failed_count": int(failed.sum()), "wall_seconds": elapsed, "output_bytes": total,
              "test_series_count": 0 if test_rows is None else len(test_rows),
              "test_failed_count": 0 if test_rows is None else int((test_rows.error.fillna("").ne("") | test_rows.n.fillna(0).lt(3)).sum())}
    write_json(output / f"shard{shard}.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if total > 19_000_000_000:
        raise RuntimeError("Shard exceeds the S1 19GB output limit")
    if elapsed > 10 * 3600:
        raise RuntimeError("Shard exceeds the S1 10 hour runtime limit")
