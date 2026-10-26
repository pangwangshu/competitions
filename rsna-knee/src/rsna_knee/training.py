"""Run reader training jobs on the Kaggle GPUs (S4 onward).

Config:
  {"name": str, "labels_sha256": str, "meta_sha256": str, "max_hours": 11.5,
   "rounds": [[job, ...], ...], "extra_labels": {"labels_t4": {"file": ..., "sha256": ...}}}
  job = {"name": str, "gpu": int, "args": {train.py flag: value}, "keep_ckpt": true,
         "resume_state": "<file under /kaggle/input>" (optional)}
Jobs of one round run concurrently (one per GPU); the next round starts when all
have finished. Alternatively "queues": [[job, ...], ...] runs each list in order on
its own GPU, independently of the other GPUs. Each job is `python -m rsna_knee.reader.train --name <name> ...` with
RK_WORK pointing at a scratch directory that symlinks the attached S1 cache shards
and RK_RUNS at /kaggle/working/runs. Every stdout line is timestamped so training,
evaluation and setup time can be separated; nvidia-smi is sampled every 5 seconds.
Past max_hours, running jobs are stopped; train.py keeps a per-epoch state file
from which a later session resumes (resume_state).
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd

from .evaluation import cache_paths, find_one, meta_digest, sha256, write_json

EPOCH = re.compile(r"^ep\s+(\d+) loss ([\d.]+) time (\d+)s(?: \| gold ([\d.]+))?(?: \| val ([\d.]+))?")
HEADER = re.compile(r"^train (\d+)\s+val (\d+)\s+gold (\d+)")
STEP = re.compile(r"^\s*ep (\d+) step (\d+)/(\d+) loss [\d.]+ ([\d.]+)s/it")


def prepare_work(work, labels_path, train_meta, cache_dirs, labels_sha256=None, meta_sha256=None):
    """The layout train.py expects: labels_v0.csv, cache384_train_meta.csv, cache384_train/<series>.npy."""
    work = Path(work)
    if labels_sha256 and sha256(labels_path) != labels_sha256:
        raise ValueError("labels_v0.csv differs from the frozen S1 table")
    if meta_sha256 and meta_digest(train_meta) != meta_sha256:
        raise ValueError("Cache metadata differs from the S1 manifest")
    cache = work / "cache384_train"
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy2(labels_path, work / "labels_v0.csv")
    train_meta.to_csv(work / "cache384_train_meta.csv", index=False)
    paths = cache_paths(cache_dirs)
    missing = set(train_meta.series[train_meta.n.fillna(0) >= 3]) - set(paths)
    if missing:
        raise FileNotFoundError(f"{len(missing)} cached training series are missing")
    for series, path in paths.items():
        link = cache / f"{series}.npy"
        if not link.exists():
            os.symlink(path, link)
    return len(paths)


def flags(args):
    out = []
    for key, value in args.items():
        out += [f"--{key}", str(value)]
    return out


def parse_log(lines):
    """lines: [(seconds since start, text)] -> setup, per-epoch train/eval seconds and scores."""
    header = next(((t, m) for t, line in lines if (m := HEADER.match(line))), None)
    epochs, mark = [], header[0] if header else None
    steps = [(t, int(m[2]), float(m[4])) for t, line in lines if (m := STEP.match(line))]
    for t, line in lines:
        m = EPOCH.match(line)
        if not m:
            continue
        train_s = float(m[3])
        epochs.append({"epoch": int(m[1]), "loss": float(m[2]), "train_seconds": train_s,
                       "eval_seconds": None if mark is None else t - mark - train_s,
                       "gold": None if m[4] is None else float(m[4]),
                       "val": None if m[5] is None else float(m[5]), "at": t})
        mark = t
    return {"setup_seconds": header[0] if header else None,
            "train_val_gold": [int(x) for x in header[1].groups()] if header else None,
            "epochs": epochs, "step_reports": steps}


class GpuMonitor(threading.Thread):
    def __init__(self, interval=5.0):
        super().__init__(daemon=True)
        self.interval, self.rows, self.stopped = interval, [], threading.Event()

    def run(self):
        while not self.stopped.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
                now = time.time()
                for row in out.stdout.strip().splitlines():
                    gpu, util, mem = (int(v) for v in row.split(","))
                    self.rows.append({"time": now, "gpu": gpu, "util": util, "memory_mib": mem})
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stopped.wait(self.interval)


def run_job(job, env, runs, deadline, log_dir):
    name, args = job["name"], dict(job["args"])
    if job.get("resume_state"):
        fold = args.get("fold", 0)
        target = Path(runs) / name / f"state_fold{fold}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(find_one(job["resume_state"])[0], target)
    cmd = [sys.executable, "-m", "rsna_knee.reader.train", "--name", name, *flags(args)]
    env = dict(env, CUDA_VISIBLE_DEVICES=str(job["gpu"]))
    start = time.time()
    # Own process group: at the deadline the DataLoader workers die too and release stdout.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1,
                            start_new_session=True)
    killed = threading.Event()

    def watchdog():
        while proc.poll() is None:
            if time.time() > deadline:
                killed.set()
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(30)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                return
            time.sleep(10)

    threading.Thread(target=watchdog, daemon=True).start()
    lines = []
    with open(Path(log_dir) / f"{name}.stdout.txt", "w") as log:
        for line in proc.stdout:
            t = time.time() - start
            lines.append((t, line.rstrip("\n")))
            log.write(f"{t:9.1f} {line}")
            log.flush()
            print(f"[{name}] {line}", end="", flush=True)
    code = proc.wait()
    result = {"name": name, "gpu": job["gpu"], "args": args, "command": cmd, "return_code": code,
              "killed_at_deadline": killed.is_set(), "start": start, "wall_seconds": time.time() - start,
              **parse_log(lines)}
    if not job.get("keep_ckpt", True):
        for ckpt in (Path(runs) / name).glob("fold*.pt"):
            ckpt.unlink()
    return result


def gpu_summary(rows, result):
    inside = [r for r in rows if r["gpu"] == result["gpu"] and
              result["start"] <= r["time"] <= result["start"] + result["wall_seconds"]]
    if not inside:
        return None
    return {"samples": len(inside), "mean_util": sum(r["util"] for r in inside) / len(inside),
            "max_memory_mib": max(r["memory_mib"] for r in inside)}


def run_queues(queues, env, runs, deadline, logs, out):
    results, lock = [], threading.Lock()

    def target(jobs):
        for job in jobs:
            if time.time() > deadline:
                result = {"name": job["name"], "skipped": "deadline"}
            else:
                try:
                    result = run_job(job, env, runs, deadline, logs)
                except Exception as error:
                    result = {"name": job["name"], "error": f"{type(error).__name__}: {error}"}
            with lock:
                results.append(result)
                write_json(out / "jobs.json", results)

    for i, jobs in enumerate(queues):
        if any(j["gpu"] != i for j in jobs):
            raise ValueError("Queue i must run on GPU i")
    threads = [threading.Thread(target=target, args=(jobs,)) for jobs in queues]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def run_kaggle(config):
    from .evaluation import kaggle_inputs
    from .reader.paths import WORK

    session_start = time.time()
    deadline = session_start + 3600 * float(config.get("max_hours", 11.5))
    out = Path(WORK)
    runs, logs, work = out / "runs", out / "logs", Path("/tmp/rk_work")
    logs.mkdir(parents=True, exist_ok=True)
    inputs = kaggle_inputs()
    train_dirs = [d for d in inputs["cache_dirs"] if d.name == "cache_train"]
    linked = prepare_work(work, inputs["labels_path"], inputs["train_meta"], train_dirs,
                          config.get("labels_sha256"), config.get("meta_sha256"))
    print(f"linked {linked} cached series into {work}", flush=True)
    # Extra label tables (e.g. labels_t4 for distillation) come from an attached dataset, pinned by hash.
    for name, pin in config.get("extra_labels", {}).items():
        source = find_one(pin["file"])[0]
        if sha256(source) != pin["sha256"]:
            raise ValueError(f"{pin['file']} differs from its pinned SHA256")
        shutil.copy2(source, work / f"{name}.csv")
        print(f"staged label table {name} from {source}", flush=True)
    import timm  # fetch pretrained weights once, before concurrent jobs race for the cache

    for backbone in sorted({j["args"].get("backbone", "convnext_tiny")
                            for r in config.get("rounds", []) + config.get("queues", []) for j in r}):
        timm.create_model(backbone, pretrained=True, num_classes=0)
    env = dict(os.environ, RK_WORK=str(work), RK_RUNS=str(runs), PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               PYTHONPATH=str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monitor = GpuMonitor()
    monitor.start()
    results = []
    if "queues" in config:
        results = run_queues(config["queues"], env, runs, deadline, logs, out)
    for round_jobs in config.get("rounds", []):
        if len({j["gpu"] for j in round_jobs}) != len(round_jobs):
            raise ValueError("Jobs of one round must use different GPUs")
        if time.time() > deadline:
            results += [{"name": j["name"], "skipped": "deadline"} for j in round_jobs]
            continue
        found = [None] * len(round_jobs)

        def target(i, job):
            try:
                found[i] = run_job(job, env, runs, deadline, logs)
            except Exception as error:  # record and keep the other GPU's job
                found[i] = {"name": job["name"], "error": f"{type(error).__name__}: {error}"}

        threads = [threading.Thread(target=target, args=(i, j)) for i, j in enumerate(round_jobs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        results += found
        write_json(out / "jobs.json", results)
    monitor.stopped.set()
    for r in results:
        if "start" in r:
            r["gpu_util"] = gpu_summary(monitor.rows, r)
    pd.DataFrame(monitor.rows).to_csv(out / "gpu_util.csv", index=False)
    write_json(out / "jobs.json", results)
    write_json(out / "session.json", {"config": config, "session_seconds": time.time() - session_start,
                                      "labels_sha256": sha256(inputs["labels_path"]),
                                      "meta_sha256": meta_digest(inputs["train_meta"])})
    # A job stopped at the deadline resumes from its state file; only real failures fail the session.
    bad = [r["name"] for r in results if r.get("error") or (r.get("return_code", 0) != 0
                                                             and not r.get("killed_at_deadline"))]
    if bad:
        raise RuntimeError(f"Training jobs failed: {bad}")
