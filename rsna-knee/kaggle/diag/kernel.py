"""Time each stage of a training session separately, saving results after every stage.

Stages: input discovery, symlink farm, internet + pretrained download (subprocess with
timeout), cache read throughput (full reads and training-like window reads, 1 and 4
threads), GPU forward/backward step time. Finishes within an internal budget.
"""
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

OUT = Path("/kaggle/working/diag.json")
RESULT = {"started": time.time(), "stages": {}}
BUDGET = 25 * 60


def save():
    OUT.write_text(json.dumps(RESULT, indent=2, default=str))


def stage(name, fn, timeout):
    box = {}

    def target():
        try:
            box["value"] = fn()
        except Exception as error:
            box["error"] = f"{type(error).__name__}: {error}"

    start = time.time()
    if time.time() - RESULT["started"] > BUDGET:
        RESULT["stages"][name] = {"skipped": "budget"}
        save()
        return None
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    RESULT["stages"][name] = {"seconds": time.time() - start, "timed_out": thread.is_alive(), **box}
    print(name, json.dumps(RESULT["stages"][name], default=str)[:500], flush=True)
    save()
    return box.get("value")


def inputs():
    from rsna_knee.evaluation import cache_paths, kaggle_inputs

    found = kaggle_inputs()
    t = time.time()
    paths = cache_paths([d for d in found["cache_dirs"] if d.name == "cache_train"])
    return {"series": len(paths), "cache_paths_seconds": time.time() - t, "paths": paths}


info = stage("inputs", inputs, 300) or {}
paths = info.pop("paths", {})
RESULT["stages"]["inputs"].get("value", {}).pop("paths", None)
save()


def symlinks():
    farm = Path("/tmp/diag_links")
    farm.mkdir(exist_ok=True)
    for series, path in paths.items():
        os.symlink(path, farm / f"{series}.npy")
    return len(paths)


stage("symlinks", symlinks, 300)


def internet():
    import urllib.request

    with urllib.request.urlopen("https://huggingface.co", timeout=15) as r:
        return r.status


stage("internet", internet, 30)


def pretrained():
    code = "import timm; timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=True, num_classes=0)"
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    return {"code": p.returncode, "stderr": p.stderr[-800:]}


stage("pretrained_download", pretrained, 320)

rng = np.random.default_rng(0)
sample = [paths[s] for s in rng.choice(sorted(paths), 240, replace=False)] if paths else []


def full_read(files):
    total = 0
    for f in files:
        total += np.load(f).nbytes
    return total


def window_read(files):
    total = 0
    for f in files:
        vol = np.load(f, mmap_mode="r")
        idx = np.sort(rng.choice(len(vol), min(24, len(vol)), replace=False))
        total += np.asarray(vol[idx]).nbytes
    return total


def throughput(fn, files, threads):
    t = time.time()
    if threads == 1:
        total = fn(files)
    else:
        with ThreadPoolExecutor(threads) as pool:
            total = sum(pool.map(fn, [files[i::threads] for i in range(threads)]))
    secs = time.time() - t
    return {"files": len(files), "mb": total / 1e6, "seconds": secs, "mb_per_s": total / 1e6 / secs}


stage("read_full_1thread", lambda: throughput(full_read, sample[:40], 1), 240)
stage("read_window_1thread", lambda: throughput(window_read, sample[40:80], 1), 240)
stage("read_full_4threads", lambda: throughput(full_read, sample[80:160], 4), 240)
stage("read_window_4threads", lambda: throughput(window_read, sample[160:240], 4), 240)


def gpu_steps():
    import torch

    from rsna_knee.reader.knee import KneeNet

    out = {"cuda": torch.cuda.is_available(), "gpus": [torch.cuda.get_device_name(i)
                                                        for i in range(torch.cuda.device_count())]}
    model = KneeNet("convnext_tiny", pretrained=False).cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randint(0, 255, (8, 6, 8, 3, 384, 384), dtype=torch.uint8).cuda()
    mask, pos = torch.ones(8, 6, dtype=torch.bool).cuda(), torch.rand(8, 6, 8).cuda()
    y = torch.rand(8, 12).cuda()
    for amp in ("fp16", "none"):
        times = []
        for i in range(8 if amp == "fp16" else 4):
            torch.cuda.synchronize()
            t = time.time()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp == "fp16"):
                logit = model(x, mask, pos, res=256, train=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            times.append(time.time() - t)
        out[f"{amp}_step_seconds"] = times
    out["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return out


stage("gpu_train_step_bs8_k8", gpu_steps, 400)
stage("nvidia_smi", lambda: subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout, 40)
RESULT["total_seconds"] = time.time() - RESULT["started"]
save()
