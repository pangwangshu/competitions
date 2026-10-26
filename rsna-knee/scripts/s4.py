"""S4: training speed test on Kaggle T4s, then the fold-0 reproduction R0.

  python scripts/s4.py configs                 # experiments/S4-speed/config.json
  python scripts/s4.py speed <pulled_dir> --gpu-hours H --kernel-version V
  python scripts/s4.py r0-config               # experiments/S4-R0/config.json (R0 + S5 T1)
  python scripts/s4.py r0 <pulled_dir> --gpu-hours H --kernel-version V
"""

import argparse
import json
import math
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsna_knee import evaluation  # noqa: E402
from rsna_knee.reader.knee import LABELS  # noqa: E402
from scripts.s2 import base_config  # noqa: E402

EXP = ROOT / "experiments"
# The author's checkpoint args: train.py defaults except this pretrained tag.
AUTHOR_ARGS = {"fold": 0, "labels": "labels_v0", "backbone": "convnext_tiny.fb_in22k_ft_in1k", "seed": 0}
SPEED_LIMIT = 512  # training studies per speed job: 64 steps at bs 8, enough to average out warm-up


def speed_job(name, gpu, **args):
    return {"name": name, "gpu": gpu, "keep_ckpt": False,
            "args": {**AUTHOR_ARGS, "limit": SPEED_LIMIT, "epochs": 1, **args}}


def write_configs():
    pins = {k: base_config()[k] for k in ("labels_sha256", "meta_sha256")}
    # v1 (no checkpointing): every job ran out of T4 memory, bs 8 at once and bs 4 after ~100 s.
    speed = {"name": "S4-speed", **pins, "max_hours": 2.0, "rounds": [
        [speed_job("fp16-bs8-gc-w2", 0, amp="fp16", bs=8, workers=2, grad_ckpt=1),
         speed_job("fp16-bs8-gc-w4", 1, amp="fp16", bs=8, workers=4, grad_ckpt=1)],
        [speed_job("fp16-bs4-gc-w2", 0, amp="fp16", bs=4, workers=2, grad_ckpt=1),
         speed_job("fp32-bs8-gc-w2", 1, amp="none", bs=8, workers=2, grad_ckpt=1)],
    ]}
    path = EXP / "S4-speed" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluation.write_json(path, speed)
    print(path)


FULL_TRAIN, EPOCHS, G_TRAIN_HOURS = 3479, 14, 10.0  # fold-0 training studies, author epochs, acceptance gate
LABELS_CSV = EXP / "S1" / "labels" / "labels_v0.csv"


def speed(pulled, gpu_hours, kernel_version):
    """Extrapolate each speed job to a full fold: setup + 14 x (steps x step time + epoch evaluation)."""
    pulled, exp = Path(pulled), EXP / "S4-speed"
    for name in ("jobs.json", "session.json", "gpu_util.csv"):
        shutil.copy2(next(pulled.rglob(name)), exp / name)
    for log in pulled.rglob("*.stdout.txt"):
        shutil.copy2(log, exp / log.name)
    jobs = json.loads((exp / "jobs.json").read_text())
    rows = []
    for job in jobs:
        if job.get("error") or job.get("return_code") or not job.get("epochs"):
            rows.append({"name": job["name"], "failed": job.get("error") or job.get("return_code")})
            continue
        bs, (train, val, gold) = job["args"]["bs"], job["train_val_gold"]
        ep = job["epochs"][0]
        step = ep["train_seconds"] / (train // bs)
        epoch = step * (FULL_TRAIN // bs) + ep["eval_seconds"]
        hours = (job["setup_seconds"] + EPOCHS * epoch) / 3600
        rows.append({"name": job["name"], "amp": job["args"]["amp"], "bs": bs, "workers": job["args"]["workers"],
                     "gpu": job["gpu"], "steps": train // bs, "seconds_per_step": step,
                     "seconds_per_train_study": step / bs, "eval_seconds": ep["eval_seconds"],
                     "setup_seconds": job["setup_seconds"], "epoch_seconds_full": epoch,
                     "fold_hours_full": hours, "gpu_util_mean": (job.get("gpu_util") or {}).get("mean_util"),
                     "gpu_memory_mib": (job.get("gpu_util") or {}).get("max_memory_mib"),
                     "loss": ep["loss"], "gold_after_limit_epoch": ep["gold"]})
    table = pd.DataFrame(rows)
    table.to_csv(exp / "speed.csv", index=False)
    ok = table[table.get("fold_hours_full").notna()] if "fold_hours_full" in table else table.iloc[0:0]
    best = ok.sort_values("fold_hours_full").iloc[0].to_dict()
    decision = {"chosen": best, "g_train_passed": best["fold_hours_full"] <= G_TRAIN_HOURS,
                "gate_hours": G_TRAIN_HOURS, "full_fold": {"train_studies": FULL_TRAIN, "epochs": EPOCHS},
                "gpu_hours_session": gpu_hours,
                "fold_runs_per_gpu_hour": 1 / best["fold_hours_full"],
                "note": "Jobs ran in pairs, one per T4, sharing 4 vCPUs as R0 and T1 will. "
                        "Step time includes warm-up; per-epoch evaluation is the full 870 val + 58 gold."}
    evaluation.write_json(exp / "decision.json", decision)
    entry = {"id": "S4-speed", "date": date.today().isoformat(),
             "config_diff": "training speed test: fold 0, author hyperparameters, limit 512, 1 epoch; fp16/fp32, bs 8/4, workers 2/4, paired on two GPUs",
             "kernel": "wangshu/rsna-knee-train", "kernel_version": kernel_version, "m_oof": None, "m_gold": None,
             "public_score": None, "gpu_hours": round(gpu_hours, 3),
             "conclusion": (f"{'G-train passed' if decision['g_train_passed'] else 'G-train failed'}: fastest {best['name']} "
                            f"{best['seconds_per_step']:.2f}s/step, per-epoch evaluation {best['eval_seconds']:.0f}s, "
                            f"one fold extrapolated to {best['fold_hours_full']:.2f}h; " +
                            "; ".join(f"{r['name']} {r['fold_hours_full']:.2f}h" for r in rows
                                     if r.get("fold_hours_full") is not None and r["name"] != best["name"]))}
    evaluation.write_json(exp / "metrics.json", entry)
    print(table.to_string())
    print(json.dumps(entry, ensure_ascii=False, indent=1))
    return decision


def write_r0_config():
    decision = json.loads((EXP / "S4-speed" / "decision.json").read_text())
    if not decision["g_train_passed"]:
        raise ValueError("G-train failed: apply the predefined downgrade steps before R0")
    best = decision["chosen"]
    args = {**AUTHOR_ARGS, "amp": best["amp"], "bs": int(best["bs"]), "workers": int(best["workers"]),
            "grad_ckpt": 1}
    pins = {k: base_config()[k] for k in ("labels_sha256", "meta_sha256")}
    config = {"name": "S4-R0", **pins, "max_hours": 11.5, "rounds": [[
        {"name": "S4-R0", "gpu": 0, "keep_ckpt": True, "args": args},
        {"name": "S5-T1", "gpu": 1, "keep_ckpt": True, "args": {**args, "mirror_p": 0.5}},
    ]]}
    path = EXP / "S4-R0" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluation.write_json(path, config)
    print(json.dumps(config, indent=1))


def author_fold0():
    """The author's fold-0 checkpoint alone, from the E-base logits: the R0 reference."""
    exp = EXP / "S4-author-fold0"
    exp.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS_CSV, index_col=0)
    logits = pd.read_csv(EXP / "S2-E-base" / "logits.csv")
    ckpt0 = logits[logits.ckpt == 0]
    weak = ckpt0[ckpt0.task == "weak"].set_index("StudyInstanceUID")
    weak = weak.loc[labels.index[labels.fold == 0].intersection(weak.index)]
    gold = ckpt0[ckpt0.task == "gold"].set_index("StudyInstanceUID")
    for name, frame in [("oof", weak), ("gold", gold)]:
        pred = 1 / (1 + np.exp(-frame[LABELS].astype(np.float64)))
        if name == "oof":
            pred.insert(0, "fold", 0)
        pred.sort_index().to_csv(exp / f"preds_{name}.csv", float_format="%.9g")
    return exp


def run_predictions(oof_csv, exp, fold=0):
    """train.py's oof_fold<k>.csv (EMA logits for val + gold) -> evaluation-style prediction files."""
    frame = pd.read_csv(oof_csv, index_col=0)
    for name, set_name in [("oof", "val"), ("gold", "gold")]:
        part = frame[frame.set == set_name]
        pred = 1 / (1 + np.exp(-part[LABELS].astype(np.float64)))
        if name == "oof":
            pred.insert(0, "fold", fold)
        pred.index.name = "StudyInstanceUID"
        pred.sort_index().to_csv(exp / f"preds_{name}.csv", float_format="%.9g")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["configs", "speed", "r0-config", "author-fold0"])
    ap.add_argument("pulled", nargs="?", type=Path)
    ap.add_argument("--gpu-hours", type=float)
    ap.add_argument("--kernel-version", type=int)
    args = ap.parse_args()
    if args.command == "configs":
        write_configs()
    elif args.command == "speed":
        speed(args.pulled, args.gpu_hours, args.kernel_version)
    elif args.command == "r0-config":
        write_r0_config()
    else:
        print(author_fold0())
