"""Train one fold of the study-level knee model.

Source: goodpjw2008/rsna-knee-2-5d-convnext-reader.

Usage: python -m rsna_knee.reader.train --name exp1 --fold 0 [--backbone convnext_tiny --res 256 ...]
  --fold k   : train on the other folds, validate on fold k (weak labels) and on the gold studies
  --fold -1  : train on all weak-labelled studies, validate on the gold studies only
Writes runs/<name>/fold<k>.pt, oof_fold<k>.csv (validation + gold predictions) and log_fold<k>.txt
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .knee import LABELS, KneeNet, StudyDataset, build_study_table
from .paths import DATA, RUNS, WORK
from .runtime import autocast_context, resolve_amp, resolve_device


def macro_auc(y, p):
    aucs = [roc_auc_score(y[:, i], p[:, i]) if 0 < y[:, i].sum() < len(y) else np.nan for i in range(y.shape[1])]
    return float(np.nanmean(aucs)), aucs


@torch.no_grad()
def predict(model, loader, res, device, amp):
    model.eval()
    out = []
    for x, mask, pos, _ in loader:
        x, mask, pos = x.to(device, non_blocking=device.type == "cuda"), mask.to(device), pos.to(device)
        with autocast_context(device, amp):
            out.append(model(x, mask, pos, res=res).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--labels", default="labels_v0")
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--k_eval", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only this many training studies")
    ap.add_argument("--eval_every", type=int, default=1,
                    help="evaluate the EMA model every N epochs (0: only after the last); training is unaffected")
    ap.add_argument("--grad_ckpt", type=int, default=0,
                    help="1: recompute encoder activations in backward (same maths, fits bs 8 on 16GB GPUs)")
    ap.add_argument("--mirror_p", type=float, default=0.0, help="probability of a left-right mirrored study")
    ap.add_argument("--device", choices=("cuda", "mps", "cpu"))
    ap.add_argument("--amp", choices=("bf16", "fp16", "none"))
    args = ap.parse_args()
    device = resolve_device(args.device)
    args.device = str(device)
    args.amp = resolve_amp(device, args.amp, training=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = f"{RUNS}/{args.name}"
    os.makedirs(out_dir, exist_ok=True)
    log = open(f"{out_dir}/log_fold{args.fold}.txt", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(json.dumps(vars(args)))
    lab = pd.read_csv(f"{WORK}/{args.labels}.csv", index_col=0)
    meta = pd.read_csv(f"{WORK}/cache384_train_meta.csv")
    table = build_study_table(pd.read_csv(f"{DATA}/train_series.csv"), dict(zip(meta.series, meta.n.fillna(0))))
    cache = f"{WORK}/cache384_train"

    gold = lab[lab.is_gold == 1]
    weak = lab[lab.is_gold == 0]
    tr = weak[weak.fold != args.fold]
    va = weak[weak.fold == args.fold]
    if args.limit:
        tr = tr.iloc[:args.limit]
    say(f"train {len(tr)}  val {len(va)}  gold {len(gold)}")

    dl_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_dl = torch.utils.data.DataLoader(
        StudyDataset(tr.index, table, cache, tr[LABELS].values.astype(np.float32), args.k, train=True,
                     mirror_p=args.mirror_p),
        batch_size=args.bs, shuffle=True, drop_last=True, **dl_kw)
    eval_sets = {"gold": gold}
    if len(va):
        eval_sets["val"] = va
    eval_dl = {k: torch.utils.data.DataLoader(StudyDataset(v.index, table, cache, None, args.k_eval),
                                              batch_size=args.bs, shuffle=False, **dl_kw) for k, v in eval_sets.items()}
    gold_y = gold[[f"gold_{c}" for c in LABELS]].values
    val_y = (va[LABELS].values > 0.5).astype(int) if len(va) else None

    model = KneeNet(args.backbone).to(device).to(memory_format=torch.channels_last)
    if args.grad_ckpt:
        model.enc.set_grad_checkpointing(True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    enc_params = list(model.enc.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr}, {"params": head_params, "lr": args.lr_head}],
                            weight_decay=args.wd)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp == "fp16")
    steps = args.epochs * len(train_dl)
    warm = len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))

    # a state file is written after every epoch so an interrupted run continues where it stopped
    state_path = f"{out_dir}/state_fold{args.fold}.pt"
    step, start_ep, preds = 0, 0, None
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        if scaler.is_enabled() and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        step, start_ep = st["step"], st["epoch"] + 1
        say(f"resumed after epoch {st['epoch']}")
    for ep in range(start_ep, args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, mask, pos, y in train_dl:
            x, mask, pos, y = x.to(device, non_blocking=device.type == "cuda"), mask.to(device), pos.to(device), y.to(device)
            with autocast_context(device, args.amp):
                logit = model(x, mask, pos, res=args.res, train=True)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            # An overflow skips the optimizer update; keep the scheduler and EMA in sync.
            if scaler.get_scale() < scale_before:
                continue
            sched.step()
            step += 1
            d = min(args.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            tot += loss.item()
            n += 1
            if n % 100 == 0:
                print(f"  ep {ep} step {n}/{len(train_dl)} loss {tot / n:.4f} {(time.time() - t0) / n:.2f}s/it", flush=True)
        if n == 0:
            raise RuntimeError("No optimizer updates completed; check batch size, training data, and fp16 overflow")
        msg = f"ep {ep:2d} loss {tot / n:.4f} time {time.time() - t0:.0f}s"
        last = ep == args.epochs - 1
        if last or (args.eval_every and (ep + 1) % args.eval_every == 0):
            preds = {k: predict(ema, dl, args.res, device, args.amp) for k, dl in eval_dl.items()}
            g_auc, g_each = macro_auc(gold_y, preds["gold"])
            msg += f" | gold {g_auc:.4f}"
            if val_y is not None:
                v_auc, v_each = macro_auc(val_y, preds["val"])
                msg += f" | val {v_auc:.4f} [" + " ".join(f"{a:.3f}" for a in v_each) + "]"
        say(msg)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "epoch": ep}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    if preds is None:  # resumed after the last epoch had already finished
        preds = {k: predict(ema, dl, args.res, device, args.amp) for k, dl in eval_dl.items()}
        g_auc, g_each = macro_auc(gold_y, preds["gold"])
    torch.save({"model": ema.state_dict(), "args": vars(args)}, f"{out_dir}/fold{args.fold}.pt")
    rows = []
    for k, v in eval_sets.items():
        df = pd.DataFrame(preds[k], index=v.index, columns=LABELS)
        df.insert(0, "set", k)
        rows.append(df)
    pd.concat(rows).to_csv(f"{out_dir}/oof_fold{args.fold}.csv")
    say("gold per class: " + " ".join(f"{c[:6]}={a:.3f}" for c, a in zip(LABELS, g_each)))
    if os.path.exists(state_path):
        os.remove(state_path)


if __name__ == "__main__":
    main()
