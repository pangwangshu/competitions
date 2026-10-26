"""Predict the 12 findings for every study straight from DICOM folders.

Used by the Kaggle submission notebook and locally to check that predictions from raw DICOMs
match those from the training cache.

Source: goodpjw2008/rsna-knee-2-5d-convnext-reader.

Usage: python -m rsna_knee.reader.infer --ckpts <ckpt> [<ckpt> ...]
  --data contains <split>.csv, <split>_series.csv and <split>_series/
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .knee import LABELS, N_SLOTS, KneeNet, build_study_table, window_centres
from .paths import DATA, RUNS
from .preprocess import MAX_SLICES, SIZE, load_series
from .runtime import autocast_context, resolve_amp, resolve_device


class DicomStudyDataset(torch.utils.data.Dataset):
    """Loads every selected series of a study once; windows are cut per model setting later."""

    def __init__(self, studies, table, root, k):
        self.studies, self.table, self.root, self.k = list(studies), table, root, k

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        st = self.studies[i]
        slots = self.table.get(st, [[] for _ in range(N_SLOTS)])
        x = np.zeros((N_SLOTS, self.k, 3, SIZE, SIZE), np.uint8)
        mask, pos = np.zeros(N_SLOTS, bool), np.zeros((N_SLOTS, self.k), np.float32)
        for s, cands in enumerate(slots):
            if not cands:
                continue
            try:
                vol, _ = load_series(f"{self.root}/{st}/{cands[0]}")
            except Exception:
                continue
            n = len(vol)
            if n < 3:
                continue
            c = window_centres(n, self.k, False)
            x[s] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])
            mask[s], pos[s] = True, c / max(n - 1, 1)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos)


def load_models(ckpts, device="cuda"):
    models = []
    for path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        # the pretrained tag after the dot is irrelevant here and unknown to older timm versions
        m = KneeNet(ck["args"]["backbone"].split(".")[0], pretrained=False)
        m.load_state_dict(ck["model"])
        models.append((m.to(device).to(memory_format=torch.channels_last).eval(), ck["args"]["res"]))
    return models


def dicom_table(se, root, studies):
    """Slot table ranked by DICOM file count, as the submission sees the data (no cache metadata)."""
    se = se[se.StudyInstanceUID.isin(set(studies))]
    n_files = {}
    for st, sr in zip(se.StudyInstanceUID, se.SeriesInstanceUID):
        d = f"{root}/{st}/{sr}"
        n_files[sr] = min(len(os.listdir(d)), MAX_SLICES) if os.path.isdir(d) else 0
    return build_study_table(se, n_files)


@torch.no_grad()
def run(data_dir, split, ckpts, k=12, studies=None, workers=None, bs=None, device=None, amp=None):
    device = resolve_device(device)
    amp = resolve_amp(device, amp)
    if not ckpts:
        raise ValueError("At least one checkpoint is required")
    workers = (4 if device.type == "cuda" else 0) if workers is None else workers
    bs = (4 if device.type == "cuda" else 1) if bs is None else bs
    se = pd.read_csv(f"{data_dir}/{split}_series.csv")
    root = f"{data_dir}/{split}_series"
    if studies is None:
        studies = pd.read_csv(f"{data_dir}/{split}.csv").StudyInstanceUID.tolist()
    table = dicom_table(se, root, studies)
    dl = torch.utils.data.DataLoader(DicomStudyDataset(studies, table, root, k), batch_size=bs,
                                     num_workers=workers, pin_memory=device.type == "cuda")
    models = load_models(ckpts, device=device)
    out, completed = [], 0
    for x, mask, pos in dl:
        x, mask, pos = x.to(device, non_blocking=device.type == "cuda"), mask.to(device), pos.to(device)
        with autocast_context(device, amp):
            p = torch.stack([m(x, mask, pos, res=res).float().sigmoid() for m, res in models]).mean(0)
        out.append(p.cpu())
        completed += len(p)
        print(f"  study {completed}/{len(studies)}: slots used {int(mask.sum())}/{mask.numel()}", flush=True)
    return pd.DataFrame(torch.cat(out).numpy(), index=pd.Index(studies, name="StudyInstanceUID"), columns=LABELS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", default=f"{RUNS}/reader_test.csv")
    ap.add_argument("--device", choices=("cuda", "mps", "cpu"))
    ap.add_argument("--amp", choices=("bf16", "fp16", "none"))
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--bs", type=int)
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    t0 = time.perf_counter()
    predictions = run(args.data, args.split, args.ckpts, k=args.k, bs=args.bs,
                      workers=args.workers, device=args.device, amp=args.amp)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out)
    print(f"wrote {args.out} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
