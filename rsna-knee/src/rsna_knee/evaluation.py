"""Score reader checkpoints on cached studies (S2 onward).

Each study is loaded once and passed through every checkpoint of its task;
per-checkpoint logits are saved, so fold-ensembling variants need no rerun.
Tasks:
  weak   all weak-labelled studies x all checkpoints ("cross"; OOF is the
         subset where the study's fold equals the checkpoint's held-out fold)
  oof_foldK  only fold K x its checkpoint (when "cross" is not requested)
  gold, test             cached volumes, all checkpoints
  gold_dicom, test_dicom straight from DICOM, as the submission reads them
Inference options (S3) add views of each study whose probabilities are averaged:
  tta="mirror"   the anatomical left-right mirror (knee.mirror_volume)
  series="all"   one pass per candidate series of each slot, not only the longest

Usage (local, test studies only):
  python -m rsna_knee.evaluation run --config c.json --data data/raw \
      --labels experiments/S1/labels/labels_v0.csv --cache runs/s1-local-test/cache_test \
      --test-meta runs/s1-local-test/files_test.csv --ckpt-dir models/convnext_reader --out runs/eval
"""
import argparse
import glob
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import bootstrap_auc, macro_auc, per_label_auc, rank_mean
from .reader.knee import LABELS, N_SLOTS, PLANES, build_study_table, mirror_volume, window_centres

DEFAULT_INFER = {"k_eval": 12, "res": None, "amp": None, "batch_size": 4, "workers": None, "ensemble": "prob",
                 "tta": "none", "series": "first", "mirror_mode": "volume"}
SETS = ("oof", "cross", "gold", "test")
DICOM_SETS = ("gold", "test")
CONFIG_KEYS = {"name", "ckpts", "infer", "sets", "dicom", "labels_sha256", "meta_sha256", "limit", "bootstrap"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value):
    """NaN AUCs (a label or set with one class) become null in strict JSON."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_finite(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def validate_config(config):
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    infer = dict(DEFAULT_INFER)
    extra = set(config.get("infer", {})) - set(DEFAULT_INFER)
    if extra:
        raise ValueError(f"Unknown infer keys: {sorted(extra)}")
    infer.update(config.get("infer", {}))
    if infer["ensemble"] not in {"prob", "logit", "rank"}:
        raise ValueError("ensemble must be prob, logit or rank")
    if infer["tta"] not in {"none", "mirror"} or infer["series"] not in {"first", "all"}:
        raise ValueError("tta must be none or mirror; series must be first or all")
    if infer["mirror_mode"] not in {"volume", "gpu"}:
        raise ValueError("mirror_mode must be volume or gpu")
    sets = list(config.get("sets", ["oof", "gold", "test"]))
    dicom = list(config.get("dicom", []))
    if not set(sets) <= set(SETS) or not set(dicom) <= set(DICOM_SETS):
        raise ValueError(f"sets must be within {SETS}; dicom within {DICOM_SETS}")
    ckpts = config.get("ckpts") or []
    if not ckpts:
        raise ValueError("At least one checkpoint is required")
    for c in ckpts:
        if set(c) - {"file", "fold", "sha256"} or "file" not in c:
            raise ValueError(f"Checkpoint entries need file and optional fold/sha256: {c}")
    folds = [c.get("fold") for c in ckpts if c.get("fold") is not None]
    if len(folds) != len(set(folds)):
        raise ValueError("Each held-out fold may belong to at most one checkpoint")
    limit = config.get("limit")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    return {"name": config.get("name", "eval"), "ckpts": ckpts, "infer": infer, "sets": sets, "dicom": dicom,
            "labels_sha256": config.get("labels_sha256"), "meta_sha256": config.get("meta_sha256"),
            "limit": limit, "bootstrap": int(config.get("bootstrap", 2000))}


def meta_digest(meta):
    """SHA256 over sorted 'series,n' lines; identifies the cache's slice counts."""
    pairs = sorted(zip(meta.series.astype(str), meta.n.fillna(0).astype(int)))
    return hashlib.sha256("\n".join(f"{s},{n}" for s, n in pairs).encode()).hexdigest()


def cache_paths(dirs):
    paths = {}
    for d in dirs:
        with os.scandir(d) as entries:
            for entry in entries:
                if entry.name.endswith(".npy"):
                    series = entry.name[:-4]
                    if series in paths:
                        raise ValueError(f"Series cached twice: {series}")
                    paths[series] = entry.path
    return paths


class CacheVolumes:
    """Picklable (study, series) -> memory-mapped cached volume, for DataLoader workers."""

    def __init__(self, paths):
        self.paths = paths

    def __call__(self, study, series):
        return np.load(self.paths[series], mmap_mode="r")


class DicomVolumes:
    """(study, series) -> canonical volume decoded from DICOM; None if unreadable (as infer.py)."""

    def __init__(self, root):
        self.root = root

    def __call__(self, study, series):
        from .reader.preprocess import load_series

        try:
            return load_series(f"{self.root}/{study}/{series}")[0]
        except Exception:
            return None


class ViewDataset:
    """One inference view per study: candidate `pick` of each slot, optionally mirrored.

    pick=0 without mirroring yields exactly what knee.StudyDataset (cache) and
    infer.DicomStudyDataset (DICOM) yield at inference.
    """

    def __init__(self, studies, table, k, volumes, pick=0, mirror=False):
        from .reader.preprocess import SIZE

        self.studies, self.table, self.k, self.volumes = list(studies), table, k, volumes
        self.pick, self.mirror, self.size = pick, mirror, SIZE

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        import torch

        st = self.studies[i]
        x, mask, pos = None, np.zeros(N_SLOTS, bool), np.zeros((N_SLOTS, self.k), np.float32)
        for s, cands in enumerate(self.table.get(st, [[] for _ in range(N_SLOTS)])):
            if not cands:
                continue
            vol = self.volumes(st, cands[self.pick % len(cands)])
            if vol is None or len(vol) < 3:
                continue
            if self.mirror:
                vol = mirror_volume(vol, s)
            n = len(vol)
            c = window_centres(n, self.k, False)
            win = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])
            if x is None:
                x = np.zeros((N_SLOTS, self.k) + win.shape[1:], np.uint8)
            x[s], mask[s], pos[s] = win, True, c / max(n - 1, 1)
        if x is None:
            x = np.zeros((N_SLOTS, self.k, 3, self.size, self.size), np.uint8)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos)


def mirror_batch(x, mask, pos):
    """The anatomical mirror applied to already-windowed inputs x [B, S, K, 3, H, W].

    Coronal/axial: flip columns. Sagittal: the mirrored volume's windows are the
    same windows with reversed slice (channel) order at position 1 - pos; window
    centres are symmetric, and the study transformer ignores token order. Equals
    ViewDataset(mirror=True) without reading or decoding the volume again.
    """
    import torch

    sag = torch.tensor([PLANES[s % 3] == "Sagittal" for s in range(x.shape[1])], device=x.device)
    flipped = torch.where(sag.view(1, -1, 1, 1, 1, 1), x.flip(3), x.flip(5))
    return flipped, mask, torch.where(sag.view(1, -1, 1) & mask.unsqueeze(-1), 1 - pos, pos)


def task_views(task, infer, studies):
    """(pick, mirror, studies) per view; extra series passes only for studies that have them.
    With mirror_mode="gpu" the mirror is computed from each loaded batch instead (mirror_batch)."""
    picks = {st: 1 for st in studies}
    if infer["series"] == "all":
        picks = {st: max([len(c) for c in task["table"].get(st, [])] or [1]) for st in studies}
    mirrors = [False, True] if infer["tta"] == "mirror" and infer.get("mirror_mode") != "gpu" else [False]
    return [(p, m, [st for st in studies if picks[st] > p]) for p in range(max(picks.values(), default=1))
            for m in mirrors]


def selection(table, studies):
    """First candidate (the one inference uses) per slot."""
    return {st: tuple(c[0] if c else None for c in table.get(st, [[] for _ in range(N_SLOTS)])) for st in studies}


def plan_tasks(config, labels, data, train_meta, test_meta, paths):
    """Build every task's slot table, keeping cache and DICOM selections identical to training/submission."""
    from .reader.infer import dicom_table

    ckpt_folds = [c.get("fold") for c in config["ckpts"]]
    all_ckpts = list(range(len(ckpt_folds)))
    weak = labels[labels.is_gold == 0]
    gold = sorted(labels.index[labels.is_gold == 1])
    tasks = []

    def add(name, source, split, studies, ckpts):
        studies = sorted(studies)[:config["limit"]] if config["limit"] else sorted(studies)
        se = pd.read_csv(Path(data) / f"{split}_series.csv")
        if source == "cache":
            meta = train_meta if split == "train" else test_meta
            if meta is None:
                raise ValueError(f"{name}: cached {split} studies need {split} cache metadata")
            table = build_study_table(se[se.StudyInstanceUID.isin(set(studies))], dict(zip(meta.series, meta.n.fillna(0))))
            missing = {s for st in studies for c in table.get(st, []) for s in c} - set(paths)
            if missing:
                raise FileNotFoundError(f"{name}: {len(missing)} selected series have no cached volume")
        else:
            table = dicom_table(se, str(Path(data) / f"{split}_series"), studies)
        tasks.append({"name": name, "source": source, "split": split, "studies": studies,
                      "ckpts": ckpts, "table": table})

    if "cross" in config["sets"]:
        add("weak", "cache", "train", weak.index, all_ckpts)
    elif "oof" in config["sets"]:
        for i, fold in enumerate(ckpt_folds):
            if fold is not None:
                add(f"oof_fold{fold}", "cache", "train", weak.index[weak.fold == fold], [i])
    if "gold" in config["sets"]:
        add("gold", "cache", "train", gold, all_ckpts)
    if "test" in config["sets"]:
        add("test", "cache", "test", pd.read_csv(Path(data) / "test.csv").StudyInstanceUID, all_ckpts)
    if "gold" in config["dicom"]:
        add("gold_dicom", "dicom", "train", gold, all_ckpts)
    if "test" in config["dicom"]:
        add("test_dicom", "dicom", "test", pd.read_csv(Path(data) / "test.csv").StudyInstanceUID, all_ckpts)
    return tasks


def run_worker(job, rank, world, device=None):
    """Predict this rank's share of every task; returns {task: (studies, logits[ckpt, study, 12], timing)}."""
    import torch

    from .reader.infer import load_models
    from .reader.runtime import autocast_context, resolve_amp, resolve_device

    device = resolve_device(device)
    infer = job["infer"]
    amp = resolve_amp(device, infer["amp"])
    workers = infer["workers"]
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) // world) if device.type == "cuda" else 0
    models = load_models(job["ckpt_paths"], device=device)
    if infer["res"] is not None:
        models = [(m, infer["res"]) for m, _ in models]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    results = {}
    for task in job["tasks"]:
        studies = task["studies"][rank::world]
        index = {st: i for i, st in enumerate(studies)}
        if task["source"] == "cache":
            volumes = CacheVolumes(job["paths"])
        else:
            volumes = DicomVolumes(str(Path(job["data"]) / f"{task['split']}_series"))
        views = task_views(task, infer, studies)
        gpu_mirror = infer["tta"] == "mirror" and infer.get("mirror_mode") == "gpu"
        raw = np.zeros((len(task["ckpts"]), len(studies), len(LABELS)), np.float32)
        prob_sum = np.zeros(raw.shape, np.float64)
        count = np.zeros(len(studies), int)
        wait = compute = 0.0
        start = time.perf_counter()
        for pick, mirror, view_studies in views:
            ds = ViewDataset(view_studies, task["table"], infer["k_eval"], volumes, pick, mirror)
            dl = torch.utils.data.DataLoader(ds, batch_size=infer["batch_size"], num_workers=workers,
                                             pin_memory=device.type == "cuda")
            done, mark = 0, time.perf_counter()
            with torch.no_grad():
                for x, mask, pos in dl:
                    got = time.perf_counter()
                    wait += got - mark
                    x, mask, pos = (t.to(device, non_blocking=device.type == "cuda") for t in (x, mask, pos))
                    passes = [(x, mask, pos)] + ([mirror_batch(x, mask, pos)] if gpu_mirror else [])
                    outs = []
                    for xb, mb, pb in passes:
                        if not mb.any():  # no readable series in the whole batch: the model needs one
                            outs.append(np.zeros((len(task["ckpts"]), len(xb), len(LABELS)), np.float32))
                            continue
                        with autocast_context(device, amp):
                            outs.append(torch.stack([models[i][0](xb, mb, pb, res=models[i][1]).float()
                                                     for i in task["ckpts"]]).cpu().numpy())
                    sync()
                    mark = time.perf_counter()
                    compute += mark - got
                    logits = outs[0]
                    rows = [index[st] for st in view_studies[done:done + logits.shape[1]]]
                    if len(views) == 1 and not gpu_mirror:
                        raw[:, rows] = logits
                    for out in outs:
                        prob_sum[:, rows] += 1 / (1 + np.exp(-out.astype(np.float64)))
                        count[rows] += 1
                    done += logits.shape[1]
                    if done % 200 < logits.shape[1] or done == len(view_studies):
                        print(f"[rank {rank}] {task['name']} view {pick}{'m' if mirror else ''}: "
                              f"{done}/{len(view_studies)} {(mark - start) / max(done, 1):.3f}s/study", flush=True)
        if len(views) > 1 or gpu_mirror:  # store the logit of the view-averaged probability
            p = np.clip(prob_sum / np.maximum(count, 1)[None, :, None], 1e-12, 1 - 1e-12)
            raw = np.log(p / (1 - p)).astype(np.float32)
        wall = time.perf_counter() - start
        results[task["name"]] = (studies, raw, {"studies": len(studies), "wall_seconds": wall,
                                                "data_wait_seconds": wait, "compute_seconds": compute,
                                                "views": [[p, m, len(v)] for p, m, v in views]})
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    return {"results": results, "device": str(device), "amp": amp, "workers": workers, "peak_memory_bytes": peak,
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}


def ensemble(logits, method):
    """logits [ckpt, study, label] -> one score per study and label."""
    probs = 1 / (1 + np.exp(-logits.astype(np.float64)))
    if method == "prob":
        return probs.mean(0)
    if method == "logit":
        return 1 / (1 + np.exp(-logits.astype(np.float64).mean(0)))
    return rank_mean(list(probs))


def merge(tasks, parts, config, out, labels):
    """Write logits.csv, per-set prediction CSVs and timing.json."""
    out = Path(out)
    ckpt_folds = [c.get("fold") for c in config["ckpts"]]
    rows, timing = [], {}
    for task in tasks:
        name = task["name"]
        studies = [s for p in parts for s in p["results"][name][0]]
        logits = np.concatenate([p["results"][name][1] for p in parts], axis=1)
        order = np.argsort(studies, kind="stable")
        studies, logits = [studies[i] for i in order], logits[:, order]
        if studies != task["studies"]:
            raise RuntimeError(f"{name}: merged studies differ from the plan")
        for j, ckpt in enumerate(task["ckpts"]):
            frame = pd.DataFrame(logits[j], columns=LABELS)
            frame.insert(0, "ckpt_fold", -1 if ckpt_folds[ckpt] is None else ckpt_folds[ckpt])
            frame.insert(0, "ckpt", ckpt)
            frame.insert(0, "StudyInstanceUID", studies)
            frame.insert(0, "source", task["source"])
            frame.insert(0, "task", name)
            rows.append(frame)
        if name in {"gold", "test", "gold_dicom", "test_dicom"}:
            pred = pd.DataFrame(ensemble(logits, config["infer"]["ensemble"]), columns=LABELS,
                                index=pd.Index(studies, name="StudyInstanceUID"))
            pred.to_csv(out / f"preds_{name}.csv", float_format="%.9g")
        per_rank = [p["results"][name][2] for p in parts]
        n = sum(t["studies"] for t in per_rank)
        parallel_wall = max(t["wall_seconds"] for t in per_rank)
        timing[name] = {"studies": n, "checkpoints": len(task["ckpts"]), "source": task["source"],
                        "parallel_wall_seconds": parallel_wall,
                        "seconds_per_study_parallel": parallel_wall / n if n else None,
                        "seconds_per_study_one_gpu": sum(t["wall_seconds"] for t in per_rank) / n if n else None,
                        "compute_seconds_per_study_per_ckpt":
                            sum(t["compute_seconds"] for t in per_rank) / n / len(task["ckpts"]) if n else None,
                        "data_wait_fraction": sum(t["data_wait_seconds"] for t in per_rank) /
                                              max(sum(t["wall_seconds"] for t in per_rank), 1e-9),
                        "per_rank": per_rank}
    logits = pd.concat(rows, ignore_index=True)
    logits.to_csv(out / "logits.csv", index=False, float_format="%.9g")
    oof = oof_predictions(logits, labels)
    if len(oof):
        oof.to_csv(out / "preds_oof.csv", float_format="%.9g")
    timing["workers"] = [{k: v for k, v in p.items() if k != "results"} for p in parts]
    write_json(out / "timing.json", timing)
    return logits


def oof_predictions(logits, labels):
    """Each weak study scored only by the checkpoint that held out its fold."""
    weak = logits[(logits.source == "cache") & logits.task.str.fullmatch(r"weak|oof_fold\d+")]
    held = weak[labels.fold.reindex(weak.StudyInstanceUID).to_numpy() == weak.ckpt_fold.to_numpy()]
    pred = 1 / (1 + np.exp(-held.set_index("StudyInstanceUID")[LABELS].astype(np.float64)))
    pred.insert(0, "fold", held.ckpt_fold.to_numpy())
    return pred.sort_index()


def targets(labels, studies, kind):
    if kind == "gold":
        return labels.loc[studies, [f"gold_{c}" for c in LABELS]].to_numpy().astype(int)
    return (labels.loc[studies, LABELS].to_numpy() > 0.5).astype(int)


def auc_summary(labels, pred, kind, n_boot):
    y = targets(labels, pred.index, kind)
    p = pred[LABELS].to_numpy()
    summary = {"studies": len(pred), "auc": macro_auc(y, p), "ci": None,
               "per_label": dict(zip(LABELS, per_label_auc(y, p).tolist())), "bootstrap": None}
    if np.isfinite(summary["auc"]):
        try:
            boot = bootstrap_auc(y, p, n=n_boot, seed=0)
        except ValueError as error:  # tiny (smoke) samples: every draw lacks a class
            summary["bootstrap"] = {"error": str(error)}
        else:
            summary.update(ci=[boot["ci_low"], boot["ci_high"]], bootstrap=boot)
    return summary


def fold_check(logits, labels):
    """AUC of each checkpoint per study fold, plus a per-study membership test.

    If our folds match the author's, a checkpoint scores clearly lower on its
    own held-out fold than on folds it trained on, and among the checkpoints
    the one with the highest loss on a study is the one that held it out.
    """
    weak = logits[(logits.task == "weak") & (logits.source == "cache")]
    if weak.empty:
        return None
    weak = weak.assign(study_fold=labels.fold.reindex(weak.StudyInstanceUID).values)
    ckpts = sorted(weak.ckpt.unique())
    result = {"auc_by_ckpt_and_study_fold": {}, "held_out_vs_trained": {}, "same_studies": {}, "membership": {}}
    for ckpt in ckpts:
        rows = weak[weak.ckpt == ckpt]
        own = int(rows.ckpt_fold.iloc[0])
        by_fold = {}
        for fold, part in rows.groupby("study_fold"):
            by_fold[str(int(fold))] = macro_auc(targets(labels, part.StudyInstanceUID, "weak"), part[LABELS].to_numpy())
        trained = rows[rows.study_fold != own]
        held = rows[rows.study_fold == own]
        result["auc_by_ckpt_and_study_fold"][str(ckpt)] = by_fold
        if own >= 0 and len(held):
            h = macro_auc(targets(labels, held.StudyInstanceUID, "weak"), held[LABELS].to_numpy())
            t = macro_auc(targets(labels, trained.StudyInstanceUID, "weak"), trained[LABELS].to_numpy())
            result["held_out_vs_trained"][str(ckpt)] = {"ckpt_fold": own, "held_out_auc": h, "trained_auc": t,
                                                        "gap": t - h, "held_out_studies": len(held),
                                                        "trained_studies": len(trained)}
    owner = {int(weak[weak.ckpt == c].ckpt_fold.iloc[0]): c for c in ckpts}
    for fold, ckpt in owner.items():
        if fold < 0:
            continue
        others = [str(c) for c in ckpts if c != ckpt]
        aucs = result["auc_by_ckpt_and_study_fold"]
        if str(fold) in aucs[str(ckpt)] and others:
            other = float(np.mean([aucs[c][str(fold)] for c in others]))
            result["same_studies"][str(fold)] = {"held_out_ckpt_auc": aucs[str(ckpt)][str(fold)],
                                                 "trained_ckpts_mean_auc": other,
                                                 "gap": other - aucs[str(ckpt)][str(fold)]}
    # Per-study soft-label BCE of every checkpoint; the held-out one should be worst.
    soft = labels[LABELS].reindex(weak.StudyInstanceUID).to_numpy()
    logit = weak[LABELS].to_numpy(np.float64)
    bce = (np.logaddexp(0, logit) - soft * logit).mean(1)
    loss = weak.assign(bce=bce).pivot(index="StudyInstanceUID", columns="ckpt", values="bce")
    worst = loss.idxmax(axis=1)
    study_fold = labels.fold.reindex(loss.index)
    for fold, ckpt in owner.items():
        members = study_fold.index[study_fold == fold]
        if fold >= 0 and len(members) and len(ckpts) > 1:
            result["membership"][str(fold)] = {"studies": len(members),
                                               "held_out_ckpt_has_highest_loss": float((worst[members] == ckpt).mean()),
                                               "chance": 1 / len(ckpts)}
    return result


def dicom_parity(out, tasks):
    report = {}
    for name in ("gold", "test"):
        a, b = Path(out) / f"preds_{name}.csv", Path(out) / f"preds_{name}_dicom.csv"
        if not (a.exists() and b.exists()):
            continue
        cache, dicom = pd.read_csv(a, index_col=0), pd.read_csv(b, index_col=0)
        common = cache.index.intersection(dicom.index)
        diff = (cache.loc[common, LABELS] - dicom.loc[common, LABELS]).abs()
        by_name = {t["name"]: t for t in tasks}
        sel_cache = selection(by_name[name]["table"], common)
        sel_dicom = selection(by_name[f"{name}_dicom"]["table"], common)
        changed = [s for s in common if sel_cache[s] != sel_dicom[s]]
        same = [s for s in common if s not in set(changed)]
        report[name] = {"studies": len(common), "max_abs_diff": float(diff.to_numpy().max()),
                        "mean_abs_diff": float(diff.to_numpy().mean()),
                        "studies_with_different_series_selection": len(changed),
                        "max_abs_diff_same_selection": float(diff.loc[same].to_numpy().max()) if same else None}
    return report


def summarize(out, labels, logits, tasks, n_boot=2000):
    out = Path(out)
    metrics = {}
    oof_path = out / "preds_oof.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path, index_col=0)
        metrics["oof"] = auc_summary(labels, oof, "weak", n_boot)
        metrics["oof"]["by_fold"] = {str(f): macro_auc(targets(labels, part.index, "weak"), part[LABELS].to_numpy())
                                     for f, part in oof.groupby("fold")}
    for name in ("gold", "gold_dicom"):
        path = out / f"preds_{name}.csv"
        if path.exists():
            metrics[name] = auc_summary(labels, pd.read_csv(path, index_col=0), "gold", n_boot)
    metrics["fold_check"] = fold_check(logits, labels)
    metrics["dicom_parity"] = dicom_parity(out, tasks)
    write_json(out / "metrics.json", metrics)
    return metrics


def environment():
    import torch

    versions = {}
    for name in ["torch", "timm", "numpy", "pandas", "pydicom", "opencv-python-headless", "opencv-python"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {"python": platform.python_version(), "versions": versions, "cuda": torch.version.cuda,
            "gpus": gpus, "cpu_count": os.cpu_count()}


def evaluate(config, *, data, labels_path, train_meta, test_meta, cache_dirs, ckpt_paths, out, device=None):
    config = validate_config(config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    inputs = {"labels_sha256": sha256(labels_path), "ckpt_sha256": [sha256(p) for p in ckpt_paths]}
    if config["labels_sha256"] and inputs["labels_sha256"] != config["labels_sha256"]:
        raise ValueError("labels_v0.csv differs from the frozen S1 table")
    for c, actual in zip(config["ckpts"], inputs["ckpt_sha256"]):
        if c.get("sha256") and c["sha256"] != actual:
            raise ValueError(f"Checkpoint {c['file']} differs from its pinned SHA256")
    if train_meta is not None:
        inputs["meta_sha256"] = meta_digest(train_meta)
        if config["meta_sha256"] and inputs["meta_sha256"] != config["meta_sha256"]:
            raise ValueError("Cache metadata differs from the S1 manifest")
    labels = pd.read_csv(labels_path, index_col=0)
    paths = cache_paths(cache_dirs)
    inputs["cached_series"] = len(paths)
    tasks = plan_tasks(config, labels, data, train_meta, test_meta, paths)
    job = {"tasks": tasks, "ckpt_paths": [str(p) for p in ckpt_paths], "infer": config["infer"],
           "paths": paths, "data": str(data)}
    print(json.dumps({t["name"]: len(t["studies"]) for t in tasks}), flush=True)
    world = _gpu_count() if device in (None, "cuda") else 1
    if world > 1:
        parts = _spawn_workers(job, world, out)
    else:
        parts = [run_worker(job, 0, 1, device)]
    logits = merge(tasks, parts, config, out, labels)
    metrics = summarize(out, labels, logits, tasks, config["bootstrap"])
    write_json(out / "run.json", {"config": config, "inputs": inputs, "environment": environment(),
                                  "tasks": {t["name"]: len(t["studies"]) for t in tasks},
                                  "total_seconds": time.perf_counter() - start})
    return metrics


def _gpu_count():
    import torch

    return torch.cuda.device_count() if torch.cuda.is_available() else 1


def _spawn_workers(job, world, out):
    job_path = out / "_job.pkl"
    job_path.write_bytes(pickle.dumps(job))
    package_root = str(Path(__file__).resolve().parents[1])
    procs = []
    for rank in range(world):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank),
                   PYTHONPATH=package_root + os.pathsep + os.environ.get("PYTHONPATH", ""))
        procs.append(subprocess.Popen([sys.executable, "-m", "rsna_knee.evaluation", "worker", str(job_path),
                                       str(rank), str(world), str(out / f"_part{rank}.pkl")], env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"Evaluation workers failed with exit codes {codes}")
    parts = [pickle.loads((out / f"_part{rank}.pkl").read_bytes()) for rank in range(world)]
    for rank in range(world):
        (out / f"_part{rank}.pkl").unlink()
    job_path.unlink()
    return parts


def find_one(name, root="/kaggle/input", depth=4):
    hits = sorted({p for d in range(depth) for p in glob.glob(f"{root}/{'*/' * d}{name}")})
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits


def kaggle_inputs():
    """The attached S1 cache shards (train + test metadata and volume dirs) and frozen labels."""
    from .reader.paths import DATA

    metas = find_one("meta_shard*.csv")
    shard_dirs = [Path(p).parent for p in metas]
    test_meta_path = find_one("meta_test.csv")
    labels = find_one("labels_v0.csv")
    if len(test_meta_path) != 1 or len(labels) != 1:
        raise ValueError(f"Expected one meta_test.csv and labels_v0.csv: {test_meta_path}, {labels}")
    return {"data": DATA, "labels_path": labels[0],
            "train_meta": pd.concat([pd.read_csv(p) for p in metas], ignore_index=True),
            "test_meta": pd.read_csv(test_meta_path[0]),
            "cache_dirs": [d / "cache_train" for d in shard_dirs] + [Path(test_meta_path[0]).parent / "cache_test"]}


def run_kaggle(config):
    """Evaluate one config, or {"runs": [config, ...]} in sequence, on all GPUs.

    Outputs go to /kaggle/working/eval (single) or eval/<name> (batch); a failed
    batch run is recorded in its error.txt and the rest still run.
    """
    from .reader.paths import WORK

    inputs = kaggle_inputs()
    runs = config["runs"] if "runs" in config else [config]
    names = [r.get("name", "eval") for r in runs]
    if len(set(names)) != len(names):
        raise ValueError("Batch run names must be unique")
    failed = []
    for run in runs:
        ckpt_paths = []
        for c in run["ckpts"]:
            hits = find_one(c["file"])
            if len({sha256(h) for h in hits}) != 1:
                raise ValueError(f"Ambiguous checkpoint {c['file']}: {hits}")
            ckpt_paths.append(hits[0])
        out = Path(WORK) / "eval" / (run["name"] if "runs" in config else "")
        print(json.dumps({"run": run.get("name"), "labels": inputs["labels_path"], "ckpts": ckpt_paths}), flush=True)
        try:
            evaluate(run, ckpt_paths=ckpt_paths, out=out, **inputs)
        except Exception as error:
            if "runs" not in config:
                raise
            out.mkdir(parents=True, exist_ok=True)
            (out / "error.txt").write_text(f"{type(error).__name__}: {error}\n")
            print(f"run {run['name']} failed: {error}", flush=True)
            failed.append(run["name"])
    if failed:
        raise RuntimeError(f"Failed batch runs: {failed}")


def submission(config, *, data, ckpt_paths, out, device=None):
    """Reader-only submission: test studies straight from DICOM, all checkpoints, config's inference."""
    from .reader.infer import dicom_table

    config = validate_config(config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for c, path in zip(config["ckpts"], ckpt_paths):
        if c.get("sha256") and sha256(path) != c["sha256"]:
            raise ValueError(f"Checkpoint {c['file']} differs from its pinned SHA256")
    sample = pd.read_csv(Path(data) / "sample_submission.csv")
    studies = sorted(pd.read_csv(Path(data) / "test.csv").StudyInstanceUID)
    table = dicom_table(pd.read_csv(Path(data) / "test_series.csv"), str(Path(data) / "test_series"), studies)
    task = {"name": "test_dicom", "source": "dicom", "split": "test", "studies": studies,
            "ckpts": list(range(len(ckpt_paths))), "table": table}
    job = {"tasks": [task], "ckpt_paths": [str(p) for p in ckpt_paths], "infer": config["infer"], "paths": {},
           "data": str(data)}
    world = _gpu_count() if device in (None, "cuda") else 1
    parts = _spawn_workers(job, world, out) if world > 1 else [run_worker(job, 0, 1, device)]
    got = [s for p in parts for s in p["results"]["test_dicom"][0]]
    logits = np.concatenate([p["results"]["test_dicom"][1] for p in parts], axis=1)
    pred = pd.DataFrame(ensemble(logits, config["infer"]["ensemble"]), index=got, columns=LABELS)
    if set(got) != set(studies) or not np.isfinite(pred.to_numpy()).all():
        raise RuntimeError("Missing or non-finite test predictions")
    pred = pred.loc[sample.StudyInstanceUID].reset_index(drop=True)
    pred.insert(0, "StudyInstanceUID", sample.StudyInstanceUID.to_numpy())
    pred[list(sample.columns)].to_csv(out / "submission.csv", index=False)
    write_json(out / "submission_run.json", {"config": config, "studies": len(studies),
                                             "seconds": time.perf_counter() - start,
                                             "workers": [{k: v for k, v in p.items() if k != "results"}
                                                         for p in parts]})
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--data", type=Path, default=Path("data/raw"))
    run.add_argument("--labels", type=Path, required=True)
    run.add_argument("--train-meta", type=Path, nargs="*", default=[])
    run.add_argument("--test-meta", type=Path)
    run.add_argument("--cache", type=Path, nargs="+", required=True)
    run.add_argument("--ckpt-dir", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--device", choices=("cuda", "mps", "cpu"))
    worker = sub.add_parser("worker")
    worker.add_argument("job", type=Path)
    worker.add_argument("rank", type=int)
    worker.add_argument("world", type=int)
    worker.add_argument("out", type=Path)
    args = ap.parse_args()
    if args.command == "worker":
        job = pickle.loads(args.job.read_bytes())
        args.out.write_bytes(pickle.dumps(run_worker(job, args.rank, args.world, "cuda")))
        return
    config = json.loads(args.config.read_text())
    train_meta = pd.concat([pd.read_csv(p) for p in args.train_meta]) if args.train_meta else None
    test_meta = pd.read_csv(args.test_meta) if args.test_meta else None
    ckpts = [args.ckpt_dir / c["file"] for c in config["ckpts"]]
    metrics = evaluate(config, data=args.data, labels_path=args.labels, train_meta=train_meta, test_meta=test_meta,
                       cache_dirs=args.cache, ckpt_paths=ckpts, out=args.out, device=args.device)
    print(json.dumps({k: v.get("auc") if isinstance(v, dict) and "auc" in v else v for k, v in metrics.items()},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
