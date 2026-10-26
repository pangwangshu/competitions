"""S6: train the final ensemble C* (confirmed on 2026-10-10).

  python scripts/s6.py configs          # experiments/S6/session-{1,2}-config.json
  python scripts/s6.py probe-configs    # experiments/S6-probe-*/config.json (reader-only and blend submissions)
  python scripts/s6.py final-configs    # after session 2: 5-fold I*2 eval, final own (C* x10) and final blend configs

C* members: ConvNeXt-Tiny with mirror training, 14 epochs (S5 T1), and ConvNeXt-Small with the
author's recipe at 10 epochs (S5 T3); five folds each, I*2 inference. Fold 0 exists from S5
(S5-T1, S5-T3-small-EP10); sessions 1 and 2 add folds 1-2 and 3-4.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsna_knee import evaluation  # noqa: E402
from scripts.s2 import base_config  # noqa: E402
from scripts.s5 import TRAIN_ARGS  # noqa: E402

EXP = ROOT / "experiments"
TINY = {"mirror_p": 0.5}
SMALL = {"backbone": "convnext_small.fb_in22k_ft_in1k", "epochs": 10}
FOLD0 = {"tiny": "S5-T1", "small": "S5-T3-small-EP10"}


def job(kind, fold, gpu):
    args = {**TRAIN_ARGS, **(TINY if kind == "tiny" else SMALL), "fold": fold}
    return {"name": f"S6-{kind}-f{fold}", "gpu": gpu, "keep_ckpt": True, "args": args}


def write_configs():
    pins = {k: base_config()[k] for k in ("labels_sha256", "meta_sha256")}
    (EXP / "S6").mkdir(parents=True, exist_ok=True)
    for i, folds in enumerate([(1, 2), (3, 4)], start=1):
        session = {"name": f"S6-session-{i}", **pins, "max_hours": 11.5,
                   "queues": [[job("tiny", f, 0) for f in folds], [job("small", f, 1) for f in folds]]}
        evaluation.write_json(EXP / "S6" / f"session-{i}-config.json", session)
        print(json.dumps([[j["name"] for j in q] for q in session["queues"]]))


CKPT_MANIFEST = ROOT / "runs" / "kx" / "ckpts" / "ckpt-manifest.json"
CKPT_FILES = {"tiny": {0: "S5-T1_fold0.pt", **{f: f"S6-tiny-f{f}_fold{f}.pt" for f in range(1, 5)}},
              "small": {0: "S5-T3-small-EP10_fold0.pt", **{f: f"S6-small-f{f}_fold{f}.pt" for f in range(1, 5)}}}
PROBES = {"S6-probe-tiny3": [("tiny", f) for f in range(3)],
          "S6-probe-tiny3-small3": [("tiny", f) for f in range(3)] + [("small", f) for f in range(3)],
          # Public ensemble curve at N=9: the author's fold 0-2 checkpoints added to probe B.
          "S6-probe-author3-tiny3-small3": ([("author", f) for f in range(3)] + [("tiny", f) for f in range(3)]
                                            + [("small", f) for f in range(3)])}


def probe_config(name):
    """Reader-only submission config: C* members trained so far, I*2 inference, probability mean."""
    from scripts.s3 import CANDIDATES

    base = base_config()
    manifest = json.loads(CKPT_MANIFEST.read_text()) | {c["file"]: c["sha256"] for c in base["ckpts"]}
    infer = {**base["infer"], **CANDIDATES["S3-Istar2"][0]}
    files = [f"cnxt_v0_fold{fold}.pt" if kind == "author" else CKPT_FILES[kind][fold] for kind, fold in PROBES[name]]
    config = {"name": name, "infer": infer, "ckpts": [{"file": f, "sha256": manifest[f]} for f in files]}
    evaluation.validate_config(config)
    return config


# kaggle/submit-blend: public CoAtNet + one of our reader probes, rank(w * CoAtNet + (1 - w) * ours).
# The weight is fixed before any blend is scored, never tuned on the public LB.
BLENDS = {"S6-probe-blend-coat07-tiny3-small3": ("S6-probe-tiny3-small3", 0.7)}


def write_probe_configs():
    for name in PROBES:
        (EXP / name).mkdir(parents=True, exist_ok=True)
        evaluation.write_json(EXP / name / "config.json", probe_config(name))
        print(name, [c["file"] for c in probe_config(name)["ckpts"]])
    for name, (reader, weight) in BLENDS.items():
        (EXP / name).mkdir(parents=True, exist_ok=True)
        config = {**probe_config(reader), "name": name, "blend": {"coatnet_weight": weight}}
        evaluation.write_json(EXP / name / "config.json", config)
        print(name, weight, reader)


FINAL_OWN = "S6-final-own"
FINAL_BLEND = "S6-final-blend-coat07"


def final_configs():
    """Final I*2 eval (one run per backbone: a fold may belong to one checkpoint per run), own C* x10 submission, blend."""
    from scripts.s3 import CANDIDATES

    manifest = json.loads(CKPT_MANIFEST.read_text())
    missing = [f for kind in CKPT_FILES.values() for f in kind.values() if f not in manifest]
    if missing:
        raise SystemExit(f"Checkpoints not in the manifest yet (run after S6 session 2): {missing}")
    base = base_config()
    infer = {**base["infer"], **CANDIDATES["S3-Istar2"][0]}
    runs = []
    for kind in ("tiny", "small"):
        config = {**base, "name": f"S6-final-{kind}5-istar2", "sets": ["oof", "gold"], "dicom": [], "infer": infer,
                  "ckpts": [{"file": f, "fold": fold, "sha256": manifest[f]} for fold, f in CKPT_FILES[kind].items()]}
        evaluation.validate_config(config)
        runs.append(config)
    evaluation.write_json(EXP / "S6" / "eval-final-config.json", {"runs": runs})
    files = [CKPT_FILES[kind][fold] for kind in ("tiny", "small") for fold in range(5)]
    own = {"name": FINAL_OWN, "infer": infer, "ckpts": [{"file": f, "sha256": manifest[f]} for f in files]}
    evaluation.validate_config(own)
    blend = {**own, "name": FINAL_BLEND, "blend": {"coatnet_weight": 0.7}}
    for name, config in ((FINAL_OWN, own), (FINAL_BLEND, blend)):
        (EXP / name).mkdir(parents=True, exist_ok=True)
        evaluation.write_json(EXP / name / "config.json", config)
    print(json.dumps({"eval": [r["name"] for r in runs], FINAL_OWN: files, FINAL_BLEND: 0.7}))


def member_dir(kind, fold):
    return EXP / (FOLD0[kind] if fold == 0 else f"S6-{kind}-f{fold}")


def record_session(pulled):
    """Copy a pulled training session into experiments/S6-<kind>-f<k>/ and pool the in-training OOF over all folds.

    Training-time predictions (EMA, single view), not I*2; the final eval re-scores the checkpoints with I*2.
    """
    import shutil

    import pandas as pd

    from rsna_knee.metrics import bootstrap_auc, macro_auc, paired_bootstrap
    from rsna_knee.reader.knee import LABELS
    from scripts.compare import DEFAULT_LABELS
    from scripts.s4 import run_predictions

    labels = pd.read_csv(DEFAULT_LABELS, index_col=0)
    for job in json.loads((pulled / "jobs.json").read_text()):
        if job["return_code"] != 0 or job["killed_at_deadline"]:
            raise SystemExit(f"{job['name']} did not finish; resume it before recording")
        fold = job["args"]["fold"]
        exp = EXP / job["name"]
        exp.mkdir(parents=True, exist_ok=True)
        run = pulled / "runs" / job["name"]
        for f in (f"log_fold{fold}.txt", f"oof_fold{fold}.csv"):
            shutil.copy(run / f, exp / f)
        shutil.copy(pulled / "logs" / f"{job['name']}.stdout.txt", exp)
        run_predictions(exp / f"oof_fold{fold}.csv", exp, fold)
    preds = {(kind, fold): {s: pd.read_csv(member_dir(kind, fold) / f"preds_{s}.csv", index_col=0)[LABELS]
                            for s in ("oof", "gold")}
             for kind in CKPT_FILES for fold in range(5) if (member_dir(kind, fold) / "preds_gold.csv").exists()}
    folds = sorted({f for _, f in preds})

    def pooled(kind):
        return pd.concat([preds[kind, f]["oof"] for f in folds]).sort_index()

    def gold(kinds):
        return sum(preds[k, f]["gold"] for k in kinds for f in folds) / (len(kinds) * len(folds))

    oof = {"tiny": pooled("tiny"), "small": pooled("small")}
    oof["tiny+small"] = (oof["tiny"] + oof["small"]) / 2
    weak = oof["tiny"].index
    y_weak = (labels.loc[weak, LABELS].to_numpy() > 0.5).astype(int)
    golds = {"tiny": gold(["tiny"]), "small": gold(["small"]), "tiny+small": gold(["tiny", "small"])}
    y_gold = labels.loc[golds["tiny"].index, [f"gold_{c}" for c in LABELS]].to_numpy().astype(int)

    def boot(y, p):
        b = bootstrap_auc(y, p.to_numpy())
        return {"auc": macro_auc(y, p.to_numpy()), "ci": [b["ci_low"], b["ci_high"]]}

    per_fold = {}
    for f in folds:
        mask = labels.loc[weak, "fold"].to_numpy() == f
        per_fold[f] = {f"{k}_oof": macro_auc(y_weak[mask], p.to_numpy()[mask]) for k, p in oof.items()}
        per_fold[f].update({f"{k}_gold": macro_auc(y_gold, preds[k, f]["gold"].loc[golds["tiny"].index].to_numpy())
                            for k in ("tiny", "small")})
    analysis = {
        "note": "in-training EMA predictions (single view), probability mean; I*2 re-score in the final eval",
        "folds": folds, "per_fold": per_fold,
        "pooled_oof": {k: boot(y_weak, p) for k, p in oof.items()},
        "pooled_oof_deltas": {f"tiny+small vs {k}": paired_bootstrap(y_weak, oof[k].to_numpy(),
                                                                     oof["tiny+small"].to_numpy())
                              for k in ("tiny", "small")},
        "gold58": {f"{k} x{len(folds) * (2 if '+' in k else 1)}": boot(y_gold, p) for k, p in golds.items()},
    }
    out = EXP / "S6" / f"{pulled.name.replace('s6-', '')}-analysis.json"
    evaluation.write_json(out, analysis)
    print(json.dumps({"per_fold": per_fold, "pooled": analysis["pooled_oof"], "gold58": analysis["gold58"],
                      "deltas": {k: [v["delta_observed"], v["ci_low"], v["ci_high"]]
                                 for k, v in analysis["pooled_oof_deltas"].items()}}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["configs", "probe-configs", "final-configs", "record-session"])
    ap.add_argument("pulled", nargs="?", type=Path, help="record-session: pulled training output, e.g. runs/s6-session-2")
    args = ap.parse_args()
    if args.command == "record-session":
        record_session(args.pulled)
    else:
        {"configs": write_configs, "probe-configs": write_probe_configs, "final-configs": final_configs}[args.command]()
