"""S5: training-side vision experiments on fold 0 (control: S4-R0).

  python scripts/s5.py configs        # eval of R0/T1 under I*2, and the next training session
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsna_knee import evaluation  # noqa: E402
from scripts.s2 import base_config  # noqa: E402
from scripts.s3 import CANDIDATES  # noqa: E402
from scripts.s4 import AUTHOR_ARGS  # noqa: E402

EXP = ROOT / "experiments"
CKPT_MANIFEST = ROOT / "runs" / "kx" / "ckpts" / "ckpt-manifest.json"
ISTAR2 = CANDIDATES["S3-Istar2"][0]
# R0's speed-test choice. eval_every 0 skips per-epoch validation (~25% of the time); training
# maths is unchanged, though the skipped evaluation loaders no longer consume torch RNG draws.
TRAIN_ARGS = {**AUTHOR_ARGS, "amp": "fp16", "bs": 8, "workers": 2, "grad_ckpt": 1, "eval_every": 0}


def eval_run(name, ckpt_file):
    manifest = json.loads(CKPT_MANIFEST.read_text())
    config = base_config()
    config.update(name=name, sets=["oof", "gold"], dicom=[],
                  ckpts=[{"file": ckpt_file, "fold": 0, "sha256": manifest[ckpt_file]}])
    config["infer"].update(ISTAR2)
    evaluation.validate_config(config)
    return config


T4S6_CKPTS = ["S5-T4S6_fold0.pt", "S5-T4S6-seed1_fold0.pt"]
T1_CKPTS = ["S5-T1_fold0.pt", "S5-T1-seed1_fold0.pt"]


def write_t4s6_followups():
    """I*2 eval of both distilled students and of the S5-T1-seed1 control (S5-T1 has S5-T1-istar2).
    Plus two reader-only public probes: E = S5-T1 x2 seeds (v0 labels), F = T4-S6 students x2."""
    runs = [eval_run(f"{f.removesuffix('_fold0.pt')}-istar2", f) for f in T4S6_CKPTS + T1_CKPTS[1:]]
    evaluation.write_json(EXP / "S5" / "eval-t4s6-config.json", {"runs": runs})
    manifest = json.loads(CKPT_MANIFEST.read_text())
    infer = {**base_config()["infer"], **ISTAR2}
    for name, files in (("S5-probe-E-t1x2", T1_CKPTS), ("S5-probe-F-t4s6x2", T4S6_CKPTS)):
        config = {"name": name, "infer": infer, "ckpts": [{"file": f, "sha256": manifest[f]} for f in files]}
        evaluation.validate_config(config)
        (EXP / name).mkdir(parents=True, exist_ok=True)
        evaluation.write_json(EXP / name / "config.json", config)


def job(name, gpu, **args):
    return {"name": name, "gpu": gpu, "keep_ckpt": True, "args": {**TRAIN_ARGS, **args}}


def write_configs():
    batch = {"runs": [eval_run("S5-R0-istar2", "S4-R0_fold0.pt"), eval_run("S5-T1-istar2", "S5-T1_fold0.pt")]}
    (EXP / "S5").mkdir(parents=True, exist_ok=True)
    evaluation.write_json(EXP / "S5" / "eval-r0-t1-config.json", batch)
    pins = {k: base_config()[k] for k in ("labels_sha256", "meta_sha256")}
    session = {"name": "S5-session-A", **pins, "max_hours": 11.5, "queues": [
        [job("S5-T2", 0, k=12)],
        [job("S5-EP10", 1, epochs=10), job("S5-R0-seed1", 1, seed=1)],
    ]}
    evaluation.write_json(EXP / "S5" / "session-A-config.json", session)
    # Session B: seed noise on fold 0 is ~0.002 OOF, so each claim gets two seeds; everything uses the
    # 10-epoch schedule (S5-EP10: indistinguishable from 14 epochs at 72% of the time).
    ep10 = {"epochs": 10}
    session_b = {"name": "S5-session-B", **pins, "max_hours": 11.5, "queues": [
        [job("S5-T1-EP10", 0, mirror_p=0.5, **ep10), job("S5-T1-EP10-seed1", 0, mirror_p=0.5, seed=1, **ep10)],
        [job("S5-EP10-seed1", 1, seed=1, **ep10),
         job("S5-T3-small-EP10", 1, backbone="convnext_small.fb_in22k_ft_in1k", **ep10)],
    ]}
    evaluation.write_json(EXP / "S5" / "session-B-config.json", session_b)
    # Session C closes S5 with two seeds per claim: mirror at 14 epochs (the single-seed gain did not
    # replicate at 10), ConvNeXt-Small, and training resolution 288 (320 does not fit a T4 at bs 8).
    small = "convnext_small.fb_in22k_ft_in1k"
    session_c = {"name": "S5-session-C", **pins, "max_hours": 11.5, "queues": [
        [job("S5-T1-seed1", 0, mirror_p=0.5, seed=1), job("S5-T5-res288-EP10", 0, res=288, **ep10)],
        [job("S5-T3-small-EP10-seed1", 1, backbone=small, seed=1, **ep10),
         job("S5-T5-res288-EP10-seed1", 1, res=288, seed=1, **ep10)],
    ]}
    evaluation.write_json(EXP / "S5" / "session-C-config.json", session_c)
    # T4: the T1 recipe (mirror, 14 epochs) on distillation labels; control = S5-T1 / S5-T1-seed1 on v0.
    session_t4 = {"queues": []}
    if (EXP / "S5-T4" / "labels_t4.json").exists():
        t4 = json.loads((EXP / "S5-T4" / "labels_t4.json").read_text())
        # Accept the original hand-written record as well as the current builder schema.
        t4_sha = t4["labels_sha256"] if "labels_sha256" in t4 else t4["labels_t4_sha256"]
        session_t4 = {"name": "S5-session-T4", **pins, "max_hours": 11.5,
                      "extra_labels": {"labels_t4": {"file": "labels_t4.csv", "sha256": t4_sha}},
                      "queues": [[job("S5-T4", 0, mirror_p=0.5, labels="labels_t4")],
                                 [job("S5-T4-seed1", 1, mirror_p=0.5, labels="labels_t4", seed=1)]]}
        evaluation.write_json(EXP / "S5" / "session-T4-config.json", session_t4)
    # T4-S6: same students, teacher = nartaa's S6 ConvNeXt (no OAI); written once the distillation labels exist.
    t4s_json = EXP / "S5-T4S6" / "labels_t4s.json"
    if t4s_json.exists():
        t4s = json.loads(t4s_json.read_text())
        session_t4s = {"name": "S5-session-T4S6", **pins, "max_hours": 11.5,
                       "extra_labels": {"labels_t4s": {"file": "labels_t4s.csv", "sha256": t4s["labels_sha256"]}},
                       "queues": [[job("S5-T4S6", 0, mirror_p=0.5, labels="labels_t4s")],
                                  [job("S5-T4S6-seed1", 1, mirror_p=0.5, labels="labels_t4s", seed=1)]]}
        evaluation.write_json(EXP / "S5" / "session-T4S6-config.json", session_t4s)
    if T4S6_CKPTS[0] in json.loads(CKPT_MANIFEST.read_text()):
        write_t4s6_followups()
    print(json.dumps({"eval": [r["name"] for r in batch["runs"]],
                      "A": [[j["name"] for j in q] for q in session["queues"]],
                      "B": [[j["name"] for j in q] for q in session_b["queues"]],
                      "C": [[j["name"] for j in q] for q in session_c["queues"]],
                      "T4": [[j["name"] for j in q] for q in session_t4["queues"]]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["configs"])
    ap.parse_args()
    write_configs()
