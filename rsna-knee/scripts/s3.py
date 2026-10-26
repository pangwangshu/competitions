"""S3: inference-side ablations of the author's checkpoints (no training).

  python scripts/s3.py configs                 # experiments/S3/batch-config.json (one Kaggle session)
  python scripts/s3.py offline-a3              # A3 fold-ensembling variants from the E-base logits
  python scripts/s3.py report <pulled_dir> --gpu-hours H --kernel-version V
"""

import argparse
import json
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
from scripts.compare import compare  # noqa: E402
from scripts.s2 import base_config  # noqa: E402

EXP = ROOT / "experiments"
CONTROL = EXP / "S2-E-base"
VARIANTS = {
    "S3-control": ({}, "E-base 配置经新的多视图代码路径重跑，应与 E-base 一致"),
    "S3-A1-k16": ({"k_eval": 16}, "A1：k_eval 12 → 16"),
    "S3-A1-k24": ({"k_eval": 24}, "A1：k_eval 12 → 24"),
    "S3-A2-mirror": ({"tta": "mirror"}, "A2：解剖镜像 TTA（冠/轴位翻列、矢状位反转切片），与原图概率平均"),
    "S3-A4-res288": ({"res": 288}, "A4：推理分辨率 256 → 288（训练 256）"),
    "S3-A5-series-all": ({"series": "all"}, "A5：每槽位所有候选序列的预测平均（原为只取切片最多的）"),
}


def variant_config(name, overrides):
    config = base_config()
    config.update(name=name, sets=["oof", "gold", "test"], dicom=[])
    config["infer"].update(overrides)
    evaluation.validate_config(config)
    return config


def write_configs():
    batch = {"runs": [variant_config(name, o) for name, (o, _) in VARIANTS.items()]}
    path = EXP / "S3" / "batch-config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluation.write_json(path, batch)
    print(path)


OUTPUTS = ["preds_oof.csv", "preds_gold.csv", "preds_test.csv", "logits.csv", "timing.json", "run.json",
           "metrics.json", "error.txt"]
VERDICT_ZH = {"adopt": "采纳", "pending": "待定", "reject": "拒绝", "gold_only": "仅 gold（无 OOF，不自动采纳）"}


def _fmt(r):
    return f"{r['delta_observed']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"


def record(exp, config_diff, kernel, version, gpu_hours, result, cost_ratio=None, extra=""):
    labels = pd.read_csv(evaluation_labels(), index_col=0)
    metrics = {}
    for name, kind in [("oof", "weak"), ("gold", "gold")]:
        path = exp / f"preds_{name}.csv"
        if path.exists():
            summary = evaluation.auc_summary(labels, pd.read_csv(path, index_col=0), kind, 2000)
            metrics[f"m_{name}"] = {"auc": summary["auc"], "ci": summary["ci"]}
    parts = [VERDICT_ZH[result["verdict"]]]
    if result["oof"] is not None:
        parts.append(f"Δ_oof {_fmt(result['oof'])}")
    parts.append(f"Δ_gold {_fmt(result['gold'])}")
    if cost_ratio is not None:
        parts.append(f"每 study 计算 ×{cost_ratio:.2f}")
    entry = {"id": exp.name, "date": date.today().isoformat(), "config_diff": config_diff, "kernel": kernel,
             "kernel_version": version, "m_oof": metrics.get("m_oof"), "m_gold": metrics.get("m_gold"),
             "public_score": None, "gpu_hours": round(gpu_hours, 3),
             "conclusion": "；".join(parts) + (f"；{extra}" if extra else "")}
    evaluation.write_json(exp / "comparison.json", result)
    evaluation.write_json(exp / "metrics.json", entry)
    return entry


def evaluation_labels():
    return EXP / "S1" / "labels" / "labels_v0.csv"


def offline_a3():
    """Rank and logit averaging of the three checkpoints, from the E-base per-checkpoint logits."""
    logits = pd.read_csv(CONTROL / "logits.csv")
    entries = []
    for method, text in [("rank", "A3：三折 rank 平均（原为概率平均），由 E-base logit 离线计算"),
                         ("logit", "A3：三折 logit 平均（原为概率平均），由 E-base logit 离线计算")]:
        exp = EXP / f"S3-A3-{method}"
        exp.mkdir(parents=True, exist_ok=True)
        for task in ("gold", "test"):
            part = logits[logits.task == task]
            studies = sorted(part.StudyInstanceUID.unique())
            stack = np.stack([part[part.ckpt == c].set_index("StudyInstanceUID").loc[studies, LABELS].to_numpy()
                              for c in sorted(part.ckpt.unique())])
            prob = evaluation.ensemble(stack, "prob")
            control = pd.read_csv(CONTROL / f"preds_{task}.csv", index_col=0).loc[studies, LABELS].to_numpy()
            if not np.allclose(prob, control, atol=1e-7):
                raise ValueError("E-base logits do not reproduce its probability ensemble")
            pd.DataFrame(evaluation.ensemble(stack, method), index=pd.Index(studies, name="StudyInstanceUID"),
                         columns=LABELS).to_csv(exp / f"preds_{task}.csv", float_format="%.9g")
        result = compare(CONTROL, exp)
        entries.append(record(exp, text, "离线（E-base logit）", None, 0.0, result))
    return entries


def control_check(control_dir):
    """Max |logit| difference between the S3 control run and E-base, per matched study x checkpoint."""
    def keyed(frame):
        kind = frame.task.where(~frame.task.str.fullmatch(r"weak|oof_fold\d+"), "weak")
        return frame.assign(kind=kind).set_index(["kind", "StudyInstanceUID", "ckpt"])[LABELS]

    a, b = keyed(pd.read_csv(CONTROL / "logits.csv")), keyed(pd.read_csv(control_dir / "logits.csv"))
    common = a.index.intersection(b.index)
    diff = (a.loc[common] - b.loc[common]).abs()
    return {"matched": len(common), "control_rows": len(b), "max_abs_logit_diff": float(diff.to_numpy().max()),
            "mean_abs_logit_diff": float(diff.to_numpy().mean())}


def report(pulled, gpu_hours, kernel_version):
    pulled = Path(pulled)
    kernel = "wangshu/rsna-knee-eval"
    runs = {p.name: p for p in pulled.rglob("*") if p.is_dir() and p.name in VARIANTS}
    missing = set(VARIANTS) - set(runs)
    if missing:
        raise FileNotFoundError(f"Missing batch runs: {sorted(missing)}")
    seconds = {}
    for name, src in runs.items():
        exp = EXP / name
        exp.mkdir(parents=True, exist_ok=True)
        for f in OUTPUTS:
            if (src / f).exists():
                shutil.copy2(src / f, exp / ("eval_metrics.json" if f == "metrics.json" else f))
        if (exp / "error.txt").exists():
            raise RuntimeError(f"{name} failed: {(exp / 'error.txt').read_text()}")
        evaluation.write_json(exp / "config.json", variant_config(name, VARIANTS[name][0]))
        seconds[name] = json.loads((exp / "run.json").read_text())["total_seconds"]
    share = {n: gpu_hours * s / sum(seconds.values()) for n, s in seconds.items()}
    check = control_check(EXP / "S3-control")
    timing = {n: json.loads((EXP / n / "timing.json").read_text()) for n in VARIANTS}

    def cost(name):  # GPU compute per OOF study, relative to the control run of the same session
        per = [t["compute_seconds_per_study_per_ckpt"] for k, t in timing[name].items() if k.startswith("oof_fold")]
        ref = [t["compute_seconds_per_study_per_ckpt"] for k, t in timing["S3-control"].items()
               if k.startswith("oof_fold")]
        return float(np.mean(per) / np.mean(ref))

    summary = {"control_vs_e_base": check, "gpu_hours_session": gpu_hours, "variants": {}}
    for name, (overrides, text) in VARIANTS.items():
        exp = EXP / name
        result = compare(CONTROL, exp)
        result["vs_session_control"] = compare(EXP / "S3-control", exp)
        extra = (f"与 E-base logit 最大差 {check['max_abs_logit_diff']:.1e}（{check['matched']} 行）"
                 if name == "S3-control" else "")
        entry = record(exp, text, kernel, kernel_version, share[name], result, cost(name), extra)
        summary["variants"][name] = {"overrides": overrides, "verdict": result["verdict"],
                                     "oof": {k: result["oof"][k] for k in ("delta_observed", "ci_low", "ci_high",
                                                                           "p_positive")},
                                     "gold": {k: result["gold"][k] for k in ("delta_observed", "ci_low", "ci_high",
                                                                             "p_positive")},
                                     "compute_ratio": cost(name), "seconds_per_study_parallel":
                                         {k: t["seconds_per_study_parallel"] for k, t in timing[name].items()
                                          if isinstance(t, dict) and "seconds_per_study_parallel" in t},
                                     "gpu_hours_share": share[name], "conclusion": entry["conclusion"]}
    evaluation.write_json(EXP / "S3" / "summary.json", summary)
    print(json.dumps({n: (v["verdict"], round(v["oof"]["delta_observed"], 4), round(v["gold"]["delta_observed"], 4),
                          round(v["compute_ratio"], 2)) for n, v in summary["variants"].items()}, indent=1))
    print(json.dumps(check))
    return summary


ISTAR = {"k_eval": 24, "tta": "mirror", "mirror_mode": "gpu", "series": "all"}
ISTAR_TEXT = "I*：A1 k_eval 24 + A2 镜像 TTA（GPU 实现）+ A5 所有候选序列"
TIME_RATIO, HIDDEN_HOURS = 3.0, 1.0  # plan gates: per-study time vs E-base, extrapolated hidden-test hours


# I* exceeded the 1h hidden-test estimate (1.08h); I*2 drops the smallest gain (all series),
# which also multiplies DICOM decoding.
CANDIDATES = {"S3-Istar": (ISTAR, ISTAR_TEXT),
              "S3-Istar2": ({"k_eval": 24, "tta": "mirror", "mirror_mode": "gpu"},
                            "I*2：A1 k_eval 24 + A2 镜像 TTA（GPU 实现）；去掉 A5 以满足 1 小时外推门槛")}


def istar_config(name="S3-Istar"):
    config = variant_config(name, CANDIDATES[name][0])
    config["dicom"] = ["gold", "test"]
    (EXP / name).mkdir(parents=True, exist_ok=True)
    evaluation.write_json(EXP / name / "config.json", config)
    print(json.dumps(config["infer"]))


def istar_report(pulled, gpu_hours, kernel_version, name="S3-Istar"):
    exp = EXP / name
    src = next(p.parent for p in Path(pulled).rglob("run.json"))
    for f in OUTPUTS + ["preds_gold_dicom.csv", "preds_test_dicom.csv"]:
        if (src / f).exists():
            shutil.copy2(src / f, exp / ("eval_metrics.json" if f == "metrics.json" else f))
    result = compare(CONTROL, exp)
    best = max(["S3-A1-k24", "S3-A2-mirror", "S3-A5-series-all"],
               key=lambda n: json.loads((EXP / n / "comparison.json").read_text())["oof"]["delta_observed"])
    vs_best = compare(EXP / best, exp)
    result["vs_best_single"] = {"variant": best, **vs_best}
    timing = json.loads((exp / "timing.json").read_text())
    base = json.loads((CONTROL / "timing.json").read_text())["gold_dicom"]["seconds_per_study_parallel"]
    per_study = timing["gold_dicom"]["seconds_per_study_parallel"]
    hidden = 1300 * 1.5 * per_study / 3600
    gates = {"not_worse_than_best_single": vs_best["oof"]["ci_high"] > 0,
             "time_ratio": per_study / base, "time_ratio_ok": per_study / base <= TIME_RATIO,
             "hidden_hours": hidden, "hidden_ok": hidden <= HIDDEN_HOURS}
    result["gates"] = gates
    extra = (f"对最佳单项 {best} Δ_oof {_fmt(vs_best['oof'])}；DICOM {per_study:.2f}s/study"
             f"（E-base ×{gates['time_ratio']:.2f}），隐藏集估计 {hidden:.2f}h")
    entry = record(exp, CANDIDATES[name][1], "wangshu/rsna-knee-eval", kernel_version, gpu_hours, result, None,
                   extra)
    print(json.dumps(gates, indent=1), entry["conclusion"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["configs", "offline-a3", "report", "istar-config", "istar-report"])
    ap.add_argument("pulled", nargs="?", type=Path)
    ap.add_argument("--gpu-hours", type=float)
    ap.add_argument("--kernel-version", type=int)
    ap.add_argument("--name", default="S3-Istar", choices=sorted(CANDIDATES))
    args = ap.parse_args()
    if args.command == "configs":
        write_configs()
    elif args.command == "istar-config":
        istar_config(args.name)
    elif args.command == "istar-report":
        istar_report(args.pulled, args.gpu_hours, args.kernel_version, args.name)
    elif args.command == "offline-a3":
        for e in offline_a3():
            print(e["id"], e["conclusion"])
    else:
        report(args.pulled, args.gpu_hours, args.kernel_version)
