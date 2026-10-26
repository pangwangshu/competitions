"""Run the packaged reader and compare it with the saved local baseline.

Run from the repository root. The report includes input, checkpoint, and code
hashes so the numerical comparison can be traced to the exact artifacts used.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.reader import __version__
from rsna_knee.reader.infer import run
from rsna_knee.reader.knee import LABELS
from rsna_knee.reader.paths import DATA
from rsna_knee.reader.runtime import resolve_device


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--ckpt-dir", default="models/convnext_reader")
    ap.add_argument("--reference", default="runs/baseline_0944/_own_local.csv")
    ap.add_argument("--out-dir", default="experiments/S0-baseline-migration")
    ap.add_argument("--device", choices=("cuda", "mps", "cpu"), default="mps")
    ap.add_argument("--tolerance", type=float, default=1e-5)
    args = ap.parse_args()
    device = resolve_device(args.device)
    if args.tolerance < 0:
        ap.error("--tolerance must be nonnegative")
    ckpts = sorted(Path(args.ckpt_dir).glob("cnxt_v0_fold*.pt"))
    reference = pd.read_csv(args.reference, index_col="StudyInstanceUID")
    if reference.index.has_duplicates or list(reference.columns) != LABELS:
        raise ValueError("The reference must contain unique studies and the 12 expected labels in order")

    print(f"device={device}, amp=none, {len(ckpts)} checkpoints", flush=True)
    t0 = time.perf_counter()
    predictions = run(args.data, "test", ckpts, k=12, bs=1, workers=0, device=str(device), amp="none")
    elapsed = time.perf_counter() - t0
    if predictions.index.has_duplicates or set(predictions.index) != set(reference.index):
        raise ValueError("Prediction study IDs do not match the baseline")
    predictions = predictions.loc[reference.index, LABELS]
    if not np.isfinite(predictions.values).all() or not np.isfinite(reference.values).all():
        raise ValueError("Predictions and reference must contain only finite values")
    delta = (predictions - reference).abs()
    passed = bool((delta.values <= args.tolerance).all())
    worst = np.unravel_index(delta.values.argmax(), delta.shape)
    data = Path(args.data)
    source = Path(__file__).resolve().parents[1] / "src" / "rsna_knee" / "reader"
    dicoms = sorted((data / "test_series").rglob("*.dcm"))
    # Fingerprint file names and contents without storing DICOM metadata in the report.
    dicom_digest = hashlib.sha256()
    for path in dicoms:
        dicom_digest.update(path.relative_to(data).as_posix().encode())
        dicom_digest.update(bytes.fromhex(sha256(path)))
    report = {
        "experiment": "S0-baseline-migration",
        "passed": passed,
        "package_version": __version__,
        "device": str(device),
        "amp": "none",
        "k_eval": 12,
        "batch_size": 1,
        "workers": 0,
        "study_count": len(predictions),
        "prediction_count": int(predictions.size),
        "tolerance": args.tolerance,
        "max_absolute_difference": float(delta.values.max()),
        "mean_absolute_difference": float(delta.values.mean()),
        "worst_study": str(delta.index[worst[0]]),
        "worst_label": delta.columns[worst[1]],
        "max_difference_by_label": {c: float(delta[c].max()) for c in LABELS},
        "elapsed_seconds": elapsed,
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in (
            "torch", "timm", "numpy", "pandas", "opencv-python-headless", "pydicom",
            "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "scikit-learn",
        )},
        "reference": {"path": str(args.reference), "sha256": sha256(args.reference)},
        "checkpoints": {p.name: sha256(p) for p in ckpts},
        "reader_source_sha256": {p.name: sha256(p) for p in sorted(source.glob("*.py"))},
        "verification_script_sha256": sha256(__file__),
        "dependency_lock_sha256": sha256(source.parents[2] / "uv.lock"),
        "test_metadata_sha256": {name: sha256(data / name) for name in ("test.csv", "test_series.csv")},
        "dicom_file_count": len(dicoms),
        "dicom_files_sha256": dicom_digest.hexdigest(),
    }
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv")
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'PASS' if passed else 'FAIL'}: {predictions.size} values, max absolute difference={delta.values.max():.9g}; {elapsed:.1f}s")
    print(f"Report: {output / 'verification.json'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
