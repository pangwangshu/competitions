"""Diagnose byte-level preprocessing differences across Kaggle and macOS."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"

import cv2
import numpy as np
import pandas as pd

from rsna_knee.reader import preprocess as p
from rsna_knee.reader.cache import write_json
from rsna_knee.reader.paths import DATA, WORK

cv2.setNumThreads(1)
se = pd.read_csv(Path(DATA) / "test_series.csv").sort_values("SeriesInstanceUID")
versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
            if any(k in d.metadata["Name"].lower() for k in ["numpy", "opencv", "pylibjpeg", "pydicom", "scikit-learn"])}
rows = []
original = p._resample
original_resize = cv2.resize
for mode in ["optimized", "scalar", "float64"]:
    cv2.setUseOptimized(mode != "scalar")
    cv2.resize = original_resize if mode != "float64" else lambda img, *args, **kwargs: original_resize(img.astype(np.float64), *args, **kwargs)
    directory = Path(WORK) / mode
    directory.mkdir(exist_ok=True)
    for i, row in enumerate(se.itertuples(index=False)):
        frames = []
        def capture(*args):
            frame = original(*args)
            frames.append(frame)
            return frame
        p._resample = capture
        vol, info = p.load_series(Path(DATA) / "test_series" / row.StudyInstanceUID / row.SeriesInstanceUID, row.Anatomical_Plane)
        floats = np.stack(frames)
        lo, hi = np.percentile(floats[:, ::4, ::4], [1, 99.5])
        path = directory / f"{row.SeriesInstanceUID}.npy"
        np.save(path, vol, allow_pickle=False)
        if i == 0:
            np.save(directory / "first_resampled_float32.npy", floats, allow_pickle=False)
        rows.append({"series": row.SeriesInstanceUID, "mode": mode,
                     "resampled_sha256": hashlib.sha256(floats.tobytes()).hexdigest(),
                     "lo": float(lo), "hi": float(hi), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "n": len(vol), "bytes": path.stat().st_size, "study": row.StudyInstanceUID})
p._resample = original
cv2.resize = original_resize
pd.DataFrame(rows).to_csv(Path(WORK) / "parity_series.csv", index=False)
write_json(Path(WORK) / "parity_environment.json", {"versions": versions, "cv2_build": cv2.getBuildInformation()})
print(json.dumps(versions, indent=2), flush=True)
