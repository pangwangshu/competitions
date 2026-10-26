"""CPU-only: how long the attached S1 cache shards take to discover and read."""
import json
import time
from pathlib import Path

import numpy as np

t0 = time.time()
result = {}
from rsna_knee.evaluation import cache_paths, kaggle_inputs

found = kaggle_inputs()
result["discover_seconds"] = time.time() - t0
t = time.time()
paths = cache_paths([d for d in found["cache_dirs"] if d.name == "cache_train"])
result.update(series=len(paths), scan_seconds=time.time() - t)
files = [paths[s] for s in np.random.default_rng(0).choice(sorted(paths), 20, replace=False)]
t = time.time()
mb = sum(np.load(f).nbytes for f in files) / 1e6
result.update(read_mb=mb, read_seconds=time.time() - t, total_seconds=time.time() - t0)
print(json.dumps(result))
Path("/kaggle/working/probe.json").write_text(json.dumps(result))
