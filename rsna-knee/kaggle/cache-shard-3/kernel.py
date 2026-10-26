"""Build one deterministic S1 cache shard using all available CPU cores."""
import json
import os

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"

from rsna_knee.reader.cache import build_shard
from rsna_knee.reader.paths import DATA, WORK

config = json.loads(os.environ["RK_CONFIG_JSON"])
build_shard(DATA, WORK, **config)
