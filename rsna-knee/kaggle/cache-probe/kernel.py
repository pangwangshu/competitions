"""200 random training series plus the 15 public test series, CPU only."""
import json
import os

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"

from rsna_knee.reader.cache import run_probe
from rsna_knee.reader.paths import DATA, WORK

config = json.loads(os.environ["RK_CONFIG_JSON"])
run_probe(DATA, WORK, **config)
