"""Reader-only submission: our checkpoints and inference config on the (hidden) test DICOMs."""
import json
import os

from rsna_knee.evaluation import find_one, sha256, submission
from rsna_knee.reader.paths import DATA

config = json.loads(os.environ["RK_CONFIG_JSON"])
paths = []
for c in config["ckpts"]:
    hits = find_one(c["file"])
    if len({sha256(h) for h in hits}) != 1:
        raise ValueError(f"Ambiguous checkpoint {c['file']}: {hits}")
    paths.append(hits[0])
submission(config, data=DATA, ckpt_paths=paths, out="/kaggle/working")
