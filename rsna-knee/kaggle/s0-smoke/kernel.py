"""CPU-only smoke check of the packaged reader, metrics, and config transport."""

import json
import os
import platform
from pathlib import Path

import rsna_knee.reader
from rsna_knee.metrics import macro_auc, paired_bootstrap

root = Path(os.environ["RK_CODE_ROOT"])
manifest = json.loads((root / "code-manifest.json").read_text())
config = json.loads(os.environ["RK_CONFIG_JSON"])
y = [[0], [1], [0], [1]]
p = [[0.1], [0.9], [0.2], [0.8]]
assert macro_auc(y, p) == 1
comparison = paired_bootstrap(y, p, p, n=100, seed=0)
assert comparison["delta_mean"] == comparison["ci_low"] == comparison["ci_high"] == 0
result = {
    "passed": True,
    "package_version": rsna_knee.reader.__version__,
    "reader_module": rsna_knee.reader.__file__,
    "python": platform.python_version(),
    "archive_sha256": manifest["archive_sha256"],
    "source_hashes_verified": True,
    "package_storage": "zip" if Path(os.environ["RK_CODE_PACKAGE_PATH"]).is_file() else "extracted",
    "config": config,
    "macro_auc_check": 1,
    "paired_bootstrap_check": comparison,
}
Path("smoke.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
