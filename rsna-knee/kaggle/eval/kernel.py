"""Evaluate reader checkpoints on the S1 caches (OOF, Gold58, test) plus DICOM parity and timing."""
import json
import os

from rsna_knee.evaluation import run_kaggle

run_kaggle(json.loads(os.environ["RK_CONFIG_JSON"]))
