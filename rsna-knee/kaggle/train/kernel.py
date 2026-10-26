"""Train reader folds on both T4s from the S1 caches; records timing and GPU utilisation."""
import json
import os

from rsna_knee.training import run_kaggle

run_kaggle(json.loads(os.environ["RK_CONFIG_JSON"]))
