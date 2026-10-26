# Reproducing the code and experiments

Run commands from this directory (`competitions/rsna-knee`), not the parent monorepo.

This release contains executable source, unit tests, figure inputs and experiment summaries. It does **not** contain a self-contained archive of the reported experiments: the per-run predictions/configs, trained student weights, private Kaggle datasets and original reference predictions are not included. Rendering a figure from its CSV is not the same as recomputing its statistics from predictions.

## 1. Environment and tests

```bash
uv sync --locked --all-extras
uv run pytest -q
uv run python report/make_figures.py
```

The release review used Python 3.12 on macOS. Without competition data, expect 61 tests to pass and one DICOM test to skip. Set `RK_DATA` to a competition data directory to enable that test. The lock file pins the local Python environment; it does not pin the complete Kaggle GPU image, CUDA libraries or external model downloads.

## 2. Inputs

Obtain the competition files through an account that has accepted the competition's access requirements. Expected layout:

```text
data/raw/
  train.csv
  train_series.csv
  train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
  test.csv
  test_series.csv
  test_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
models/convnext_reader/
  cnxt_v0_fold0.pt
  cnxt_v0_fold1.pt
  cnxt_v0_fold2.pt
```

The baseline weights are from `goodpjw2008/rsna-knee-2-5d-convnext-reader`. Teacher sources and weights are identified in the report's references and the builder docstrings. The student checkpoints used in the reported submissions are not distributed.

Download the four label tables at the versions in the committed manifest:

```bash
uv run python scripts/fetch_s1_labels.py
```

This uses your authenticated Kaggle account, verifies each SHA-256 in `experiments/S1/source_datasets.json`, and leaves the manifest unchanged. Downloads require network access. Existing matching files are reused.

## 3. Local labels, cache and one training run

The following recipe does not depend on the author's private Kaggle datasets. It requires the full training images, substantial storage (the original cache was about 110 GB) and a suitable GPU for practical training time.

```bash
export RK_DATA="$PWD/data/raw"
export RK_EXT="$PWD/data/ext"
export RK_WORK="$PWD/runs/work"
export RK_RUNS="$PWD/runs"

uv run python -m rsna_knee.reader.make_labels v0
uv run python -m rsna_knee.reader.preprocess train

uv run python -m rsna_knee.reader.train \
  --name tiny-mirror-seed0 --fold 0 --labels labels_v0 \
  --backbone convnext_tiny.fb_in22k_ft_in1k \
  --seed 0 --epochs 14 --mirror_p 0.5 \
  --res 256 --k 8 --k_eval 12 --bs 8 \
  --lr 0.0001 --lr_head 0.0005 --wd 0.02 --ema 0.998 \
  --workers 2 --grad_ckpt 1 --eval_every 0 --device cuda --amp fp16

uv run python -m rsna_knee.reader.oof labels_v0 tiny-mirror-seed0
```

The pretrained encoder is downloaded by timm on first use. Inspect the cache metadata for missing or failed series before training. The fixed fold table should contain 58 Gold58 rows with fold −1 and weak-fold sizes 870, 870, 870, 870, 869. With the recorded inputs, `labels_v0.csv` has SHA-256 `043e13ea59551bc394ee1a1c2c3c946ad742f3a1649cf90cc89fd70c5c75234d`.

Repeat with `--seed 1` and a different `--name` for the second seed. The matched control changes `--mirror_p` to 0. The Small recipe uses `convnext_small.fb_in22k_ft_in1k`, 10 epochs and no mirror augmentation. Final C* uses one Tiny and one Small model per fold; two-seed replication refers to the fold-0 recipe comparisons, not 20 final models.

The command above reports the training-time evaluation (`k_eval=12`, no mirror TTA). The report's I*2 results use `k=24` and anatomical mirror TTA through `rsna_knee.evaluation` / `kaggle/eval/kernel.py`; changing `--k` in the simple inference CLI alone does not enable TTA. See the config generators in `scripts/s3.py` and `scripts/s5.py`. These historical generators also require S1 artifacts and the checkpoint manifest described below.

An interrupted training job restores model, EMA, optimizer, scheduler and scaler state, but not RNG states or persistent data-loader worker states. Resume is functional, not bitwise-equivalent to an uninterrupted run. The release review did not rerun full GPU training.

## 4. Baseline inference and parity

```bash
uv run python -m rsna_knee.reader.infer --device cpu --amp none \
  --ckpts models/convnext_reader/cnxt_v0_fold0.pt \
          models/convnext_reader/cnxt_v0_fold1.pt \
          models/convnext_reader/cnxt_v0_fold2.pt \
  --out runs/reader_test.csv
```

Use `--device mps` or `--device cuda` where supported. This produces predictions from the three visible test studies when using the competition's visible test files.

`scripts/verify_reader_baseline.py` additionally needs an independently produced reference CSV via `--reference`. The default points to a development artifact that is not shipped. Match the reference device and precision when testing tight numerical parity. In this review, CPU inference completed, but its maximum difference from the historical MPS reference was `3.9104e-5`, above the verifier's `1e-5` threshold. That parity check failed; its tolerance was not changed.

## 5. Distillation and statistical comparisons

Teacher inference must cover fold-0 training studies and Gold58, excluding held-out fold 0. The supplied `kaggle/teacher-s6/config.json` selects `fold0_train`, which includes both the 3,479 training studies and Gold58. The separate gold config runs only Gold58 for a diagnostic check. Use the resulting probability table indexed by `StudyInstanceUID` with the 12 finding columns.

```bash
uv run python scripts/build_distill_labels.py \
  --labels runs/work/labels_v0.csv \
  --teacher runs/teacher_probs_s6.csv --teacher-name S6 \
  --out runs/work/labels_t4s.csv
```

Train two Tiny mirror students with the recipe above and `--labels labels_t4s`, seeds 0 and 1. Missing teacher rows keep v0. Do not supply held-out teacher predictions to this builder: it fits on every non-gold row present in the teacher table. With the historical teacher outputs, the regenerated target SHA-256 is `0e6edb6c8659e51789b8f15a978bb8d613857a3ddc4704093295a6f8193415c2`.

For two evaluation directories containing aligned `preds_oof.csv` and `preds_gold.csv`:

```bash
uv run python scripts/compare.py runs/control runs/variant \
  --labels runs/work/labels_v0.csv --labels-changed \
  --out runs/distillation-comparison.json
```

Use `--labels-changed` for distillation. Both offline deltas are printed, but the verdict is `gold_only`; OOF improvement alone must not trigger adoption. Average each arm's two seeds in probability space before comparing the ensembles. Bootstrap intervals concern evaluation-study sampling, not training randomness.

## 6. Checks against the development records

Before release, the public code was run against the inputs and records kept in my private development repository:

- With the competition DICOMs available through `RK_DATA`, all 62 tests pass, including the DICOM input-consistency test.
- `make_labels v0` regenerates the 4,407-row `labels_v0.csv` and its fold table byte for byte (SHA-256 above).
- `build_distill_labels.py`, refitted on the historical teacher outputs, regenerates `labels_t4s.csv` byte for byte (SHA-256 above).
- Recomputed from the original per-study predictions, the distillation gain is OOF +0.009343 [+0.006646, +0.012187] and Gold58 +0.010845 [+0.000358, +0.021393], and C\* against Tiny × 5 is OOF +0.005248 [+0.004237, +0.006294], matching the report.
- CPU inference with the three baseline checkpoints produced all 36 probabilities for the three visible test studies in about 33 seconds; its parity against the historical MPS reference failed as described in section 4.

These checks support the core calculations. They used inputs that are not in this release, and they did not retrain the models, rebuild the 110 GB cache or rerun the hidden test set.

## 7. Historical Kaggle workflow and missing release artifacts

The definitions in `kaggle/` are the author's execution records. `scripts/kx.py` requires the `wangshu` account, and the definitions reference private `wangshu/...` datasets and kernel outputs. They cannot be pushed unchanged by another account. Porting requires changing the owner, dataset IDs and kernel dependencies together, and rebuilding cache/label/checkpoint datasets under that account. See [kaggle/README.md](kaggle/README.md).

The stage generators require artifacts not included here, notably `experiments/S1/labels/labels_v0.csv`, S1 cache manifests and `runs/kx/ckpts/ckpt-manifest.json`. Builders for teacher/blend kernels take local downloaded `.py`/`.ipynb` paths; they do not fetch sources. Their embedded hashes and patch-anchor checks fail if upstream notebooks have changed.

To make **all reported experiments independently reproducible from the public release**, a follow-up artifact release needs:

- Exact per-run configs, dependency/image versions, source notebook versions/hashes, and the checkpoint manifest.
- Per-study predictions and the evaluation targets/fold table, or a documented access-controlled way to regenerate them, plus the analysis commands behind each report table.
- Downloadable student checkpoints or a verified path to retrain, evaluate and assemble the final models under a new account.

Until then, the supported claim is executable source with locally verified core calculations, not a complete reproduction of every reported run from a fresh public checkout.
