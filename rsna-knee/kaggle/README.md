# Private Kaggle workflow

Run from `competitions/rsna-knee` after `uv sync --locked --all-extras`. The account is fixed to `wangshu`.

These are my historical execution definitions. They depend on private datasets and kernel outputs, so another account cannot run them unchanged. [REPRODUCING.md](../REPRODUCING.md) lists the full input requirements, the local training path and the unreleased material. Moving to another account means changing the owner in `scripts/kx.py` together with the IDs, datasets and kernel dependencies in each `kernel.json`.

## Publishing the source

```bash
uv run python scripts/kx.py publish-code --message "S0 reader and metrics"
```

The first run creates the private dataset `wangshu/rsna-knee-code`; later runs create a new version when the source has changed. When it has not, the tool reads the remote manifest and reuses the existing version. Privacy and version are checked again after publishing.
The source ZIP uses a fixed file order and fixed timestamps, and contains the Python files under `src/rsna_knee/` plus the reader's LICENSE and NOTICE; `code-manifest.json` records the SHA-256 of every file and of the ZIP.
Kaggle extracts the ZIP into a directory of the same name; the kernel loader accepts both the original ZIP and the extracted layout and verifies every source file's hash.

The reader's source dataset is licensed Apache-2.0, and its license text and attribution are packaged with it. The code dataset as a whole is marked `other`, to indicate that rights to individual files belong to their respective authors.

## Defining, rendering and running kernels

Each kernel directory contains:

- `kernel.json`: id, title, code_file, GPU/internet settings and other data sources.
- `kernel.py`: the source script.
- `config.json`: optional run parameters.

```bash
uv run python scripts/kx.py render kaggle/s0-smoke --config kaggle/s0-smoke/config.json
uv run python scripts/kx.py push kaggle/s0-smoke --config kaggle/s0-smoke/config.json
uv run python scripts/kx.py wait wangshu/rsna-knee-s0-smoke --timeout 1800
uv run python scripts/kx.py pull wangshu/rsna-knee-s0-smoke experiments/S0-kaggle-smoke/output
```

`render` only writes `.rendered/kernel.py` and `kernel-metadata.json` locally.
`push` checks that the source matches the published ZIP, attaches the code dataset and the competition data, and publishes a private script. Before running, the rendered script verifies the source hashes, adds the source to `sys.path`, and passes the configuration in the environment variable `RK_CONFIG_JSON` and the code dataset path in `RK_CODE_ROOT`.
All run configuration is embedded in the single uploaded script, and the source script reads it like this:

```python
import json
import os

config = json.loads(os.environ["RK_CONFIG_JSON"])
```

`wait` polls every 120 seconds by default. It exits successfully on COMPLETE; on ERROR or CANCELLED it pulls the logs and exits non-zero. `--once` queries a single time. `pull` downloads all outputs and logs.
Kaggle's status and output APIs return the latest run, so the tools compare the latest version against the local receipt or an explicit `owner/slug/version` and refuse to switch silently to another version.

For cache kernels, pull only the manifests and acceptance reports instead of the full volume files:

```bash
uv run python scripts/kx.py pull wangshu/rsna-knee-cache-shard-0 experiments/S1/shards/0 \
  --file-pattern '\.(csv|json)$'
```

`--file-pattern` is a regular expression on file names; logs are always kept. S1 first runs `cache-probe` and `labels`. With the probe output in `experiments/S1/probe/`, `uv run python scripts/s1.py plan` writes the cache configuration and the `cache-shard-i` kernels.
`uv run python scripts/run_s1_cache.py --parallel 4 --interval 90` runs or resumes the full cache build; existing receipts and complete archives are reused, and pending shards take free slots first.
When the build is complete, run `uv run python scripts/s1.py verify`.

Outputs are downloaded only from terminal states (COMPLETE, ERROR, CANCELLED), so a running kernel's stale outputs are never picked up.
Listing uses 100 items per page with 3 seconds between pages; 429 and 5xx responses are retried with Retry-After or exponential backoff.
The intermediate cursor is saved in the output directory as `kx-output-partial.json`, so a repeated run resumes; it is deleted on success.

Local staging and publication receipts live in the gitignored `runs/kx/`. When an experiment is accepted, the relevant receipts, outputs and metrics are archived under `experiments/<id>/`. If authentication expires, run `kaggle auth login` and retry.

Platform field references: [dataset metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md), [kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).
The local tools use the installed Kaggle Python API, which calls the same services as the corresponding CLI subcommands.

## Generated files not in this repository

The `kernel.py` files of `teacher/`, `teacher-s6/`, `submit-blend/` and `submit-blend-s6/` embed public notebooks by nartaa and heliosli and are not redistributed here. Regenerate them from the public notebooks with `scripts/build_teacher_kernel.py`, `build_teacher_s6_kernel.py`, `build_blend_kernel.py` and `build_blend_s6_kernel.py`.
