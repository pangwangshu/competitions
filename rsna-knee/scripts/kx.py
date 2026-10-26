"""Publish private code and run private Kaggle script kernels.

All local staging/receipts live in runs/kx/. Kernel definitions live in
kaggle/<name>/kernel.json. Only source Python files from src/rsna_knee are
uploaded with the reader's license notices as a deterministic ZIP; local data,
models, and credentials are excluded.
"""

import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runs" / "kx"
OWNER = "wangshu"
CODE_DATASET = f"{OWNER}/rsna-knee-code"
COMPETITION = "rsna-knee-abnormality-detection"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_api():
    from kaggle import api

    if api.get_config_value("username") != OWNER:
        raise ValueError(f"Expected Kaggle account {OWNER}; authenticate that account before running kx")
    return api


def package_code():
    from rsna_knee import __version__

    source = ROOT / "src" / "rsna_knee"
    stage = STATE / "code"
    stage.mkdir(parents=True, exist_ok=True)
    files = sorted([*source.rglob("*.py"), source / "reader" / "LICENSE", source / "reader" / "NOTICE"])
    if not files:
        raise ValueError("No package source files found")
    hashes = {}
    with zipfile.ZipFile(stage / "rsna_knee.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if not path.resolve().is_relative_to(source.resolve()):
                raise ValueError(f"Source file points outside the package: {path}")
            relative = path.relative_to(source.parent).as_posix()
            content = path.read_bytes()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, content)
            hashes[relative] = hashlib.sha256(content).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src/rsna_knee"], cwd=ROOT, text=True).strip())
    manifest = {"package_version": __version__, "git_commit": commit, "source_dirty": dirty,
                "archive_sha256": digest(stage / "rsna_knee.zip"), "source_sha256": hashes}
    write_json(stage / "code-manifest.json", manifest)
    write_json(stage / "dataset-metadata.json", {
        "id": CODE_DATASET, "title": "RSNA Knee Code", "licenses": [{"name": "other"}],
        "description": "Private development package. Reader adapted from "
                       "goodpjw2008/rsna-knee-2-5d-convnext-reader under Apache-2.0 (see reader/LICENSE and NOTICE). "
                       "Copyright remains with the respective authors; no blanket license is assigned to the other project files.",
    })
    return stage, manifest


def dataset_info(api):
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

    request = ApiGetDatasetRequest()
    request.owner_slug, request.dataset_slug = CODE_DATASET.split("/")
    with api.build_kaggle_client() as client:
        return client.datasets.dataset_api_client.get_dataset(request)


def owned_code_dataset(api):
    # Kaggle may return 403 for a nonexistent dataset. An authenticated listing
    # of our own datasets distinguishes that from an expired/invalid login.
    page = 1
    while True:
        datasets = api.dataset_list(mine=True, page=page) or []
        for dataset in datasets:
            if dataset.ref == CODE_DATASET:
                return dataset
        if not datasets:
            return None
        page += 1


def check_response(response):
    if response is None:
        raise RuntimeError("Kaggle returned no response")
    if response.error:
        raise RuntimeError(response.error)
    for name in ["invalid_dataset_sources", "invalid_competition_sources", "invalid_kernel_sources", "invalid_model_sources"]:
        invalid = getattr(response, name, None)
        if invalid:
            raise RuntimeError(f"Kaggle rejected {name}: {invalid}")


def remote_code_manifest(api):
    output = STATE / "remote-code"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "code-manifest.json"
    path.unlink(missing_ok=True)
    api.dataset_download_file(CODE_DATASET, "code-manifest.json", path=str(output), force=True, quiet=True)
    return read_json(path)


def publish_code(api, message, timeout=900):
    from requests.exceptions import HTTPError

    if timeout <= 0:
        raise ValueError("timeout must be positive")
    stage, manifest = package_code()
    existing = owned_code_dataset(api)
    if existing is not None:
        existing_info = dataset_info(api)
        if not existing_info.is_private:
            raise ValueError("The code dataset is public; refusing to update it under the private-only plan")
        remote = remote_code_manifest(api)
        if remote["archive_sha256"] == manifest["archive_sha256"]:
            manifest = remote
            operation, expected_version = "reuse", existing_info.current_version_number
        else:
            response = api.dataset_create_version(str(stage), message, quiet=True)
            check_response(response)
            operation, expected_version = "version", existing_info.current_version_number + 1
    else:
        response = api.dataset_create_new(str(stage), public=False, quiet=True)
        check_response(response)
        operation, expected_version = "create", 1
    write_json(STATE / "code-publication-pending.json", {
        "dataset": CODE_DATASET, "operation": operation, "expected_version": expected_version, "manifest": manifest,
    })
    deadline = time.monotonic() + timeout
    while True:
        try:
            state = json.loads(api.dataset_status(CODE_DATASET, format="json"))
            print(f"Code dataset: {state['status']}, v{state['current_version_number']}", flush=True)
            if state["status"].lower() == "ready" and state["current_version_number"] == expected_version:
                break
            if state["status"].lower() in {"error", "failed"}:
                raise RuntimeError(f"Dataset processing failed: {state['status']}")
        except HTTPError as error:
            if error.response.status_code not in {403, 404} or operation == "reuse":
                raise
            # A newly accepted dataset may briefly be invisible to status/read
            # endpoints. Verify that authentication still works before retrying.
            api.dataset_list(mine=True, page=1)
            print("Code dataset accepted; waiting for status indexing", flush=True)
        if time.monotonic() >= deadline:
            raise TimeoutError("Code dataset is not ready yet; check kaggle datasets status")
        time.sleep(min(15, max(0, deadline - time.monotonic())))
    info = dataset_info(api)
    if not info.is_private:
        raise RuntimeError("Kaggle reports that the code dataset is not private")
    receipt = {"dataset": CODE_DATASET, "version": info.current_version_number,
               "is_private": True, "operation": operation, "manifest": manifest}
    write_json(STATE / "code-publication.json", receipt)
    (STATE / "code-publication-pending.json").unlink(missing_ok=True)
    return receipt


def render_kernel(kernel_dir, config=None, expected_archive_sha256=None, expected_source_sha256=None):
    from slugify import slugify

    kernel_dir = Path(kernel_dir).resolve()
    definition = read_json(kernel_dir / "kernel.json")
    slug = definition["id"]
    if slug.count("/") != 1 or slug.split("/")[0] != OWNER:
        raise ValueError(f"Kernel id must belong to {OWNER} and contain no version")
    if slugify(definition["title"]) != slug.split("/")[1]:
        raise ValueError("Kernel title must resolve to the explicit id; Kaggle derives new slugs from the title")
    if definition.get("is_private", True) is not True:
        raise ValueError("Only private kernels are authorized")
    if "id_no" in definition:
        raise ValueError("Use an explicit kernel id rather than id_no")
    source = (kernel_dir / definition["code_file"]).resolve()
    if source.suffix != ".py" or not source.is_relative_to(kernel_dir):
        raise ValueError("The kernel source must be a Python file inside its kernel directory")
    config = {} if config is None else config
    if not isinstance(config, dict):
        raise ValueError("Kernel config must be a JSON object")
    owner, dataset = CODE_DATASET.split("/")
    candidates = [f"/kaggle/input/{dataset}", f"/kaggle/input/datasets/{owner}/{dataset}"]
    # Kaggle pushes only the single code_file. Embed configuration and source
    # in that script; source compilation preserves docstrings/future imports.
    bootstrap = (
        "import hashlib as _rk_hashlib\nimport json as _rk_json\nimport os as _rk_os\nimport sys as _rk_sys\n"
        "import zipfile as _rk_zipfile\n"
        "from pathlib import Path as _rk_Path\n"
        f"_rk_candidates = {candidates!r}\n"
        "_rk_root = next((_rk_Path(p) for p in _rk_candidates if (_rk_Path(p) / 'code-manifest.json').is_file()), None)\n"
        "if _rk_root is None:\n    raise RuntimeError('Attached code dataset does not contain code-manifest.json')\n"
        "_rk_manifest = _rk_json.loads((_rk_root / 'code-manifest.json').read_text())\n"
        f"_rk_expected = {expected_archive_sha256!r}\n"
        f"_rk_expected_files = {expected_source_sha256!r}\n"
        "if _rk_expected and _rk_manifest['archive_sha256'] != _rk_expected:\n"
        "    raise RuntimeError('Attached code manifest does not match the pushed code version')\n"
        "if _rk_expected_files is not None and _rk_manifest['source_sha256'] != _rk_expected_files:\n"
        "    raise RuntimeError('Attached source manifest does not match the pushed code version')\n"
        "_rk_archive = _rk_root / 'rsna_knee.zip'\n"
        "if _rk_archive.is_file():\n"
        "    if _rk_expected and _rk_hashlib.sha256(_rk_archive.read_bytes()).hexdigest() != _rk_expected:\n"
        "        raise RuntimeError('Attached code archive has changed')\n"
        "    _rk_home = _rk_archive\n"
        "    with _rk_zipfile.ZipFile(_rk_archive) as _rk_z:\n"
        "        _rk_bytes = {name: _rk_z.read(name) for name in _rk_manifest['source_sha256']}\n"
        "else:\n"
        "    _rk_home = next((p for p in (_rk_root, _rk_root / 'rsna_knee') if (p / 'rsna_knee' / '__init__.py').is_file()), None)\n"
        "    if _rk_home is None:\n        raise RuntimeError('The extracted reader package is missing')\n"
        "    _rk_bytes = {name: (_rk_home / name).read_bytes() for name in _rk_manifest['source_sha256']}\n"
        "for _rk_name, _rk_hash in _rk_manifest['source_sha256'].items():\n"
        "    if _rk_hashlib.sha256(_rk_bytes[_rk_name]).hexdigest() != _rk_hash:\n"
        "        raise RuntimeError('Attached source file has changed: ' + _rk_name)\n"
        "_rk_sys.path.insert(0, str(_rk_home))\n"
        "_rk_os.environ['RK_CODE_ROOT'] = str(_rk_root)\n"
        "_rk_os.environ['RK_CODE_PACKAGE_PATH'] = str(_rk_home)\n"
        f"_rk_os.environ['RK_CONFIG_JSON'] = {json.dumps(config, ensure_ascii=False)!r}\n"
        f"exec(compile({source.read_text()!r}, {source.name!r}, 'exec'))\n"
    )
    generated = kernel_dir / ".rendered" / "kernel.py"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(bootstrap)
    metadata = {k: v for k, v in definition.items() if k != "session_timeout_seconds"}
    metadata.update({"code_file": ".rendered/kernel.py", "language": "python", "kernel_type": "script",
                     "is_private": True, "enable_gpu": definition.get("enable_gpu", False),
                     "enable_tpu": False, "enable_internet": definition.get("enable_internet", False)})
    metadata["dataset_sources"] = list(dict.fromkeys([CODE_DATASET, *definition.get("dataset_sources", [])]))
    metadata["competition_sources"] = list(dict.fromkeys([COMPETITION, *definition.get("competition_sources", [])]))
    metadata.setdefault("kernel_sources", [])
    metadata.setdefault("model_sources", [])
    write_json(kernel_dir / "kernel-metadata.json", metadata)
    return metadata, generated


def kernel_info(api, slug):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    owner, kernel = slug.split("/")
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = owner, kernel
    with api.build_kaggle_client() as client:
        return client.kernels.kernels_api_client.get_kernel(request).metadata


def push_kernel(api, kernel_dir, config=None):
    publication = read_json(STATE / "code-publication.json")
    if publication["dataset"] != CODE_DATASET or publication["is_private"] is not True:
        raise ValueError("A verified private code publication is required")
    _, manifest = package_code()
    if manifest["archive_sha256"] != publication["manifest"]["archive_sha256"]:
        raise ValueError("Package source has changed; run publish-code before pushing a kernel")
    code_info = dataset_info(api)
    if not code_info.is_private or code_info.current_version_number != publication["version"]:
        raise ValueError("Code dataset privacy/version differs from the local publication receipt")
    metadata, script = render_kernel(kernel_dir, config, manifest["archive_sha256"], manifest["source_sha256"])
    # A server-side session limit stops a hung run without anyone opening the web UI
    # (the public API exposes no session id to cancel). Cancelled runs keep no output,
    # so kernels must also finish on their own before this limit.
    timeout = read_json(Path(kernel_dir) / "kernel.json").get("session_timeout_seconds")
    if not timeout:
        raise ValueError("kernel.json needs session_timeout_seconds")
    response = api.kernels_push(str(Path(kernel_dir).resolve()), timeout=str(int(timeout)))
    check_response(response)
    # SaveKernel may return owner/slug/version in ref; readback needs owner/slug.
    slug = metadata["id"]
    info = kernel_info(api, slug)
    if not info.is_private:
        raise RuntimeError("Kaggle reports that the kernel is not private")
    receipt = {"kernel": slug, "version": response.version_number,
               "url": response.url, "is_private": True, "code_dataset": CODE_DATASET,
               "code_dataset_version": publication["version"], "code_archive_sha256": manifest["archive_sha256"],
               "rendered_script_sha256": digest(script), "config": config or {}, "enable_gpu": metadata["enable_gpu"],
               "session_timeout_seconds": int(timeout)}
    write_json(STATE / "kernels" / f"{receipt['kernel'].split('/')[1]}.json", receipt)
    return receipt


def expected_kernel(api, target):
    parts = target.split("/")
    if len(parts) not in {2, 3} or parts[0] != OWNER:
        raise ValueError(f"Expected {OWNER}/kernel-slug[/version]")
    slug = "/".join(parts[:2])
    receipt_path = STATE / "kernels" / f"{parts[1]}.json"
    version = int(parts[2]) if len(parts) == 3 else (read_json(receipt_path)["version"] if receipt_path.exists() else None)
    info = kernel_info(api, slug)
    if version is not None and info.current_version_number != version:
        raise ValueError("The latest kernel version has changed; status/output APIs refer to the latest run")
    if not info.is_private:
        raise ValueError("Only private kernels are authorized")
    return slug, info.current_version_number


def _pause(seconds):
    while seconds > 0:
        chunk = min(30, seconds)
        time.sleep(chunk)
        seconds -= chunk


def _read_with_retry(action, attempts=7):
    from requests.exceptions import HTTPError

    for attempt in range(attempts):
        try:
            return action()
        except HTTPError as error:
            code = error.response.status_code
            if code not in {429, 500, 502, 503, 504} or attempt == attempts - 1:
                raise
            retry_after = error.response.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else min(60 * 2**attempt, 300)
            print(f"Kaggle HTTP {code}; retrying the same read in {delay:g}s", flush=True)
            _pause(delay)


def pull_kernel(api, target, out_dir, file_pattern=None):
    slug, version = expected_kernel(api, target)
    state = api.kernels_status(slug)
    status = getattr(state.status, "name", str(state.status)).upper()
    if status not in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}:
        raise ValueError(f"Refusing output download while {slug} is {status}; wait for a terminal run")
    # Cache kernels have thousands of files; avoid hundreds of list API calls.
    kwargs = {"force": True, "quiet": True, "page_size": 100}
    if file_pattern is not None:
        kwargs["file_pattern"] = file_pattern
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "kx-output-partial.json"
    token, files, done = "", [], False
    if checkpoint.exists():
        saved = read_json(checkpoint)
        if (saved["kernel"], saved["version"], saved["file_pattern"]) == (slug, version, file_pattern) and all(Path(p).exists() for p in saved["files"]):
            token, files, done = saved["next_page_token"], saved["files"], saved["done"]
    while not done:
        # An explicit (even empty) page_token disables the SDK's unpaced recursion.
        page_files, next_token = _read_with_retry(lambda: api.kernels_output(slug, str(out_dir), page_token=token, **kwargs))
        files = list(dict.fromkeys([*files, *map(str, page_files)]))
        token, done = next_token, not bool(next_token)
        write_json(checkpoint, {"kernel": slug, "version": version, "file_pattern": file_pattern,
                               "next_page_token": token, "done": done, "files": files})
        if not done:
            _pause(3)
    _pause(3)
    log = _read_with_retry(lambda: api.kernels_logs(slug))
    if log:
        log_path = out_dir / f"{slug.split('/')[1]}.log"
        log_path.write_text(log)
        files = list(dict.fromkeys([*files, str(log_path)]))
    # Recheck after download so a concurrent push cannot silently mix runs.
    expected_kernel(api, f"{slug}/{version}")
    receipt = {"kernel": slug, "version": version, "file_pattern": file_pattern,
               "status_at_download": status,
               "files": [str(Path(p)) for p in files]}
    write_json(Path(out_dir) / "kx-output.json", receipt)
    checkpoint.unlink(missing_ok=True)
    return receipt


def wait_kernel(api, target, interval=120, timeout=43200, once=False):
    if interval <= 0 or timeout <= 0:
        raise ValueError("interval and timeout must be positive")
    deadline = time.monotonic() + timeout
    initial_slug, initial_version = expected_kernel(api, target)
    target = f"{initial_slug}/{initial_version}"
    while True:
        slug, version = expected_kernel(api, target)
        response = api.kernels_status(slug)
        status = getattr(response.status, "name", str(response.status)).upper()
        receipt = {"kernel": slug, "version": version, "status": status, "failure_message": response.failure_message}
        write_json(STATE / "kernels" / f"{slug.split('/')[1]}-status.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        if status == "COMPLETE":
            return receipt
        if status in {"ERROR", "CANCELLED", "CANCELED"}:
            out = STATE / "errors" / slug.split("/")[1] / f"v{version}"
            try:
                pull_kernel(api, f"{slug}/{version}", out, file_pattern=r"\.(csv|json)$")
            except Exception as error:
                print(f"Could not download error output: {error}", flush=True)
            raise RuntimeError(f"{slug} ended with {status}: {response.failure_message}; logs: {out}")
        if once:
            return receipt
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{slug} is still {status}; rerun wait to continue polling")
        wake = min(deadline, time.monotonic() + interval)
        while time.monotonic() < wake:
            time.sleep(min(30, max(0, wake - time.monotonic())))


CKPT_DATASET = f"{OWNER}/rsna-knee-ckpts"


def publish_ckpts(api, files, message):
    """Private dataset of trained checkpoints, so later kernels can attach them by name.

    Kaggle attaches only a kernel's latest output, so checkpoints from earlier training
    versions must live in a dataset. Each version holds every checkpoint published so far.
    """
    stage = STATE / "ckpts"
    stage.mkdir(parents=True, exist_ok=True)
    for f in files:
        f = Path(f)
        target = stage / f.name
        if target.exists() and digest(target) != digest(f):
            raise ValueError(f"A different {f.name} is already published; rename it")
        if not target.exists():
            target.write_bytes(f.read_bytes())
    manifest = {p.name: digest(p) for p in sorted(stage.iterdir())
                if p.suffix in {".pt", ".csv"}}  # checkpoints and extra label tables
    write_json(stage / "ckpt-manifest.json", manifest)
    write_json(stage / "dataset-metadata.json", {"id": CKPT_DATASET, "title": "RSNA Knee Ckpts",
                                                 "licenses": [{"name": "other"}]})
    if any(d.ref == CKPT_DATASET for d in api.dataset_list(mine=True, search="rsna-knee-ckpts") or []):
        check_response(api.dataset_create_version(str(stage), message, quiet=True))
    else:
        check_response(api.dataset_create_new(str(stage), public=False, quiet=True))
    return {"dataset": CKPT_DATASET, "files": manifest}


def quota(api):
    """Weekly GPU usage; experiment GPU hours are measured as the difference around a run."""
    response = api.quota_view()
    gpu = response.gpu_quota
    return {"gpu_used_hours": gpu.time_used.total_seconds() / 3600,
            "gpu_total_hours": gpu.total_time_allowed.total_seconds() / 3600,
            "refresh": response.quota_refresh_time.isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish-code")
    publish.add_argument("--message", default="Update rsna_knee package")
    publish.add_argument("--timeout", type=float, default=900)
    push = commands.add_parser("push")
    push.add_argument("kernel_dir", type=Path)
    push.add_argument("--config", type=Path)
    render = commands.add_parser("render", help="Render and inspect a kernel locally without uploading")
    render.add_argument("kernel_dir", type=Path)
    render.add_argument("--config", type=Path)
    wait = commands.add_parser("wait")
    wait.add_argument("slug")
    wait.add_argument("--interval", type=float, default=120)
    wait.add_argument("--timeout", type=float, default=43200)
    wait.add_argument("--once", action="store_true")
    pull = commands.add_parser("pull")
    pull.add_argument("slug")
    pull.add_argument("out_dir", type=Path)
    pull.add_argument("--file-pattern", help="Download only matching output filenames (regex); logs are always included")
    commands.add_parser("quota")
    ckpts = commands.add_parser("publish-ckpts")
    ckpts.add_argument("files", nargs="+", type=Path)
    ckpts.add_argument("--message", default="Add checkpoints")
    args = parser.parse_args()
    try:
        if args.command == "render":
            metadata, _ = render_kernel(args.kernel_dir, read_json(args.config) if args.config else None)
            result = metadata
        else:
            api = get_api()
            if args.command == "publish-code":
                result = publish_code(api, args.message, args.timeout)
            elif args.command == "push":
                result = push_kernel(api, args.kernel_dir, read_json(args.config) if args.config else None)
            elif args.command == "publish-ckpts":
                result = publish_ckpts(api, args.files, args.message)
            elif args.command == "quota":
                result = quota(api)
            elif args.command == "wait":
                result = wait_kernel(api, args.slug, args.interval, args.timeout, args.once)
            else:
                result = pull_kernel(api, args.slug, args.out_dir, args.file_pattern)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    except Exception as error:
        parser.exit(1, f"kx: {error}\n")


if __name__ == "__main__":
    main()
