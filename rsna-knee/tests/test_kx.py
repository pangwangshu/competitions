import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import kx


def kernel_dir(tmp_path, **overrides):
    definition = {"id": "wangshu/rsna-knee-test", "title": "RSNA Knee Test", "code_file": "kernel.py"}
    definition.update(overrides)
    (tmp_path / "kernel.json").write_text(json.dumps(definition))
    (tmp_path / "kernel.py").write_text('"""source docstring"""\nfrom __future__ import annotations\nprint("test")\n')
    return tmp_path


def test_render_attaches_inputs_and_embeds_config(tmp_path):
    metadata, source = kx.render_kernel(kernel_dir(tmp_path), {"k": 16}, "abc")
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"] == [kx.CODE_DATASET]
    assert metadata["competition_sources"] == [kx.COMPETITION]
    compiled = compile(source.read_text(), str(source), "exec")
    assert compiled is not None
    assert '"k": 16' in source.read_text()


@pytest.mark.parametrize("overrides", [{"is_private": False}, {"id": "someone/rsna-knee-test"},
                                        {"code_file": "../outside.py"}, {"title": "RSNA Different Title"}])
def test_render_rejects_unauthorized_uploads(tmp_path, overrides):
    with pytest.raises(ValueError):
        kx.render_kernel(kernel_dir(tmp_path, **overrides))


def test_status_and_output_refuse_a_different_latest_version(monkeypatch, tmp_path):
    monkeypatch.setattr(kx, "STATE", tmp_path)
    monkeypatch.setattr(kx, "kernel_info", lambda *_: SimpleNamespace(is_private=True, current_version_number=2))
    api = Mock()
    with pytest.raises(ValueError, match="latest kernel version"):
        kx.pull_kernel(api, "wangshu/rsna-knee-test/1", tmp_path / "output")
    api.kernels_output.assert_not_called()


def test_pull_can_fetch_only_manifests_without_downloading_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(kx, "expected_kernel", lambda *_: ("wangshu/rsna-knee-test", 1))
    api = Mock()
    api.kernels_status.return_value = SimpleNamespace(status=SimpleNamespace(name="COMPLETE"))
    api.kernels_output.return_value = ([tmp_path / "meta.csv"], None)
    api.kernels_logs.return_value = "complete log"
    monkeypatch.setattr(kx, "_pause", lambda _: None)
    result = kx.pull_kernel(api, "wangshu/rsna-knee-test/1", tmp_path, r"\.(csv|json)$")
    api.kernels_output.assert_called_once_with("wangshu/rsna-knee-test", str(tmp_path), force=True,
                                              quiet=True, page_size=100, page_token="", file_pattern=r"\.(csv|json)$")
    assert result["version"] == 1 and result["file_pattern"] == r"\.(csv|json)$"


def test_pull_refuses_running_version_to_avoid_stale_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(kx, "expected_kernel", lambda *_: ("wangshu/rsna-knee-test", 2))
    api = Mock()
    api.kernels_status.return_value = SimpleNamespace(status=SimpleNamespace(name="RUNNING"))
    with pytest.raises(ValueError, match="terminal run"):
        kx.pull_kernel(api, "wangshu/rsna-knee-test/2", tmp_path)
    api.kernels_output.assert_not_called()


def test_output_paging_retries_same_page_on_rate_limit(monkeypatch, tmp_path):
    from requests import Response
    from requests.exceptions import HTTPError

    monkeypatch.setattr(kx, "expected_kernel", lambda *_: ("wangshu/rsna-knee-test", 1))
    pauses = []
    monkeypatch.setattr(kx, "_pause", pauses.append)
    file = tmp_path / "meta.csv"
    file.write_text("series,n\nx,3\n")
    response = Response()
    response.status_code = 429
    response.headers["Retry-After"] = "30"
    api = Mock()
    api.kernels_status.return_value = SimpleNamespace(status=SimpleNamespace(name="COMPLETE"))
    api.kernels_output.side_effect = [([], "next-page"), HTTPError(response=response), ([file], None)]
    api.kernels_logs.return_value = "complete"
    result = kx.pull_kernel(api, "wangshu/rsna-knee-test/1", tmp_path, r"\.csv$")
    assert [call.kwargs["page_token"] for call in api.kernels_output.call_args_list] == ["", "next-page", "next-page"]
    assert 30 in pauses and str(file) in result["files"]
    assert not (tmp_path / "kx-output-partial.json").exists()


def test_public_dataset_is_never_updated(monkeypatch):
    monkeypatch.setattr(kx, "package_code", lambda: (Path("/tmp/stage"), {}))
    monkeypatch.setattr(kx, "owned_code_dataset", lambda _: SimpleNamespace(ref=kx.CODE_DATASET))
    monkeypatch.setattr(kx, "dataset_info", lambda _: SimpleNamespace(is_private=False))
    api = Mock()
    with pytest.raises(ValueError, match="public"):
        kx.publish_code(api, "test")
    api.dataset_create_new.assert_not_called()
    api.dataset_create_version.assert_not_called()


def test_push_accepts_a_versioned_response_ref(monkeypatch, tmp_path):
    monkeypatch.setattr(kx, "STATE", tmp_path)
    kx.write_json(tmp_path / "code-publication.json", {
        "dataset": kx.CODE_DATASET, "is_private": True, "version": 1,
        "manifest": {"archive_sha256": "abc", "source_sha256": {}},
    })
    monkeypatch.setattr(kx, "package_code", lambda: (tmp_path, {"archive_sha256": "abc", "source_sha256": {}}))
    monkeypatch.setattr(kx, "dataset_info", lambda _: SimpleNamespace(is_private=True, current_version_number=1))
    source = tmp_path / "kernel.py"
    source.write_text('print("test")')
    kx.write_json(tmp_path / "kernel.json", {"session_timeout_seconds": 600})
    monkeypatch.setattr(kx, "render_kernel", lambda *_: ({"id": "wangshu/rsna-knee-test", "enable_gpu": False}, source))
    info_calls = []
    def info(_, slug):
        info_calls.append(slug)
        return SimpleNamespace(is_private=True)
    monkeypatch.setattr(kx, "kernel_info", info)
    api = Mock()
    api.kernels_push.return_value = SimpleNamespace(ref="wangshu/rsna-knee-test/1", version_number=1,
                                                   error="", url="https://www.kaggle.com/code/wangshu/rsna-knee-test")
    result = kx.push_kernel(api, tmp_path)
    assert info_calls == ["wangshu/rsna-knee-test"]
    assert result["kernel"] == "wangshu/rsna-knee-test" and result["version"] == 1
    assert api.kernels_push.call_args.kwargs["timeout"] == "600" and result["session_timeout_seconds"] == 600


def test_render_loads_kaggle_extracted_archive_and_rejects_changed_source(tmp_path, monkeypatch):
    import hashlib
    import os
    import sys

    dataset = tmp_path / "dataset"
    package = dataset / "rsna_knee" / "rsna_knee"
    package.mkdir(parents=True)
    content = b'# mock package\n'
    (package / "__init__.py").write_bytes(content)
    hashes = {"rsna_knee/__init__.py": hashlib.sha256(content).hexdigest()}
    kx.write_json(dataset / "code-manifest.json", {"archive_sha256": "abc", "source_sha256": hashes})
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    kernel_dir(kernel)
    (kernel / "kernel.py").write_text('from __future__ import annotations\nassert 1 == 1\n')
    _, script = kx.render_kernel(kernel, {"k": 16}, "abc", hashes)
    source = script.read_text().replace('/kaggle/input/rsna-knee-code', str(dataset))
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ['RK_CONFIG_JSON', 'RK_CODE_ROOT', 'RK_CODE_PACKAGE_PATH']:
        monkeypatch.setenv(name, '')
    exec(compile(source, str(script), 'exec'), {"__name__": "__main__"})
    assert json.loads(os.environ['RK_CONFIG_JSON']) == {"k": 16}
    assert sys.path[0] == str(dataset / "rsna_knee")
    (package / "__init__.py").write_text('# changed source\n')
    with pytest.raises(RuntimeError, match="source file has changed"):
        exec(compile(source, str(script), 'exec'), {"__name__": "__main__"})
