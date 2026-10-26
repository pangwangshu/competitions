"""Regression checks for pinned inputs and regenerated experiment configs."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import fetch_s1_labels, s5


def test_label_fetch_uses_pinned_version_and_preserves_manifest(tmp_path):
    content = b"study,label\ns1,0.5\n"
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"dataset": "author/labels", "version": 7,
        "filename": "labels.csv", "sha256": hashlib.sha256(content).hexdigest()}]))
    original = manifest.read_bytes()
    calls = []

    class API:
        def dataset_download_file(self, ref, filename, **kwargs):
            calls.append(ref)
            (Path(kwargs["path"]) / filename).write_bytes(content)

    fetch_s1_labels.fetch(API(), manifest, tmp_path / "ext")
    fetch_s1_labels.fetch(API(), manifest, tmp_path / "ext")
    assert calls == ["author/labels/7"]
    assert manifest.read_bytes() == original


def test_changed_label_download_is_rejected(tmp_path):
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"dataset": "author/labels", "version": 7,
        "filename": "labels.csv", "sha256": "0" * 64}]))

    class API:
        def dataset_download_file(self, ref, filename, **kwargs):
            (Path(kwargs["path"]) / filename).write_text("changed")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch_s1_labels.fetch(API(), manifest, tmp_path / "ext")


@pytest.mark.parametrize("legacy_teacher", [False, True])
def test_s5_configs_accept_builder_hash_and_optional_legacy_teacher(tmp_path, monkeypatch, legacy_teacher):
    monkeypatch.setattr(s5, "EXP", tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(s5, "CKPT_MANIFEST", manifest)
    monkeypatch.setattr(s5, "base_config", lambda: {"labels_sha256": "labels", "meta_sha256": "meta"})
    monkeypatch.setattr(s5, "eval_run", lambda name, ckpt: {"name": name})
    names = ["S5-T4S6"] + (["S5-T4"] if legacy_teacher else [])
    for name in names:
        folder = tmp_path / name
        folder.mkdir()
        (folder / ("labels_t4.json" if name == "S5-T4" else "labels_t4s.json")).write_text(
            json.dumps({"labels_sha256": "builder-hash"}))
    s5.write_configs()
    config = json.loads((tmp_path / "S5/session-T4S6-config.json").read_text())
    assert config["extra_labels"]["labels_t4s"]["sha256"] == "builder-hash"
    if legacy_teacher:
        config = json.loads((tmp_path / "S5/session-T4-config.json").read_text())
        assert config["extra_labels"]["labels_t4"]["sha256"] == "builder-hash"

        # Archived development records used the older field name.
        (tmp_path / "S5-T4/labels_t4.json").write_text(json.dumps({"labels_t4_sha256": "old-hash"}))
        s5.write_configs()
        config = json.loads((tmp_path / "S5/session-T4-config.json").read_text())
        assert config["extra_labels"]["labels_t4"]["sha256"] == "old-hash"
