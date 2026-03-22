import importlib.util
import sys
from pathlib import Path

import pytest


def _load_skill_module():
    path = Path(".agents/skills/dataset-publishing/scripts/publish_dataset_to_hf.py").resolve()
    spec = importlib.util.spec_from_file_location("dataset_publishing_skill_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeHfApi:
    def __init__(self):
        self.created = []
        self.uploaded = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def upload_file(self, **kwargs):
        self.uploaded.append(kwargs)


def test_infer_metadata_path_finds_sidecar(tmp_path):
    module = _load_skill_module()
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text('{"a": 1}\n', encoding="utf-8")
    metadata = tmp_path / "sample.metadata.json"
    metadata.write_text("{}", encoding="utf-8")

    assert module.infer_metadata_path(dataset) == metadata


def test_infer_metadata_path_handles_dotted_dataset_names(tmp_path):
    module = _load_skill_module()
    dataset = tmp_path / "sample.v1.2.3.jsonl"
    dataset.write_text('{"a": 1}\n', encoding="utf-8")
    metadata = tmp_path / "sample.v1.2.3.metadata.json"
    metadata.write_text("{}", encoding="utf-8")

    assert module.infer_metadata_path(dataset) == metadata


def test_build_upload_targets_includes_metadata_by_default(tmp_path):
    module = _load_skill_module()
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text('{"a": 1}\n', encoding="utf-8")
    metadata = tmp_path / "sample.metadata.json"
    metadata.write_text("{}", encoding="utf-8")

    targets = module.build_upload_targets(dataset)

    assert [target.path_in_repo for target in targets] == ["sample.jsonl", "sample.metadata.json"]


def test_build_upload_targets_can_skip_metadata(tmp_path):
    module = _load_skill_module()
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text('{"a": 1}\n', encoding="utf-8")
    metadata = tmp_path / "sample.metadata.json"
    metadata.write_text("{}", encoding="utf-8")

    targets = module.build_upload_targets(dataset, include_metadata=False)

    assert [target.path_in_repo for target in targets] == ["sample.jsonl"]


def test_publish_dataset_dry_run_does_not_touch_api(tmp_path):
    module = _load_skill_module()
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text('{"a": 1}\n', encoding="utf-8")

    result = module.publish_dataset(
        dataset,
        "professorsynapse/test-dataset",
        include_metadata=False,
        dry_run=True,
    )

    assert result.repo_id == "professorsynapse/test-dataset"
    assert result.uploaded_files == ["sample.jsonl"]


def test_publish_dataset_uploads_dataset_and_metadata(tmp_path):
    module = _load_skill_module()
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text('{"a": 1}\n', encoding="utf-8")
    metadata = tmp_path / "sample.metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    api = FakeHfApi()

    result = module.publish_dataset(
        dataset,
        "professorsynapse/test-dataset",
        api=api,
        token="hf_test",
        commit_message="Upload sample dataset",
    )

    assert result.uploaded_files == ["sample.jsonl", "sample.metadata.json"]
    assert api.created[0]["repo_type"] == "dataset"
    assert [item["path_in_repo"] for item in api.uploaded] == ["sample.jsonl", "sample.metadata.json"]
    assert all(item["repo_type"] == "dataset" for item in api.uploaded)
    assert all(item["commit_message"] == "Upload sample dataset" for item in api.uploaded)


def test_publish_dataset_requires_repo_namespace(tmp_path):
    module = _load_skill_module()
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text('{"a": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="repo_id"):
        module.publish_dataset(dataset, "badrepo", include_metadata=False, dry_run=True)
