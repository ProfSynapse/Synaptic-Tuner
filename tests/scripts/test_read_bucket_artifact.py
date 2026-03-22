from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from shared.utilities import bucket_artifacts


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".skills" / "fine-tuning" / "scripts" / "read_bucket_artifact.py"
SPEC = spec_from_file_location("read_bucket_artifact", SCRIPT_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_tail_lines_returns_last_n_lines() -> None:
    result = MODULE._tail_lines(["a\n", "b\n", "c\n"], 2)
    assert result == ["b\n", "c\n"]


def test_latest_jsonl_record_reads_last_nonempty_record() -> None:
    lines = ['{"step": 1}\n', "\n", '{"step": 2, "loss": 0.5}\n']
    assert MODULE._latest_jsonl_record(lines) == {"step": 2, "loss": 0.5}


def test_read_artifact_tail_on_local_file(tmp_path: Path) -> None:
    path = tmp_path / "training_latest.jsonl"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    output = MODULE.read_artifact(str(path), tail=2)

    assert output == "two\nthree\n"


def test_read_artifact_jsonl_latest_pretty(tmp_path: Path) -> None:
    path = tmp_path / "training_latest.jsonl"
    path.write_text('{"step": 1}\n{"step": 2, "loss": 0.25}\n', encoding="utf-8")

    output = MODULE.read_artifact(str(path), jsonl_latest=True, pretty=True)

    assert '"step": 2' in output
    assert '"loss": 0.25' in output


def test_read_artifact_raises_on_empty_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No JSONL records found"):
        MODULE.read_artifact(str(path), jsonl_latest=True)


def test_read_artifact_uses_hf_filesystem_for_bucket_uris(monkeypatch) -> None:
    opened = {}

    class FakeFS:
        def __init__(self, token=None):
            opened["token"] = token

        def open(self, path, mode="r", encoding=None):
            opened["path"] = path
            opened["mode"] = mode
            opened["encoding"] = encoding
            return open(__file__, "r", encoding="utf-8")

    monkeypatch.setattr(bucket_artifacts, "HfFileSystem", FakeFS)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    MODULE.read_artifact("hf://buckets/test/example.json")

    assert opened == {
        "token": "hf_test_token",
        "path": "hf://buckets/test/example.json",
        "mode": "r",
        "encoding": "utf-8",
    }
