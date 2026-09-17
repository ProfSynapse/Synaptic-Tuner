"""Tests for SynthChat.result_writer — streaming output, sidecar metadata and path generation."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from SynthChat.result_writer import (
    StreamingResultWriter,
    generate_output_path,
    metadata_path,
    save_results,
)


class FakeResult:
    def __init__(self, example=None, success=True, iterations=1):
        self.example = example if example is not None else {"conversations": [{"role": "user", "content": "hi"}]}
        self.scenario_key = "test"
        self.metadata = {}
        self.success = success
        self.iterations = iterations


# ---- generate_output_path ----

class TestGenerateOutputPath:
    def test_generate_mode_path(self, tmp_path):
        output_dir = str(tmp_path / "output")
        settings = {"output": {"default_dir": output_dir}}
        path = generate_output_path(settings)
        assert isinstance(path, Path)
        assert path.suffix == ".jsonl"
        assert "synthchat" in path.name
        assert str(path).startswith(output_dir)

    def test_improve_mode_path(self, tmp_path):
        input_file = tmp_path / "dataset.jsonl"
        input_file.touch()
        settings = {"output": {"default_dir": str(tmp_path)}}
        path = generate_output_path(settings, input_path=input_file)
        assert isinstance(path, Path)
        assert "dataset" in path.name
        assert path.suffix == ".jsonl"

    def test_strips_version_suffix(self, tmp_path):
        input_file = tmp_path / "dataset_v1.8.jsonl"
        input_file.touch()
        settings = {"output": {"default_dir": str(tmp_path)}}
        path = generate_output_path(settings, input_path=input_file)
        assert "_v1.8" not in path.name


def test_metadata_path_is_sidecar_next_to_dataset(tmp_path):
    assert metadata_path(tmp_path / "out.jsonl") == tmp_path / "out.jsonl.meta.json"


# ---- StreamingResultWriter ----

class TestStreamingResultWriter:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            settings = {"output": {"include_metadata": False}}

            writer = StreamingResultWriter(output, settings)
            writer.__enter__()
            try:
                writer.write(FakeResult())
            finally:
                writer.__exit__(None, None, None)

            assert output.exists()
            lines = output.read_text().strip().splitlines()
            assert len(lines) == 1
            assert json.loads(lines[0]) == FakeResult().example
            assert writer.metadata_file is None
            assert not metadata_path(output).exists()

    def test_data_file_is_homogeneous_and_metadata_goes_to_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            settings = {
                "output": {"include_metadata": True},
                "privacy_preprocess": {"enabled": True, "profile": "mask_only", "apply_to": {"docs": True}},
            }

            with StreamingResultWriter(output, settings) as writer:
                assert writer.metadata_file == metadata_path(output)
                started = json.loads(writer.metadata_file.read_text())
                assert started["streaming"] is True
                assert started["rows_written"] == 0
                assert "finished_at" not in started
                writer.write(FakeResult({"conversations": [{"role": "user", "content": "a"}]}))
                writer.write(FakeResult({"conversations": [{"role": "user", "content": "b"}]}))

            rows = [json.loads(line) for line in output.read_text().splitlines()]
            assert len(rows) == 2
            assert all("_meta" not in row and "conversations" in row for row in rows)

            finished = json.loads(metadata_path(output).read_text())
            assert finished["synthchat_version"] == "1.0.0"
            assert finished["rows_written"] == 2
            assert finished["streaming"] is True
            assert finished["started_at"] == started["started_at"]
            assert "finished_at" in finished
            assert finished["privacy_preprocess"] == {"profile": "mask_only", "apply_to": {"docs": True}}

    def test_open_is_non_truncating(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            settings = {"output": {"include_metadata": True}}
            with StreamingResultWriter(output, settings) as writer:
                writer.write(FakeResult({"conversations": [{"role": "user", "content": "first"}]}))
            with StreamingResultWriter(output, settings) as writer:
                writer.write(FakeResult({"conversations": [{"role": "user", "content": "second"}]}))

            rows = [json.loads(line)["conversations"][0]["content"] for line in output.read_text().splitlines()]
            assert rows == ["first", "second"]

    def test_thread_safety_count(self):
        """Writer count should match writes even from same thread."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            settings = {"output": {"include_metadata": False}}

            writer = StreamingResultWriter(output, settings)
            writer.__enter__()
            try:
                for _ in range(5):
                    writer.write(FakeResult({"conversations": []}))
                assert writer.count == 5
            finally:
                writer.__exit__(None, None, None)


# ---- save_results ----

def test_save_results_writes_rows_and_sidecar(tmp_path):
    output = tmp_path / "batch.jsonl"
    results = [FakeResult(success=True, iterations=1), FakeResult(success=False, iterations=3)]
    save_results(results, output, {"output": {"include_metadata": True}})

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert all("_meta" not in row for row in rows)
    metadata = json.loads(metadata_path(output).read_text())
    assert metadata["rows_written"] == 2
    assert metadata["streaming"] is False
    assert metadata["stats"] == {"total": 2, "passed": 1, "failed": 1, "avg_iterations": 2.0}


def test_save_results_without_metadata_writes_no_sidecar(tmp_path):
    output = tmp_path / "batch.jsonl"
    save_results([FakeResult()], output, {"output": {"include_metadata": False}})
    assert len(output.read_text().splitlines()) == 1
    assert not metadata_path(output).exists()
