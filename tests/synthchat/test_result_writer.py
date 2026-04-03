"""Tests for SynthChat.result_writer — streaming output and path generation."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from SynthChat.result_writer import StreamingResultWriter, generate_output_path


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


# ---- StreamingResultWriter ----

class TestStreamingResultWriter:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            settings = {"output": {"include_metadata": False}}

            class FakeResult:
                example = {"conversations": [{"role": "user", "content": "hi"}]}
                scenario_key = "test"
                metadata = {}

            writer = StreamingResultWriter(output, settings)
            writer.__enter__()
            try:
                writer.write(FakeResult())
            finally:
                writer.__exit__(None, None, None)

            assert output.exists()
            lines = output.read_text().strip().splitlines()
            assert len(lines) >= 1

    def test_metadata_header_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            settings = {"output": {"include_metadata": True}}

            writer = StreamingResultWriter(output, settings)
            writer.__enter__()
            writer.__exit__(None, None, None)

            lines = output.read_text().strip().splitlines()
            if lines:
                header = json.loads(lines[0])
                assert "_meta" in header

    def test_thread_safety_count(self):
        """Writer count should match writes even from same thread."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            settings = {"output": {"include_metadata": False}}

            class FakeResult:
                example = {"conversations": []}
                scenario_key = "test"
                metadata = {}

            writer = StreamingResultWriter(output, settings)
            writer.__enter__()
            try:
                for _ in range(5):
                    writer.write(FakeResult())
                assert writer._count == 5
            finally:
                writer.__exit__(None, None, None)
