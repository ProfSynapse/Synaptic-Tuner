"""Tests for shared/experiment_tracking/registry.py — RunRegistry JSONL store."""
from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from shared.experiment_tracking.registry import RunRegistry
import shared.experiment_tracking.registry as registry_module
from shared.experiment_tracking.schema import RunFilter, RunRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**overrides) -> RunRecord:
    """Build a RunRecord with sensible defaults."""
    defaults = dict(
        run_id="run-001",
        run_type="sft",
        name="SFT run",
        timestamp="2026-03-14T18:00:00+00:00",
        status="completed",
        output_dir="/runs/sft_20260314",
    )
    defaults.update(overrides)
    return RunRecord(**defaults)


def _hold_registry_lock(path: str, ready, release) -> None:
    from shared.experiment_tracking.registry import _PathLock

    with _PathLock(Path(path)):
        ready.set()
        release.wait(timeout=10)


def _acquire_registry_lock_then_exit(path: str, ready) -> None:
    from shared.experiment_tracking.registry import _PathLock

    with _PathLock(Path(path)):
        ready.set()
        os._exit(0)


_LIGHTWEIGHT_REGISTRY_CHILD = r"""
import importlib.util
import json
import pathlib
import sys
import types

repo_root = pathlib.Path(sys.argv[1])
registry_path = sys.argv[2]
index = int(sys.argv[3])

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

try:
    shared = types.ModuleType("shared")
    shared.__path__ = [str(repo_root / "shared")]
    tracking = types.ModuleType("shared.experiment_tracking")
    tracking.__path__ = [str(repo_root / "shared" / "experiment_tracking")]
    shared.experiment_tracking = tracking
    sys.modules["shared"] = shared
    sys.modules["shared.experiment_tracking"] = tracking
    schema = load_module(
        "shared.experiment_tracking.schema",
        repo_root / "shared" / "experiment_tracking" / "schema.py",
    )
    registry = load_module(
        "shared.experiment_tracking.registry",
        repo_root / "shared" / "experiment_tracking" / "registry.py",
    )
    if sys.stdin.buffer.read(1) != b"1":
        raise RuntimeError("parent did not release start gate")
    record = schema.RunRecord(
        run_id=f"process-{index}",
        run_type="sft",
        name="SFT run",
        timestamp="2026-03-14T18:00:00+00:00",
        status="completed",
        output_dir=f"/runs/process-{index}",
    )
    run_id = registry.RunRegistry(registry_path).register_run(record)
    result = {
        "rc": 0,
        "run_id": run_id,
        "heavy_modules_absent": not any(
            name == "torch" or name.startswith("torch.")
            or name == "transformers" or name.startswith("transformers.")
            for name in sys.modules
        ),
    }
except BaseException as exc:
    result = {"rc": 1, "run_id": f"process-{index}", "error": repr(exc)}
print(json.dumps(result), flush=True)
"""


def _run_lightweight_first_writers(
    repo_root: Path, path: Path, process_count: int
) -> list[dict[str, object]]:
    processes: list[subprocess.Popen[bytes]] = []
    outputs: list[tuple[bytes, bytes]] = []
    deadline = time.monotonic() + 45
    try:
        for index in range(process_count):
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        _LIGHTWEIGHT_REGISTRY_CHILD,
                        str(repo_root),
                        str(path),
                        str(index),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                )
            )
        for process in processes:
            assert process.stdin is not None
            process.stdin.write(b"1")
            process.stdin.flush()
        for process in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, 45)
            outputs.append(process.communicate(timeout=remaining))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        for process in processes:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()

    assert all(process.returncode == 0 for process in processes)
    results = []
    for stdout, stderr in outputs:
        assert not stderr, stderr.decode("utf-8", errors="replace")
        results.append(json.loads(stdout.decode("utf-8")))
    return results


# ===========================================================================
# RunRegistry — Core Operations
# ===========================================================================

class TestRunRegistryCore:
    """Register, read back, and query runs."""

    def test_register_and_read_back(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        record = _make_record()
        run_id = registry.register_run(record)

        assert run_id == "run-001"
        runs = registry.find_runs()
        assert len(runs) == 1
        assert runs[0].run_id == "run-001"
        assert runs[0].run_type == "sft"

    def test_register_multiple_runs(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(run_id="run-001", run_type="sft", output_dir="/runs/sft"))
        registry.register_run(_make_record(run_id="run-002", run_type="kto", output_dir="/runs/kto"))
        registry.register_run(_make_record(run_id="run-003", run_type="ml", output_dir="/runs/ml"))

        runs = registry.find_runs()
        assert len(runs) == 3
        assert [r.run_id for r in runs] == ["run-001", "run-002", "run-003"]

    def test_get_run_found(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(run_id="run-001", output_dir="/runs/001"))
        registry.register_run(_make_record(run_id="run-002", output_dir="/runs/002"))

        result = registry.get_run("run-002")
        assert result is not None
        assert result.run_id == "run-002"

    def test_get_run_not_found(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(run_id="run-001"))

        result = registry.get_run("nonexistent")
        assert result is None

    def test_creates_parent_directories(self, tmp_path: Path):
        deep_path = tmp_path / "deep" / "nested" / "registry.jsonl"
        registry = RunRegistry(deep_path)
        registry.register_run(_make_record())

        assert deep_path.exists()

    def test_registry_file_is_valid_jsonl(self, tmp_path: Path):
        path = tmp_path / "registry.jsonl"
        registry = RunRegistry(path)
        registry.register_run(_make_record(run_id="run-001", output_dir="/runs/001"))
        registry.register_run(_make_record(run_id="run-002", output_dir="/runs/002"))

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "run_id" in data


# ===========================================================================
# RunRegistry — Filtering
# ===========================================================================

class TestRunRegistryFiltering:
    """Test find_runs with various filters."""

    @pytest.fixture
    def populated_registry(self, tmp_path: Path) -> RunRegistry:
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(
            run_id="sft-001", run_type="sft", status="completed",
            output_dir="/runs/sft-001",
            timestamp="2026-03-14T10:00:00+00:00",
            model_name="unsloth/Qwen2.5-7B",
            tags={"method": "sft", "provider": "local"},
        ))
        registry.register_run(_make_record(
            run_id="kto-001", run_type="kto", status="completed",
            output_dir="/runs/kto-001",
            timestamp="2026-03-14T15:00:00+00:00",
            model_name="unsloth/Qwen2.5-7B-SFT",
            tags={"method": "kto", "provider": "local"},
        ))
        registry.register_run(_make_record(
            run_id="ml-001", run_type="ml", status="completed",
            output_dir="/runs/ml-001",
            timestamp="2026-03-14T12:00:00+00:00",
            model_name="lightgbm",
            tags={"method": "ml", "algorithm": "lightgbm"},
        ))
        registry.register_run(_make_record(
            run_id="sft-002", run_type="sft", status="failed",
            output_dir="/runs/sft-002",
            timestamp="2026-03-15T08:00:00+00:00",
            model_name="unsloth/Qwen2.5-7B",
            tags={"method": "sft", "provider": "cloud"},
        ))
        return registry

    def test_filter_by_run_type(self, populated_registry: RunRegistry):
        runs = populated_registry.find_runs(RunFilter(run_type="sft"))
        assert len(runs) == 2
        assert all(r.run_type == "sft" for r in runs)

    def test_filter_by_status(self, populated_registry: RunRegistry):
        runs = populated_registry.find_runs(RunFilter(status="failed"))
        assert len(runs) == 1
        assert runs[0].run_id == "sft-002"

    def test_filter_by_model_name(self, populated_registry: RunRegistry):
        runs = populated_registry.find_runs(RunFilter(model_name="qwen"))
        assert len(runs) == 3  # All Qwen models

    def test_filter_by_timestamp_range(self, populated_registry: RunRegistry):
        runs = populated_registry.find_runs(RunFilter(
            since="2026-03-14T11:00:00+00:00",
            until="2026-03-14T16:00:00+00:00",
        ))
        assert len(runs) == 2
        assert {r.run_id for r in runs} == {"kto-001", "ml-001"}

    def test_filter_by_tags(self, populated_registry: RunRegistry):
        runs = populated_registry.find_runs(RunFilter(tags={"provider": "cloud"}))
        assert len(runs) == 1
        assert runs[0].run_id == "sft-002"

    def test_filter_combined(self, populated_registry: RunRegistry):
        runs = populated_registry.find_runs(RunFilter(
            run_type="sft", status="completed",
        ))
        assert len(runs) == 1
        assert runs[0].run_id == "sft-001"

    def test_no_filter_returns_all(self, populated_registry: RunRegistry):
        runs = populated_registry.find_runs()
        assert len(runs) == 4

    def test_filter_no_matches(self, populated_registry: RunRegistry):
        runs = populated_registry.find_runs(RunFilter(run_type="grpo"))
        assert runs == []


# ===========================================================================
# RunRegistry — Linkage
# ===========================================================================

class TestRunRegistryLinkage:
    """Test link_runs and get_linked_runs."""

    def test_link_and_query_child(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(run_id="train-001", run_type="sft", output_dir="/runs/train-001"))
        registry.register_run(_make_record(run_id="eval-001", run_type="evaluation", output_dir="/runs/eval-001"))
        registry.link_runs(child_run_id="eval-001", parent_run_id="train-001")

        # Query from parent side → find child
        linked = registry.get_linked_runs("train-001")
        assert len(linked) == 1
        assert linked[0].run_id == "eval-001"

    def test_link_and_query_parent(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(run_id="train-001", run_type="sft", output_dir="/runs/train-001"))
        registry.register_run(_make_record(run_id="eval-001", run_type="evaluation", output_dir="/runs/eval-001"))
        registry.link_runs(child_run_id="eval-001", parent_run_id="train-001")

        # Query from child side → find parent
        linked = registry.get_linked_runs("eval-001")
        assert len(linked) == 1
        assert linked[0].run_id == "train-001"

    def test_multiple_links(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(run_id="train-001", run_type="sft", output_dir="/runs/train-001"))
        registry.register_run(_make_record(run_id="eval-001", run_type="evaluation", output_dir="/runs/eval-001"))
        registry.register_run(_make_record(run_id="eval-002", run_type="evaluation", output_dir="/runs/eval-002"))
        registry.link_runs(child_run_id="eval-001", parent_run_id="train-001")
        registry.link_runs(child_run_id="eval-002", parent_run_id="train-001")

        linked = registry.get_linked_runs("train-001")
        assert len(linked) == 2
        assert {r.run_id for r in linked} == {"eval-001", "eval-002"}

    def test_link_with_relationship_filter(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(run_id="train-001", run_type="sft", output_dir="/runs/train-001"))
        registry.register_run(_make_record(run_id="eval-001", run_type="evaluation", output_dir="/runs/eval-001"))
        registry.register_run(_make_record(run_id="derived-001", run_type="kto", output_dir="/runs/derived-001"))
        registry.link_runs("eval-001", "train-001", relationship="parent")
        registry.link_runs("derived-001", "train-001", relationship="derived_from")

        # Filter by relationship
        parent_linked = registry.get_linked_runs("train-001", relationship="parent")
        assert len(parent_linked) == 1
        assert parent_linked[0].run_id == "eval-001"

        derived_linked = registry.get_linked_runs("train-001", relationship="derived_from")
        assert len(derived_linked) == 1
        assert derived_linked[0].run_id == "derived-001"

    def test_no_links_returns_empty(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(run_id="train-001"))

        linked = registry.get_linked_runs("train-001")
        assert linked == []

    def test_links_stored_in_separate_file(self, tmp_path: Path):
        """Links are stored in links.jsonl alongside the registry."""
        path = tmp_path / "registry.jsonl"
        registry = RunRegistry(path)
        registry.register_run(_make_record(run_id="train-001", output_dir="/runs/train-001"))
        registry.register_run(_make_record(run_id="eval-001", output_dir="/runs/eval-001"))
        registry.link_runs("eval-001", "train-001")

        # Records should still load correctly
        runs = registry.find_runs()
        assert len(runs) == 2

        # Registry file has only run records (2 lines)
        reg_lines = path.read_text().strip().split("\n")
        assert len(reg_lines) == 2

        # Links file exists separately
        links_path = tmp_path / "links.jsonl"
        assert links_path.exists()
        link_lines = links_path.read_text().strip().split("\n")
        assert len(link_lines) == 1


# ===========================================================================
# RunRegistry — Edge Cases and Robustness
# ===========================================================================

class TestRunRegistryEdgeCases:
    """Empty registry, malformed lines, concurrent-ish writes."""

    def test_empty_registry_returns_empty_list(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        assert registry.find_runs() == []

    def test_nonexistent_file_returns_empty_list(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "nonexistent" / "registry.jsonl")
        assert registry.find_runs() == []

    def test_malformed_lines_skipped(self, tmp_path: Path):
        """Malformed JSON lines are skipped; valid lines still load."""
        path = tmp_path / "registry.jsonl"
        record = _make_record(run_id="good-001")
        good_line = record.to_json_line()

        # Write a mix of valid, malformed, and empty lines
        path.write_text(
            f"{good_line}\n"
            "this is not valid json\n"
            "\n"
            '{"run_id": "good-002", "run_type": "kto", "name": "KTO", '
            '"timestamp": "2026-01-01T00:00:00Z", "status": "completed", '
            '"output_dir": "/out"}\n'
        )

        registry = RunRegistry(path)
        runs = registry.find_runs()
        assert len(runs) == 2
        assert runs[0].run_id == "good-001"
        assert runs[1].run_id == "good-002"

    def test_blank_lines_skipped(self, tmp_path: Path):
        path = tmp_path / "registry.jsonl"
        record = _make_record(run_id="run-001")
        path.write_text(f"\n\n{record.to_json_line()}\n\n")

        registry = RunRegistry(path)
        runs = registry.find_runs()
        assert len(runs) == 1

    def test_multiple_appends_preserve_order(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        for i in range(10):
            registry.register_run(_make_record(run_id=f"run-{i:03d}", output_dir=f"/runs/{i:03d}"))

        runs = registry.find_runs()
        assert [r.run_id for r in runs] == [f"run-{i:03d}" for i in range(10)]

    def test_append_preserves_historical_bytes_without_trailing_newline(self, tmp_path: Path):
        path = tmp_path / "registry.jsonl"
        original = _make_record(run_id="legacy", output_dir="/runs/legacy").to_json_line().encode()
        path.write_bytes(original)

        RunRegistry(path).register_run(
            _make_record(run_id="new", output_dir="/runs/new")
        )

        updated = path.read_bytes()
        assert updated.startswith(original + b"\n")
        assert [record.run_id for record in RunRegistry(path).find_runs()] == ["legacy", "new"]

    def test_get_linked_runs_no_registry_file(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "nonexistent.jsonl")
        assert registry.get_linked_runs("any-id") == []

    def test_concurrent_writes_no_corruption(self, tmp_path: Path):
        """Multiple threads writing should not corrupt the JSONL file."""
        import threading

        path = tmp_path / "registry.jsonl"
        num_threads = 8
        records_per_thread = 5
        errors: list[Exception] = []

        def write_records(thread_id: int) -> None:
            try:
                # Each thread gets its own registry instance (realistic)
                registry = RunRegistry(path)
                for i in range(records_per_thread):
                    registry.register_run(_make_record(
                        run_id=f"t{thread_id}-r{i}",
                        output_dir=f"/runs/t{thread_id}/r{i}",
                    ))
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=write_records, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

        # Every line in the file must be valid JSON (no corruption)
        lines = [ln for ln in path.read_text().strip().split("\n") if ln.strip()]
        for i, line in enumerate(lines):
            data = json.loads(line)  # Must not raise
            assert "run_id" in data, f"Line {i} missing run_id"

        expected = num_threads * records_per_thread
        assert len(lines) == expected
        assert {json.loads(line)["run_id"] for line in lines} == {
            f"t{thread_id}-r{record_id}"
            for thread_id in range(num_threads)
            for record_id in range(records_per_thread)
        }

    def test_concurrent_duplicate_registration_is_idempotent(self, tmp_path: Path):
        """The complete idempotency check and write form one transaction."""
        import threading

        path = tmp_path / "registry.jsonl"
        barrier = threading.Barrier(8)
        results: list[str] = []
        errors: list[Exception] = []

        def register_duplicate(thread_id: int) -> None:
            try:
                barrier.wait()
                results.append(
                    RunRegistry(path).register_run(
                        _make_record(
                            run_id=f"candidate-{thread_id}",
                            output_dir="/runs/shared",
                        )
                    )
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register_duplicate, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert len(set(results)) == 1
        assert len(path.read_text(encoding="utf-8").splitlines()) == 1
        assert path.with_name("registry.jsonl.lock").read_bytes() == b"\0"

    def test_active_owner_is_never_stolen_when_lock_mtime_is_old(
        self, tmp_path: Path, monkeypatch
    ):
        path = tmp_path / "registry.jsonl"
        RunRegistry(path).register_run(
            _make_record(run_id="original", output_dir="/runs/original")
        )
        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        release = context.Event()
        process = context.Process(
            target=_hold_registry_lock,
            args=(str(path), ready, release),
        )
        process.start()
        try:
            assert ready.wait(timeout=5)
            lock_path = path.with_name("registry.jsonl.lock")
            os.utime(lock_path, (1, 1))
            monkeypatch.setattr(registry_module, "_LOCK_TIMEOUT_SECONDS", 0.2)

            with pytest.raises(TimeoutError, match="Timed out acquiring registry lock"):
                RunRegistry(path).register_run(
                    _make_record(run_id="blocked", output_dir="/runs/blocked")
                )

            assert [record.run_id for record in RunRegistry(path).find_runs()] == [
                "original"
            ]
        finally:
            release.set()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

        RunRegistry(path).register_run(
            _make_record(run_id="after-release", output_dir="/runs/after-release")
        )
        assert [record.run_id for record in RunRegistry(path).find_runs()] == [
            "original",
            "after-release",
        ]

    def test_dead_owner_lock_is_recovered_by_the_kernel(self, tmp_path: Path):
        path = tmp_path / "registry.jsonl"
        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        process = context.Process(
            target=_acquire_registry_lock_then_exit,
            args=(str(path), ready),
        )
        process.start()
        assert ready.wait(timeout=5)
        process.join(timeout=5)
        assert not process.is_alive()

        RunRegistry(path).register_run(
            _make_record(run_id="recovered", output_dir="/runs/recovered")
        )

        assert [record.run_id for record in RunRegistry(path).find_runs()] == [
            "recovered"
        ]
        assert path.with_name("registry.jsonl.lock").read_bytes() == b"\0"

    def test_spawned_first_writers_publish_one_complete_sentinel_without_loss(
        self, tmp_path: Path
    ):
        process_count = 32
        repo_root = Path(__file__).resolve().parents[3]
        for round_index in range(2):
            path = tmp_path / f"round-{round_index}" / "registry.jsonl"
            path.parent.mkdir()
            received = _run_lightweight_first_writers(
                repo_root, path, process_count
            )

            assert all(result["rc"] == 0 for result in received), received
            assert all(result["heavy_modules_absent"] for result in received)
            assert {result["run_id"] for result in received} == {
                f"process-{index}" for index in range(process_count)
            }
            records = RunRegistry(path).find_runs()
            assert len(records) == process_count
            assert {record.run_id for record in records} == {
                f"process-{index}" for index in range(process_count)
            }
            assert len(path.read_text(encoding="utf-8").splitlines()) == process_count
            assert path.with_name("registry.jsonl.lock").read_bytes() == b"\0"
            assert list(path.parent.glob("*.init")) == []
            assert list(path.parent.glob("*.tmp")) == []

    def test_registry_replace_retries_transient_windows_failure(self, tmp_path: Path, monkeypatch):
        path = tmp_path / "registry.jsonl"
        RunRegistry(path).register_run(
            _make_record(run_id="original", output_dir="/runs/original")
        )
        real_replace = registry_module.os.replace
        calls = 0

        def transient_then_success(source, target):
            nonlocal calls
            calls += 1
            if calls < 3:
                error = PermissionError(13, "injected Windows sharing violation")
                error.winerror = 5
                raise error
            return real_replace(source, target)

        monkeypatch.setattr(registry_module.os, "replace", transient_then_success)
        RunRegistry(path).register_run(
            _make_record(run_id="after-retry", output_dir="/runs/after-retry")
        )

        assert calls == 3
        assert [record.run_id for record in RunRegistry(path).find_runs()] == [
            "original",
            "after-retry",
        ]
        assert list(tmp_path.glob("*.tmp")) == []

    def test_registry_replace_timeout_preserves_old_bytes_and_cleans_temp(
        self, tmp_path: Path, monkeypatch
    ):
        path = tmp_path / "registry.jsonl"
        RunRegistry(path).register_run(
            _make_record(run_id="original", output_dir="/runs/original")
        )
        original = path.read_bytes()
        calls = 0

        def persistent_transient(_source, _target):
            nonlocal calls
            calls += 1
            error = PermissionError(13, "injected persistent Windows sharing violation")
            error.winerror = 32
            raise error

        monkeypatch.setattr(registry_module.os, "replace", persistent_transient)
        monkeypatch.setattr(registry_module, "_REPLACE_TIMEOUT_SECONDS", 0.03)
        monkeypatch.setattr(registry_module, "_REPLACE_POLL_SECONDS", 0.005)

        with pytest.raises(TimeoutError, match="Timed out replacing registry"):
            RunRegistry(path).register_run(
                _make_record(run_id="must-not-land", output_dir="/runs/must-not-land")
            )

        assert calls >= 2
        assert path.read_bytes() == original
        assert [record.run_id for record in RunRegistry(path).find_runs()] == ["original"]
        assert list(tmp_path.glob("*.tmp")) == []

    def test_registry_replace_does_not_retry_non_transient_error(
        self, tmp_path: Path, monkeypatch
    ):
        path = tmp_path / "registry.jsonl"
        RunRegistry(path).register_run(
            _make_record(run_id="original", output_dir="/runs/original")
        )
        original = path.read_bytes()
        calls = 0

        def semantic_failure(_source, _target):
            nonlocal calls
            calls += 1
            raise FileNotFoundError(2, "injected semantic path failure")

        monkeypatch.setattr(registry_module.os, "replace", semantic_failure)
        with pytest.raises(FileNotFoundError, match="semantic path failure"):
            RunRegistry(path).register_run(
                _make_record(run_id="must-not-land", output_dir="/runs/must-not-land")
            )

        assert calls == 1
        assert path.read_bytes() == original
        assert list(tmp_path.glob("*.tmp")) == []

    def test_unicode_content_roundtrips(self, tmp_path: Path):
        """Unicode model names and tags survive JSONL serialization."""
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(
            run_id="unicode-001",
            model_name="日本語モデル/Qwen2.5-7B",
            tags={"描述": "微调模型", "emoji": "rocket-launch"},
        ))

        runs = registry.find_runs()
        assert len(runs) == 1
        assert runs[0].model_name == "日本語モデル/Qwen2.5-7B"
        assert runs[0].tags["描述"] == "微调模型"

    def test_idempotent_register_skips_duplicate_output_dir(self, tmp_path: Path):
        """Registering a run with the same output_dir returns existing run_id."""
        registry = RunRegistry(tmp_path / "registry.jsonl")
        first_id = registry.register_run(_make_record(
            run_id="run-001", output_dir="/runs/same_dir",
        ))
        second_id = registry.register_run(_make_record(
            run_id="run-002", output_dir="/runs/same_dir",
        ))

        assert first_id == "run-001"
        assert second_id == "run-001"  # Returns existing, not new
        assert len(registry.find_runs()) == 1
