"""Tests for shared/utilities/run_log.py -- resumable per-item run log."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from shared.utilities.run_log import RunLog, RunLogError


CONFIG_A = {"arm": "baseline", "seed": 1}
CONFIG_B = {"arm": "baseline", "seed": 2}


class TestResume:
    def test_resume_skips_done(self, tmp_path: Path):
        log_path = tmp_path / "rows.jsonl"

        log = RunLog(log_path, CONFIG_A)
        log.record("row-1", {"score": 1})
        log.record("row-2", {"score": 2})
        log.close()

        resumed = RunLog(log_path, CONFIG_A)
        assert resumed.done_keys() == {"row-1", "row-2"}

        items = ["row-1", "row-2", "row-3", "row-4"]
        pending = list(resumed.iter_pending(items, key_fn=lambda x: x))
        assert pending == ["row-3", "row-4"]
        resumed.close()

    def test_record_updates_done_keys_live(self, tmp_path: Path):
        log = RunLog(tmp_path / "rows.jsonl", CONFIG_A)
        assert log.done_keys() == set()
        log.record("row-1", {"score": 1})
        assert log.done_keys() == {"row-1"}
        log.close()


class TestTornFinalLine:
    def test_torn_final_line_tolerated_and_truncated(self, tmp_path: Path):
        log_path = tmp_path / "rows.jsonl"

        log = RunLog(log_path, CONFIG_A)
        log.record("row-1", {"score": 1})
        log.close()

        good_size = log_path.stat().st_size
        # Simulate a kill mid-write: append a truncated JSON line with no
        # trailing newline and no closing brace.
        with log_path.open("ab") as fh:
            fh.write(b'{"key": "row-2", "score": 2, "note": "unfinis')

        assert log_path.stat().st_size > good_size

        resumed = RunLog(log_path, CONFIG_A)
        assert resumed.done_keys() == {"row-1"}
        # The torn tail was dropped from disk before the next append.
        assert log_path.stat().st_size == good_size

        # And the log is writable again after the truncation.
        resumed.record("row-2", {"score": 2})
        resumed.close()

        reread = RunLog(log_path, CONFIG_A)
        assert reread.done_keys() == {"row-1", "row-2"}
        reread.close()

    def test_malformed_non_final_line_raises(self, tmp_path: Path):
        log_path = tmp_path / "rows.jsonl"
        log_path.write_text(
            '{"key": "row-1", "score": 1}\nnot json at all\n{"key": "row-2"}\n',
            encoding="utf-8",
        )
        with pytest.raises(RunLogError):
            RunLog(log_path, CONFIG_A)


class TestFingerprintMismatch:
    def test_mismatched_config_refuses(self, tmp_path: Path):
        log_path = tmp_path / "rows.jsonl"
        log = RunLog(log_path, CONFIG_A)
        log.record("row-1", {"score": 1})
        log.close()

        with pytest.raises(RunLogError):
            RunLog(log_path, CONFIG_B)

    def test_fresh_true_bypasses_mismatch(self, tmp_path: Path):
        log_path = tmp_path / "rows.jsonl"
        log = RunLog(log_path, CONFIG_A)
        log.record("row-1", {"score": 1})
        log.close()

        fresh_log = RunLog(log_path, CONFIG_B, fresh=True)
        assert fresh_log.done_keys() == set()
        fresh_log.close()

    def test_matching_config_reopens_cleanly(self, tmp_path: Path):
        log_path = tmp_path / "rows.jsonl"
        RunLog(log_path, CONFIG_A).close()
        # Should not raise.
        RunLog(log_path, CONFIG_A).close()


class TestFinalizeAtomicity:
    def test_summary_written_atomically(self, tmp_path: Path):
        log_path = tmp_path / "rows.jsonl"
        log = RunLog(log_path, CONFIG_A)
        log.record("row-1", {"score": 1})
        log.finalize({"n": 1, "mean": 1.0})

        summary_path = log_path.with_name(log_path.name + ".summary.json")
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert data == {"n": 1, "mean": 1.0}

        meta = json.loads((log_path.with_name(log_path.name + ".meta.json")).read_text())
        assert meta["complete"] is True

    def test_old_summary_intact_if_killed_between_tmp_and_replace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        log_path = tmp_path / "rows.jsonl"
        log = RunLog(log_path, CONFIG_A)
        log.record("row-1", {"score": 1})
        log.finalize({"n": 1, "mean": 1.0})

        summary_path = log_path.with_name(log_path.name + ".summary.json")
        original_bytes = summary_path.read_bytes()

        def _boom(*args, **kwargs):
            raise OSError("simulated crash between tmp write and os.replace")

        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(OSError):
            log.finalize({"n": 2, "mean": 2.0})

        # The prior summary is untouched: os.replace never ran.
        assert summary_path.read_bytes() == original_bytes

        tmp_summary = summary_path.with_name(summary_path.name + ".tmp")
        assert tmp_summary.exists()
        assert json.loads(tmp_summary.read_text())["n"] == 2


class TestIterPendingStability:
    def test_order_preserved_with_mixed_done_and_pending(self, tmp_path: Path):
        log = RunLog(tmp_path / "rows.jsonl", CONFIG_A)
        log.record("b", {})
        log.record("d", {})

        items = ["a", "b", "c", "d", "e"]
        pending = list(log.iter_pending(items, key_fn=lambda x: x))
        assert pending == ["a", "c", "e"]
        log.close()

    def test_iter_pending_empty_when_all_done(self, tmp_path: Path):
        log = RunLog(tmp_path / "rows.jsonl", CONFIG_A)
        for k in ["a", "b", "c"]:
            log.record(k, {})
        pending = list(log.iter_pending(["a", "b", "c"], key_fn=lambda x: x))
        assert pending == []
        log.close()


class TestLiveReadWhileWriting:
    def test_peek_done_keys_sees_flushed_records_without_disturbing_writer(
        self, tmp_path: Path
    ):
        log_path = tmp_path / "rows.jsonl"
        writer = RunLog(log_path, CONFIG_A)
        writer.record("row-1", {"score": 1})

        # A concurrent reader peeking at the same path while the writer is
        # still open must see the flushed record and must not touch the file.
        peeked = RunLog.peek_done_keys(log_path)
        assert peeked == {"row-1"}

        # The writer keeps working after being peeked at.
        writer.record("row-2", {"score": 2})
        writer.close()

        final_peek = RunLog.peek_done_keys(log_path)
        assert final_peek == {"row-1", "row-2"}

    def test_peek_before_any_write_is_empty(self, tmp_path: Path):
        log_path = tmp_path / "rows.jsonl"
        assert RunLog.peek_done_keys(log_path) == set()
