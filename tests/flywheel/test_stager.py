"""Tests for shared.flywheel.stager — DatasetStager JSONL assembly."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.flywheel.catalog import DatasetVersion, InferenceLogRecord, LogFilter
from shared.flywheel.config import FlywheelConfig
from shared.flywheel.stager import DatasetStager, StagingResult


def _make_record(
    log_id: str,
    source_file: str = "",
    line_number: int = 0,
    **kwargs,
) -> InferenceLogRecord:
    defaults = dict(
        timestamp="2026-01-15T12:00:00Z",
        model_id="test-model",
    )
    defaults.update(kwargs)
    return InferenceLogRecord(
        log_id=log_id,
        source_file=source_file,
        line_number=line_number,
        **defaults,
    )


def _write_log_file(path: Path, records: list[dict]) -> None:
    """Write multiple log content dicts as JSONL lines."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestStagingResult:
    """StagingResult dataclass defaults."""

    def test_defaults(self):
        r = StagingResult()
        assert r.version_id == ""
        assert r.sft_count == 0
        assert r.kto_pos_count == 0
        assert r.grpo_count == 0
        assert r.file_paths == {}


class TestDatasetStagerWrite:
    """DatasetStager writes correct JSONL formats for each training type."""

    def _make_stager(self, tmp_path, **config_kwargs):
        catalog = AsyncMock()
        catalog.find_logs = AsyncMock(return_value=[])
        catalog.get_latest_dataset_version = AsyncMock(return_value=None)
        catalog.create_dataset_version = AsyncMock(return_value="v001")
        catalog.mark_used = AsyncMock()
        cfg = FlywheelConfig(**config_kwargs)
        return DatasetStager(catalog, cfg, datasets_dir=tmp_path / "datasets")

    @pytest.mark.asyncio
    async def test_sft_jsonl_format(self, tmp_path):
        """SFT output has conversations + label: true."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [{
            "messages": [{"role": "user", "content": "Hi"}],
            "response_content": "Hello!",
        }])

        sft_record = _make_record(
            "sft-1", source_file=str(log_file), line_number=0, tag="sft",
        )

        stager = self._make_stager(tmp_path)
        stager._catalog.find_logs = AsyncMock(side_effect=[
            [sft_record],  # sft query
            [],            # kto query
            [],            # grpo query
        ])

        with patch.object(stager, "_register_flywheel_cycle", return_value="run-1"):
            result = await stager.stage_dataset()

        assert result.sft_count == 1
        sft_path = Path(result.file_paths["sft"])
        assert sft_path.exists()

        with open(sft_path) as f:
            example = json.loads(f.readline())
        assert example["label"] is True
        assert example["conversations"][-1]["role"] == "assistant"
        assert example["conversations"][-1]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_kto_jsonl_format(self, tmp_path):
        """KTO output has conversations + label field (true for positive, false for negative)."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [
            {"messages": [{"role": "user", "content": "Good"}], "response_content": "Great!"},
            {"messages": [{"role": "user", "content": "Bad"}], "response_content": "Wrong"},
        ])

        sft_record = _make_record(
            "kto-pos", source_file=str(log_file), line_number=0, tag="sft",
        )
        kto_record = _make_record(
            "kto-neg", source_file=str(log_file), line_number=1, tag="kto",
        )

        stager = self._make_stager(tmp_path)
        stager._catalog.find_logs = AsyncMock(side_effect=[
            [sft_record],   # sft query
            [kto_record],   # kto query
            [],             # grpo query
        ])

        with patch.object(stager, "_register_flywheel_cycle", return_value="run-1"):
            result = await stager.stage_dataset()

        assert result.kto_pos_count == 1
        assert result.kto_neg_count == 1

        kto_path = Path(result.file_paths["kto"])
        lines = kto_path.read_text().strip().splitlines()
        assert len(lines) == 2

        pos = json.loads(lines[0])
        neg = json.loads(lines[1])
        assert pos["label"] is True
        assert neg["label"] is False

    @pytest.mark.asyncio
    async def test_grpo_jsonl_format(self, tmp_path):
        """GRPO output has the static trainer schema."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [{
            "messages": [{"role": "user", "content": "Search for X"}],
            "response_content": "Found X",
            "tool_calls": [{
                "function": {
                    "name": "search",
                    "arguments": "{\"query\":\"X\"}",
                },
            }],
        }])

        grpo_record = _make_record(
            "grpo-1", source_file=str(log_file), line_number=0,
            tag="grpo", fitness_score=0.7, is_valid=True,
            tools_requested=True, tool_calls=[{}],
        )

        stager = self._make_stager(tmp_path)
        stager._catalog.find_logs = AsyncMock(side_effect=[
            [],             # sft query
            [],             # kto query
            [grpo_record],  # grpo query
        ])

        with patch.object(stager, "_register_flywheel_cycle", return_value="run-1"):
            result = await stager.stage_dataset()

        assert result.grpo_count == 1
        grpo_path = Path(result.file_paths["grpo"])
        example = json.loads(grpo_path.read_text().strip())
        assert example == {
            "prompt": [{"role": "user", "content": "Search for X"}],
            "ground_truth_tool": "search",
            "ground_truth_args_json": "{\"query\":\"X\"}",
        }
        assert "reward" not in example
        assert "conversations" not in example

        grpo_filter = stager._catalog.find_logs.call_args_list[2].args[0]
        assert grpo_filter.tag == ["sft", "grpo"]
        assert grpo_filter.unused_only is True
        assert grpo_filter.is_valid is True
        assert grpo_filter.has_tool_calls is True

    @pytest.mark.asyncio
    async def test_grpo_eligible_sft_tool_call_log_staged(self, tmp_path):
        """SFT-tagged valid tool-call logs are eligible for GRPO staging."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [{
            "messages": [{"role": "user", "content": "Lookup A"}],
            "response_content": "",
            "tool_calls": [{
                "function": {"name": "lookup", "arguments": {"id": "A"}},
            }],
        }])

        record = _make_record(
            "sft-tool-1", source_file=str(log_file), line_number=0,
            tag="sft", fitness_score=1.0, is_valid=True,
            tools_requested=True, tool_calls=[{}],
        )

        stager = self._make_stager(tmp_path)
        stager._catalog.find_logs = AsyncMock(side_effect=[
            [],        # sft query
            [],        # kto query
            [record],  # grpo eligibility query can return sft-tagged logs
        ])

        with patch.object(stager, "_register_flywheel_cycle", return_value="run-1"):
            result = await stager.stage_dataset()

        assert result.grpo_count == 1
        grpo_path = Path(result.file_paths["grpo"])
        example = json.loads(grpo_path.read_text().strip())
        assert example["ground_truth_tool"] == "lookup"
        assert example["ground_truth_args_json"] == "{\"id\": \"A\"}"

    @pytest.mark.asyncio
    async def test_grpo_requires_tools_requested(self, tmp_path):
        """GRPO staging locally rejects rows without tools requested."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [{
            "messages": [{"role": "user", "content": "Lookup A"}],
            "response_content": "",
            "tool_calls": [{
                "function": {"name": "lookup", "arguments": "{}"},
            }],
        }])

        record = _make_record(
            "no-tools-requested", source_file=str(log_file), line_number=0,
            tag="grpo", fitness_score=1.0, is_valid=True,
            tools_requested=False, tool_calls=[{}],
        )

        stager = self._make_stager(tmp_path)
        stager._catalog.find_logs = AsyncMock(side_effect=[
            [], [], [record],
        ])

        result = await stager.stage_dataset()

        assert result.version_id == ""
        assert result.grpo_count == 0

    @pytest.mark.asyncio
    async def test_grpo_disabled_skips(self, tmp_path):
        """GRPO logs not staged when grpo_enabled=False."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [{
            "messages": [{"role": "user", "content": "X"}],
            "response_content": "Y",
            "tool_calls": [{
                "function": {"name": "lookup", "arguments": "{}"},
            }],
        }])

        grpo_record = _make_record(
            "grpo-skip", source_file=str(log_file), line_number=0,
            tag="grpo", fitness_score=0.6, is_valid=True,
            tools_requested=True, tool_calls=[{}],
        )

        stager = self._make_stager(tmp_path, grpo_enabled=False)
        stager._catalog.find_logs = AsyncMock(side_effect=[
            [],              # sft
            [],              # kto
            [grpo_record],   # grpo
        ])

        with patch.object(
            stager, "_register_flywheel_cycle", return_value="",
        ) as register:
            result = await stager.stage_dataset()

        assert result.version_id == ""
        assert result.grpo_count == 0
        assert result.total_records == 0
        assert "grpo" not in result.file_paths
        stager._catalog.create_dataset_version.assert_not_called()
        stager._catalog.mark_used.assert_not_called()
        register.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_logs_returns_empty_result(self, tmp_path):
        stager = self._make_stager(tmp_path)
        stager._catalog.find_logs = AsyncMock(return_value=[])

        result = await stager.stage_dataset()
        assert result.version_id == ""
        assert result.total_records == 0


class TestDatasetStagerVersioning:
    """DatasetStager version ID generation and dataset version creation."""

    def test_next_version_id_empty_dir(self, tmp_path):
        catalog = AsyncMock()
        stager = DatasetStager(catalog, FlywheelConfig(), datasets_dir=tmp_path / "ds")
        assert stager._next_version_id() == "v001"

    def test_next_version_id_increments(self, tmp_path):
        ds_dir = tmp_path / "ds"
        (ds_dir / "v001").mkdir(parents=True)
        (ds_dir / "v002").mkdir()

        catalog = AsyncMock()
        stager = DatasetStager(catalog, FlywheelConfig(), datasets_dir=ds_dir)
        assert stager._next_version_id() == "v003"

    @pytest.mark.asyncio
    async def test_creates_dataset_version_in_catalog(self, tmp_path):
        """stage_dataset creates a DatasetVersion entry in the catalog."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [{
            "messages": [{"role": "user", "content": "Hi"}],
            "response_content": "Hello!",
        }])

        sft_record = _make_record(
            "v-1", source_file=str(log_file), line_number=0, tag="sft",
        )

        catalog = AsyncMock()
        catalog.find_logs = AsyncMock(side_effect=[
            [sft_record], [], [],  # sft, kto, grpo
        ])
        catalog.get_latest_dataset_version = AsyncMock(return_value=None)
        catalog.create_dataset_version = AsyncMock(return_value="v001")
        catalog.mark_used = AsyncMock()

        stager = DatasetStager(catalog, FlywheelConfig(), datasets_dir=tmp_path / "ds")

        with patch.object(stager, "_register_flywheel_cycle", return_value="run-1"):
            result = await stager.stage_dataset()

        catalog.create_dataset_version.assert_called_once()
        version_arg = catalog.create_dataset_version.call_args[0][0]
        assert isinstance(version_arg, DatasetVersion)
        assert version_arg.record_counts["sft"] == 1

    @pytest.mark.asyncio
    async def test_marks_logs_as_used(self, tmp_path):
        """stage_dataset calls mark_used with all consumed log IDs."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [{
            "messages": [{"role": "user", "content": "Hi"}],
            "response_content": "Hello!",
        }])

        record = _make_record(
            "mu-1", source_file=str(log_file), line_number=0, tag="sft",
        )

        catalog = AsyncMock()
        catalog.find_logs = AsyncMock(side_effect=[[record], [], []])
        catalog.get_latest_dataset_version = AsyncMock(return_value=None)
        catalog.create_dataset_version = AsyncMock(return_value="v001")
        catalog.mark_used = AsyncMock()

        stager = DatasetStager(catalog, FlywheelConfig(), datasets_dir=tmp_path / "ds")

        with patch.object(stager, "_register_flywheel_cycle", return_value=""):
            await stager.stage_dataset()

        catalog.mark_used.assert_called_once()
        log_ids = catalog.mark_used.call_args[0][0]
        assert "mu-1" in log_ids


class TestContentHash:
    """DatasetStager._compute_content_hash produces consistent hashes."""

    def test_same_content_same_hash(self, tmp_path):
        catalog = AsyncMock()
        stager = DatasetStager(catalog, FlywheelConfig())

        f1 = tmp_path / "a.jsonl"
        f1.write_text('{"test": 1}\n')
        f2 = tmp_path / "b.jsonl"
        f2.write_text('{"test": 2}\n')

        h1 = stager._compute_content_hash({"sft": f1, "kto": f2})
        h2 = stager._compute_content_hash({"sft": f1, "kto": f2})
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self, tmp_path):
        catalog = AsyncMock()
        stager = DatasetStager(catalog, FlywheelConfig())

        f1 = tmp_path / "a.jsonl"
        f1.write_text('{"test": 1}\n')
        f2 = tmp_path / "b.jsonl"
        f2.write_text('{"test": 2}\n')
        f3 = tmp_path / "c.jsonl"
        f3.write_text('{"test": 3}\n')

        h1 = stager._compute_content_hash({"sft": f1})
        h2 = stager._compute_content_hash({"sft": f3})
        assert h1 != h2


class TestDatasetStagerFilters:
    """DatasetStager applies declarative staging filters at each write seam."""

    def _make_catalog(self, *, sft=None, kto=None, grpo=None):
        catalog = AsyncMock()
        catalog.find_logs = AsyncMock(side_effect=[
            sft or [], kto or [], grpo or [],
        ])
        catalog.get_latest_dataset_version = AsyncMock(return_value=None)
        catalog.create_dataset_version = AsyncMock(return_value="v001")
        catalog.mark_used = AsyncMock()
        return catalog

    @pytest.mark.asyncio
    async def test_no_filters_output_unchanged(self, tmp_path):
        """Regression lock: no filters configured -> identical counts and rows.

        Stage the same logs with an empty filter config and with no filter key
        at all; both must produce the same counts and the same JSONL bytes.
        """
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [
            {
                "messages": [{"role": "user", "content": "a"}],
                "response_content": "A",
                "tool_calls": [{
                    "function": {"name": "answer", "arguments": "{}"},
                }],
            },
            {"messages": [{"role": "user", "content": "b"}], "response_content": "B"},
            {"messages": [{"role": "user", "content": "c"}], "response_content": "C"},
        ])

        def records():
            sft = [
                _make_record("s1", str(log_file), 0, tag="sft", fitness_score=0.95),
                _make_record("s2", str(log_file), 1, tag="sft", fitness_score=0.5),
            ]
            kto = [_make_record("k1", str(log_file), 2, tag="kto", fitness_score=0.1)]
            grpo = [
                _make_record(
                    "g1", str(log_file), 0, tag="grpo", fitness_score=0.95,
                    is_valid=True, tools_requested=True, tool_calls=[{}],
                )
            ]
            return sft, kto, grpo

        # Run with no filters key.
        sft, kto, grpo = records()
        cat1 = self._make_catalog(sft=sft, kto=kto, grpo=grpo)
        stager1 = DatasetStager(cat1, FlywheelConfig(), datasets_dir=tmp_path / "ds1")
        with patch.object(stager1, "_register_flywheel_cycle", return_value=""):
            r1 = await stager1.stage_dataset()

        # Run with explicit empty filters list.
        sft, kto, grpo = records()
        cat2 = self._make_catalog(sft=sft, kto=kto, grpo=grpo)
        stager2 = DatasetStager(
            cat2, FlywheelConfig(filters=[]), datasets_dir=tmp_path / "ds2",
        )
        with patch.object(stager2, "_register_flywheel_cycle", return_value=""):
            r2 = await stager2.stage_dataset()

        assert (r1.sft_count, r1.kto_pos_count, r1.kto_neg_count, r1.grpo_count) == (2, 2, 1, 1)
        assert (r1.sft_count, r1.kto_pos_count, r1.kto_neg_count, r1.grpo_count) == \
               (r2.sft_count, r2.kto_pos_count, r2.kto_neg_count, r2.grpo_count)

        for key in ("sft", "kto", "grpo"):
            b1 = Path(r1.file_paths[key]).read_bytes()
            b2 = Path(r2.file_paths[key]).read_bytes()
            assert b1 == b2

    @pytest.mark.asyncio
    async def test_fitness_score_filter_drops_low_from_targets(self, tmp_path):
        """fitness_score gte 0.9 drops low-score logs from sft/kto_positive/grpo."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [
            {
                "messages": [{"role": "user", "content": "hi"}],
                "response_content": "ok",
                "tool_calls": [{
                    "function": {"name": "answer", "arguments": "{}"},
                }],
            },
        ] * 4)

        sft = [
            _make_record("hi", str(log_file), 0, tag="sft", fitness_score=0.95),
            _make_record("lo", str(log_file), 0, tag="sft", fitness_score=0.5),
        ]
        # KTO positive comes from sft_logs; negative from kto_logs.
        kto = [_make_record("kn", str(log_file), 0, tag="kto", fitness_score=0.1)]
        grpo = [
            _make_record(
                "ghi", str(log_file), 0, tag="grpo", fitness_score=0.95,
                is_valid=True, tools_requested=True, tool_calls=[{}],
            ),
            _make_record(
                "glo", str(log_file), 0, tag="grpo", fitness_score=0.2,
                is_valid=True, tools_requested=True, tool_calls=[{}],
            ),
        ]

        cat = self._make_catalog(sft=sft, kto=kto, grpo=grpo)
        cfg = FlywheelConfig(filters=[
            {"field": "fitness_score", "op": "gte", "value": 0.9},
        ])
        stager = DatasetStager(cat, cfg, datasets_dir=tmp_path / "ds")
        with patch.object(stager, "_register_flywheel_cycle", return_value=""):
            r = await stager.stage_dataset()

        # sft: only hi (0.95) kept; lo dropped.
        assert r.sft_count == 1
        # kto_positive uses the same sft_logs view -> same 1 kept.
        assert r.kto_pos_count == 1
        # grpo: only ghi (0.95) kept.
        assert r.grpo_count == 1
        # kto_negative is NOT in default targets -> never filtered.
        assert r.kto_neg_count == 1

    @pytest.mark.asyncio
    async def test_missing_field_passes_under_on_missing_keep(self, tmp_path):
        """A record missing the addressed field passes under on_missing: keep.

        ``content.score`` is genuinely absent from logs that don't carry it, so
        the dot-path resolves to MISSING and on_missing: keep retains the row.
        """
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [
            # First log carries content.score; second does not.
            {"messages": [{"role": "user", "content": "a"}], "response_content": "A", "score": 0.95},
            {"messages": [{"role": "user", "content": "b"}], "response_content": "B"},
            {"messages": [{"role": "user", "content": "c"}], "response_content": "C", "score": 0.1},
        ])

        sft = [
            _make_record("has_hi", str(log_file), 0, tag="sft"),
            _make_record("no_field", str(log_file), 1, tag="sft"),
            _make_record("has_lo", str(log_file), 2, tag="sft"),
        ]
        cat = self._make_catalog(sft=sft, kto=[], grpo=[])
        cfg = FlywheelConfig(filters=[
            {"field": "content.score", "op": "gte", "value": 0.9, "on_missing": "keep"},
        ])
        stager = DatasetStager(cat, cfg, datasets_dir=tmp_path / "ds")
        with patch.object(stager, "_register_flywheel_cycle", return_value=""):
            r = await stager.stage_dataset()

        # has_hi (0.95) kept by predicate; no_field kept by on_missing: keep;
        # has_lo (0.1) dropped.
        assert r.sft_count == 2

    @pytest.mark.asyncio
    async def test_kto_negative_only_filtered_when_scoped(self, tmp_path):
        """Unscoped filter never touches kto_negative; a scoped one does."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [
            {"messages": [{"role": "user", "content": "hi"}], "response_content": "ok"},
        ] * 4)

        def neg_records():
            return [
                _make_record("n_hi", str(log_file), 0, tag="kto", fitness_score=0.95),
                _make_record("n_lo", str(log_file), 0, tag="kto", fitness_score=0.1),
            ]

        # Unscoped filter (defaults exclude kto_negative): both negatives kept.
        cat1 = self._make_catalog(sft=[], kto=neg_records(), grpo=[])
        cfg1 = FlywheelConfig(filters=[
            {"field": "fitness_score", "op": "gte", "value": 0.9},
        ])
        stager1 = DatasetStager(cat1, cfg1, datasets_dir=tmp_path / "ds1")
        with patch.object(stager1, "_register_flywheel_cycle", return_value=""):
            r1 = await stager1.stage_dataset()
        assert r1.kto_neg_count == 2  # unscoped -> 0 dropped

        # Scoped to kto_negative: low-score negative dropped.
        cat2 = self._make_catalog(sft=[], kto=neg_records(), grpo=[])
        cfg2 = FlywheelConfig(filters=[
            {
                "field": "fitness_score", "op": "gte", "value": 0.9,
                "applies_to": ["kto_negative"],
            },
        ])
        stager2 = DatasetStager(cat2, cfg2, datasets_dir=tmp_path / "ds2")
        with patch.object(stager2, "_register_flywheel_cycle", return_value=""):
            r2 = await stager2.stage_dataset()
        assert r2.kto_neg_count == 1  # n_lo dropped

    @pytest.mark.asyncio
    async def test_kto_interleaving_preserved_after_filtering(self, tmp_path):
        """After filtering, positives/negatives still interleave (pos, neg, pos, ...)."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [
            {"messages": [{"role": "user", "content": "hi"}], "response_content": "ok"},
        ] * 6)

        # Two positives survive (0.95), one dropped (0.1). Two negatives (unscoped
        # filter never touches kto_negative).
        sft = [
            _make_record("p1", str(log_file), 0, tag="sft", fitness_score=0.95),
            _make_record("pdrop", str(log_file), 0, tag="sft", fitness_score=0.1),
            _make_record("p2", str(log_file), 0, tag="sft", fitness_score=0.95),
        ]
        kto = [
            _make_record("n1", str(log_file), 0, tag="kto", fitness_score=0.1),
            _make_record("n2", str(log_file), 0, tag="kto", fitness_score=0.1),
        ]

        cat = self._make_catalog(sft=sft, kto=kto, grpo=[])
        cfg = FlywheelConfig(filters=[
            {"field": "fitness_score", "op": "gte", "value": 0.9},
        ])
        stager = DatasetStager(cat, cfg, datasets_dir=tmp_path / "ds")
        with patch.object(stager, "_register_flywheel_cycle", return_value=""):
            r = await stager.stage_dataset()

        assert r.kto_pos_count == 2
        assert r.kto_neg_count == 2

        lines = Path(r.file_paths["kto"]).read_text().strip().splitlines()
        labels = [json.loads(line)["label"] for line in lines]
        # zip_longest interleave on filtered lists: pos, neg, pos, neg.
        assert labels == [True, False, True, False]

    @pytest.mark.asyncio
    async def test_filter_provenance_recorded_in_version(self, tmp_path):
        """The DatasetVersion records the staging filter specs + stats."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [
            {"messages": [{"role": "user", "content": "hi"}], "response_content": "ok"},
        ] * 2)

        sft = [
            _make_record("hi", str(log_file), 0, tag="sft", fitness_score=0.95),
            _make_record("lo", str(log_file), 0, tag="sft", fitness_score=0.1),
        ]
        cat = self._make_catalog(sft=sft, kto=[], grpo=[])
        cfg = FlywheelConfig(filters=[
            {"field": "fitness_score", "op": "gte", "value": 0.9},
        ])
        stager = DatasetStager(cat, cfg, datasets_dir=tmp_path / "ds")
        with patch.object(stager, "_register_flywheel_cycle", return_value=""):
            await stager.stage_dataset()

        version_arg = cat.create_dataset_version.call_args[0][0]
        crit = version_arg.filter_criteria
        # Existing keys preserved.
        assert "sft_threshold" in crit
        assert "scoring_method" in crit
        # New provenance keys.
        assert crit["staging_filters"] == [
            {"field": "fitness_score", "op": "gte", "value": 0.9, "on_missing": "keep"},
        ]
        assert "sft" in crit["staging_filter_stats"]

    def test_invalid_filter_spec_raises_at_init(self, tmp_path):
        """An invalid filter spec raises ValueError at stager construction."""
        catalog = AsyncMock()
        cfg = FlywheelConfig(filters=[{"field": "fitness_score", "op": "bogus", "value": 1}])
        with pytest.raises(ValueError):
            DatasetStager(catalog, cfg, datasets_dir=tmp_path / "ds")

    def test_content_prefix_addressable(self, tmp_path):
        """The filter view exposes parsed content under a 'content.' prefix."""
        record = _make_record("c1", fitness_score=0.5)
        content = {
            "messages": [{"role": "user", "content": "hi"}],
            "response_content": "the answer",
        }
        view = DatasetStager._filter_view(record, content)
        # Record field at top level.
        assert view["fitness_score"] == 0.5
        # Content nested under 'content'.
        assert view["content"]["response_content"] == "the answer"
