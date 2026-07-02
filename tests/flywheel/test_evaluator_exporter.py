import json

import pytest
import yaml

from shared.flywheel.catalog import InferenceLogRecord, LogFilter
from shared.flywheel.evaluator_exporter import EvaluatorFixtureExporter


class FakeCatalog:
    def __init__(self, records):
        self.records = records
        self.last_filter = None

    async def find_logs(self, filters: LogFilter):
        self.last_filter = filters
        return self.records


def _record(tmp_path, *, log_id="log-1", response_content="done", fitness_score=0.9):
    source = tmp_path / f"{log_id}.jsonl"
    content = {
        "messages": [{"role": "user", "content": "Do the task"}],
        "response_content": response_content,
        "tool_calls": [{"function": {"name": "useTools", "arguments": {"tool": "ls"}}}],
    }
    source.write_text(json.dumps(content) + "\n", encoding="utf-8")
    return InferenceLogRecord(
        log_id=log_id,
        timestamp="2026-01-01T00:00:00Z",
        model_id="model-a",
        fitness_score=fitness_score,
        is_valid=True,
        tag="sft",
        source_file=str(source),
        line_number=0,
    )


def _config(tmp_path, **overrides):
    data = {
        "catalog_filter": {"tag": "sft", "limit": 5},
        "output": {
            "name": "Frozen Flywheel Fixtures",
            "description": "Exported from flywheel logs",
            "test": {
                "id": "flywheel_{{ log_id }}",
                "messages": "{{ content.messages }}",
                "tags": ["flywheel", "{{ tag }}"],
                "correct": {
                    "tool_calls": "{{ content.tool_calls }}",
                    "response_text": "{{ content.response_content }}",
                },
            },
        },
    }
    data.update(overrides)
    path = tmp_path / "export.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_export_preserves_exact_placeholder_native_values(tmp_path):
    record = _record(tmp_path)
    catalog = FakeCatalog([record])
    output = tmp_path / "scenario.yaml"

    result = await EvaluatorFixtureExporter(catalog).export(
        _config(tmp_path),
        output,
        overwrite=False,
    )

    assert result.exported_count == 1
    assert catalog.last_filter == LogFilter(tag="sft", limit=5)
    exported = yaml.safe_load(output.read_text(encoding="utf-8"))
    test = exported["tests"][0]
    assert test["id"] == "flywheel_log-1"
    assert test["messages"] == [{"role": "user", "content": "Do the task"}]
    assert test["correct"]["tool_calls"][0]["function"]["name"] == "useTools"


@pytest.mark.asyncio
async def test_exported_yaml_loads_as_evaluator_scenario(tmp_path):
    config_dir = tmp_path / "Evaluator" / "config"
    scenarios_dir = config_dir / "scenarios"
    scenarios_dir.mkdir(parents=True)
    output = scenarios_dir / "flywheel_frozen.yaml"

    await EvaluatorFixtureExporter(FakeCatalog([_record(tmp_path)])).export(
        _config(tmp_path),
        output,
    )

    from Evaluator.config_loader import ConfigLoader

    cases = ConfigLoader(config_dir).load_all_scenarios(["flywheel_frozen.yaml"])

    assert len(cases) == 1
    assert cases[0].case_id == "flywheel_log-1"
    assert cases[0].metadata["messages"] == [{"role": "user", "content": "Do the task"}]
    assert cases[0].metadata["correct"]["response_text"] == "done"


@pytest.mark.asyncio
async def test_export_applies_optional_rollout_filters_to_content(tmp_path):
    keep = _record(tmp_path, log_id="keep", response_content="keep this", fitness_score=0.8)
    drop = _record(tmp_path, log_id="drop", response_content="drop this", fitness_score=0.2)
    config = _config(
        tmp_path,
        filters=[{"field": "fitness_score", "op": "gte", "value": 0.5}],
    )

    result = await EvaluatorFixtureExporter(FakeCatalog([keep, drop])).export(
        config,
        tmp_path / "scenario.yaml",
    )

    assert result.selected_count == 2
    assert result.exported_count == 1
    assert result.skipped_by_filter_count == 1


@pytest.mark.asyncio
async def test_export_unresolved_template_variable_is_fatal(tmp_path):
    config = _config(tmp_path)
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["output"]["test"]["id"] = "bad_{{ content.nope }}"
    config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Unresolved template variable: content.nope"):
        await EvaluatorFixtureExporter(FakeCatalog([_record(tmp_path)])).export(
            config,
            tmp_path / "scenario.yaml",
        )


@pytest.mark.asyncio
async def test_export_refuses_existing_output_without_overwrite(tmp_path):
    output = tmp_path / "scenario.yaml"
    output.write_text("name: existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Pass --yes"):
        await EvaluatorFixtureExporter(FakeCatalog([_record(tmp_path)])).export(
            _config(tmp_path),
            output,
        )


@pytest.mark.asyncio
async def test_dry_run_does_not_write_output(tmp_path):
    output = tmp_path / "scenario.yaml"

    result = await EvaluatorFixtureExporter(FakeCatalog([_record(tmp_path)])).export(
        _config(tmp_path),
        output,
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.exported_count == 1
    assert not output.exists()
