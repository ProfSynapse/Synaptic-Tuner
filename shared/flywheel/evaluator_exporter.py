"""Export flywheel inference logs as frozen Evaluator scenario fixtures."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from shared.validation import RolloutFilterSet
from shared.validation.rollout_filters import MISSING, get_path

from .catalog import InferenceLogRecord, LogCatalog, LogFilter
from .utils import read_log_content


_PLACEHOLDER_RE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*}}")
_EXACT_PLACEHOLDER_RE = re.compile(r"^\s*{{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*}}\s*$")
_DEFAULT_FILTER_TARGET = "evaluator_fixture"


@dataclass
class EvaluatorFixtureExportResult:
    """Summary of an Evaluator fixture export."""

    output_path: str
    selected_count: int
    hydrated_count: int
    exported_count: int
    skipped_by_filter_count: int
    missing_content_count: int
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvaluatorFixtureExporter:
    """Config-driven exporter from flywheel logs to Evaluator YAML scenarios."""

    def __init__(self, catalog: LogCatalog) -> None:
        self._catalog = catalog

    async def export(
        self,
        export_config_path: str | Path,
        output_path: str | Path,
        *,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> EvaluatorFixtureExportResult:
        config_path = Path(export_config_path)
        output = Path(output_path)
        config = load_export_config(config_path)

        if output.exists() and not overwrite and not dry_run:
            raise FileExistsError(
                f"Output file already exists: {output}. Pass --yes to overwrite."
            )

        records = await self._catalog.find_logs(_build_log_filter(config.get("catalog_filter") or {}))
        scenario, result = build_evaluator_scenario(
            records,
            config,
            output_path=output,
        )

        if dry_run:
            result.dry_run = True
            return result

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            yaml.safe_dump(scenario, f, sort_keys=False, allow_unicode=False)
        return result


def load_export_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate an exporter config YAML file."""
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("export config must be a YAML object")

    output = config.get("output")
    if not isinstance(output, dict):
        raise ValueError("export config requires an 'output' mapping")

    test_template = output.get("test")
    if not isinstance(test_template, dict):
        raise ValueError("export config requires output.test mapping")

    correct = test_template.get("correct")
    if not isinstance(correct, dict) or not correct:
        raise ValueError("output.test.correct is required and must be a non-empty mapping")

    has_prompt = any(key in test_template for key in ("question", "messages"))
    if not has_prompt:
        raise ValueError("output.test must configure either 'question' or 'messages'")

    if "catalog_filter" in config and not isinstance(config["catalog_filter"], dict):
        raise ValueError("catalog_filter must be a mapping")
    if "filters" in config and not isinstance(config["filters"], list):
        raise ValueError("filters must be a list")

    return config


def build_evaluator_scenario(
    records: list[InferenceLogRecord],
    config: dict[str, Any],
    *,
    output_path: str | Path,
) -> tuple[dict[str, Any], EvaluatorFixtureExportResult]:
    """Build an Evaluator scenario payload from already-selected records."""
    output_cfg = config["output"]
    test_template = output_cfg["test"]
    filters = RolloutFilterSet(
        filters=config.get("filters") or [],
        default_targets=[str(config.get("filter_target") or _DEFAULT_FILTER_TARGET)],
    )
    filter_target = str(config.get("filter_target") or _DEFAULT_FILTER_TARGET)

    tests: list[dict[str, Any]] = []
    hydrated = 0
    skipped_by_filter = 0
    missing_content = 0

    for index, record in enumerate(records):
        content = read_log_content(record)
        if content is None:
            missing_content += 1
            continue
        hydrated += 1

        context = _template_context(record, content, index)
        if not filters.is_empty and not filters.apply(context, filter_target).passed:
            skipped_by_filter += 1
            continue

        tests.append(_render_test(test_template, context))

    scenario = {
        "name": _render_value(output_cfg.get("name", "Flywheel Evaluator Fixtures"), {"derived": {"export_count": len(tests)}}),
        "description": _render_value(output_cfg.get("description", ""), {"derived": {"export_count": len(tests)}}),
    }
    defaults = output_cfg.get("defaults")
    if isinstance(defaults, dict) and defaults:
        scenario["defaults"] = _render_value(defaults, {"derived": {"export_count": len(tests)}})
    metadata = output_cfg.get("metadata")
    if isinstance(metadata, dict) and metadata:
        scenario["metadata"] = _render_value(metadata, {"derived": {"export_count": len(tests)}})
    scenario["tests"] = tests

    return scenario, EvaluatorFixtureExportResult(
        output_path=str(output_path),
        selected_count=len(records),
        hydrated_count=hydrated,
        exported_count=len(tests),
        skipped_by_filter_count=skipped_by_filter,
        missing_content_count=missing_content,
    )


def _build_log_filter(raw: dict[str, Any]) -> LogFilter:
    valid_fields = set(LogFilter.__dataclass_fields__)
    unknown = set(raw) - valid_fields
    if unknown:
        raise ValueError(f"catalog_filter has unknown key(s): {sorted(unknown)}")
    return LogFilter(**raw)


def _template_context(
    record: InferenceLogRecord,
    content: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    context = asdict(record)
    context["content"] = content
    context["record"] = asdict(record)
    context["derived"] = {
        "index": index,
        "ordinal": index + 1,
    }
    return context


def _render_test(template: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    rendered = _render_value(template, context)
    if not isinstance(rendered, dict):
        raise ValueError("rendered test must be a mapping")
    if not rendered.get("id"):
        raise ValueError("rendered test is missing required id")
    if "correct" not in rendered or not isinstance(rendered["correct"], dict) or not rendered["correct"]:
        raise ValueError(f"rendered test {rendered.get('id')!r} is missing non-empty correct mapping")
    if "question" not in rendered and "messages" not in rendered:
        raise ValueError(f"rendered test {rendered.get('id')!r} is missing question/messages")
    return rendered


def _render_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        exact = _EXACT_PLACEHOLDER_RE.match(value)
        if exact:
            return _resolve_template_path(context, exact.group(1))

        def replace(match: re.Match[str]) -> str:
            resolved = _resolve_template_path(context, match.group(1))
            return "" if resolved is None else str(resolved)

        return _PLACEHOLDER_RE.sub(replace, value)
    if isinstance(value, list):
        return [_render_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _render_value(item, context) for key, item in value.items()}
    return value


def _resolve_template_path(context: dict[str, Any], path: str) -> Any:
    value = get_path(context, path)
    if value is MISSING:
        raise ValueError(f"Unresolved template variable: {path}")
    return value


__all__ = [
    "EvaluatorFixtureExportResult",
    "EvaluatorFixtureExporter",
    "build_evaluator_scenario",
    "load_export_config",
]
