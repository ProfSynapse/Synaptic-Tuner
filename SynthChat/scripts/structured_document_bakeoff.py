#!/usr/bin/env python3
"""Run a config-defined structured-document model bakeoff.

The runner deliberately knows nothing about a particular corpus or document
format.  A YAML file supplies the document, prompt, JSON Schema, candidate
OpenRouter models, and (optionally) one fixed structured-output judge.  Every
remote result is written only to the configured local output directory.

Minimal configuration::

    source:
      path: private/source.md
      strip_yaml_frontmatter: true
    prompt_template: "Create a reverse outline for this document:\\n\\n{document}"
    response_schema:
      type: object
      properties: {outline: {type: string}}
      required: [outline]
      additionalProperties: false
    models:
      - id: provider/model-a
      - id: provider/model-b
        provider_routing: {data_collection: deny}
    temperature: 0.2
    max_tokens: 1800
    output_dir: .tracking/private-bakeoff
    judge:
      model: provider/fixed-judge
      prompt_template: "Source:\\n{document}\\n\\nCandidate:\\n{candidate}"
      response_schema: {type: object, properties: {score: {type: number}}, required: [score]}

Prompt templates use literal ``{document}`` and, for a judge, ``{candidate}``.
No model call is made by ``--dry-run``.  The script is intentionally a small
consumer of ``shared.llm`` rather than a second provider implementation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import jsonschema
import yaml

from shared.llm import create_client


ClientFactory = Callable[..., Any]


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a bakeoff YAML configuration without making calls."""
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Bakeoff config not found: {path}") from None
    config = _mapping(loaded, "bakeoff config")

    source = _mapping(config.get("source"), "source")
    _text(source.get("path"), "source.path")
    if "strip_yaml_frontmatter" in source and not isinstance(source["strip_yaml_frontmatter"], bool):
        raise ValueError("source.strip_yaml_frontmatter must be a boolean")
    config["source"] = source

    _template(config.get("prompt_template"), "prompt_template", "{document}")
    config["response_schema"] = _schema(config.get("response_schema"), "response_schema")
    if not isinstance(config.get("models"), list) or not config["models"]:
        raise ValueError("models must be a non-empty list")
    config["models"] = [_model_spec(item, index) for index, item in enumerate(config["models"])]
    config["temperature"] = _number(config.get("temperature"), "temperature")
    config["max_tokens"] = _positive_int(config.get("max_tokens"), "max_tokens")
    config["output_dir"] = _text(config.get("output_dir"), "output_dir")

    judge = config.get("judge")
    if judge is not None:
        config["judge"] = _judge_spec(judge, config["temperature"], config["max_tokens"])
    return config


def _template(value: object, name: str, *required_tokens: str) -> str:
    text = _text(value, name)
    missing = [token for token in required_tokens if token not in text]
    if missing:
        raise ValueError(f"{name} must include {', '.join(missing)}")
    return text


def _schema(value: object, name: str) -> dict[str, Any]:
    schema = _mapping(value, name)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise ValueError(f"{name} is not a valid JSON Schema: {exc.message}") from None
    return schema


def _instance_path(parts: Any) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        elif isinstance(part, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            path += f".{part}"
        else:
            path += f"[{json.dumps(part, ensure_ascii=False)}]"
    return path


def _validate_payload(value: object, schema: Mapping[str, Any], name: str) -> None:
    try:
        jsonschema.Draft202012Validator(dict(schema)).validate(value)
    except jsonschema.ValidationError as exc:
        location = _instance_path(exc.absolute_path)
        raise ValueError(f"{name} failed JSON Schema validation at {location}: {exc.message}") from None


def _model_spec(value: object, index: int) -> dict[str, Any]:
    if isinstance(value, str):
        return {"id": _text(value, f"models[{index}]")}
    model = _mapping(value, f"models[{index}]")
    allowed = {"id", "provider_routing"}
    unknown = set(model) - allowed
    if unknown:
        raise ValueError(f"models[{index}] has unsupported fields: {', '.join(sorted(unknown))}")
    model["id"] = _text(model.get("id"), f"models[{index}].id")
    if "provider_routing" in model:
        model["provider_routing"] = _mapping(model["provider_routing"], f"models[{index}].provider_routing")
    return model


def _judge_spec(value: object, default_temperature: float, default_max_tokens: int) -> dict[str, Any]:
    judge = _mapping(value, "judge")
    allowed = {"model", "prompt_template", "response_schema", "provider_routing", "temperature", "max_tokens"}
    unknown = set(judge) - allowed
    if unknown:
        raise ValueError(f"judge has unsupported fields: {', '.join(sorted(unknown))}")
    judge["model"] = _text(judge.get("model"), "judge.model")
    _template(judge.get("prompt_template"), "judge.prompt_template", "{document}", "{candidate}")
    judge["response_schema"] = _schema(judge.get("response_schema"), "judge.response_schema")
    if "provider_routing" in judge:
        judge["provider_routing"] = _mapping(judge["provider_routing"], "judge.provider_routing")
    judge["temperature"] = _number(judge.get("temperature", default_temperature), "judge.temperature")
    judge["max_tokens"] = _positive_int(judge.get("max_tokens", default_max_tokens), "judge.max_tokens")
    return judge


_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n.*?\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)


def read_source(config: Mapping[str, Any], config_path: Path) -> tuple[Path, str]:
    """Read the requested document; resolve relative paths from the YAML file."""
    source = _mapping(config["source"], "source")
    source_path = Path(_text(source["path"], "source.path"))
    if not source_path.is_absolute():
        source_path = (config_path.parent / source_path).resolve()
    text = source_path.read_text(encoding="utf-8")
    if source.get("strip_yaml_frontmatter", False):
        text = _FRONTMATTER.sub("", text, count=1)
    if not text.strip():
        raise ValueError(f"source document is empty after preprocessing: {source_path}")
    return source_path, text


def _render(template: str, **values: str) -> str:
    """Replace declared tokens without treating unrelated braces as formatting."""
    rendered = template
    for token, value in values.items():
        rendered = rendered.replace("{" + token + "}", value)
    return rendered


def _client(factory: ClientFactory, model: str, provider_routing: Mapping[str, Any] | None = None) -> Any:
    defaults: dict[str, Any] = {"provider": "openrouter", "model": model}
    if provider_routing is not None:
        defaults["provider_routing"] = dict(provider_routing)
    return factory(config_defaults=defaults)


def _usage_payload(usage: object) -> dict[str, Any] | None:
    if usage is None:
        return None
    to_dict = getattr(usage, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, dict):
            return payload
    return {"unserializable_usage_type": type(usage).__name__}


def _error_payload(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _result_path(output_dir: Path, index: int, model: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-") or "model"
    return output_dir / f"{index:02d}-{slug}.json"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON artifact, leaving the prior result intact on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(encoded)
    try:
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _judge_record(
    record: dict[str, Any],
    *,
    document: str,
    judge: Mapping[str, Any],
    judge_client: Any,
    clock: Callable[[], float],
) -> None:
    """Add a judge result to a successful generation record in place."""
    started = clock()
    try:
        candidate = json.dumps(record["generation"]["payload"], ensure_ascii=False, indent=2, sort_keys=True)
        judge_prompt = _render(judge["prompt_template"], document=document, candidate=candidate)
        judgment = judge_client.structured_output(
            [{"role": "user", "content": judge_prompt}],
            judge["response_schema"],
            temperature=judge["temperature"],
            max_tokens=judge["max_tokens"],
        )
        _validate_payload(judgment.value, judge["response_schema"], "judge payload")
        record["judge"] = {
            "model": judge["model"],
            "elapsed_seconds": clock() - started,
            "usage": _usage_payload(judgment.usage),
            "payload": judgment.value,
            "error": None,
        }
    except Exception as exc:
        record["judge"] = {
            "model": judge["model"],
            "elapsed_seconds": clock() - started,
            "usage": None,
            "payload": None,
            "error": _error_payload(exc),
        }


def run_bakeoff(config_path: Path, *, client_factory: ClientFactory = create_client, clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """Run each configured candidate and optionally a fixed judge; return the manifest."""
    config_path = config_path.resolve()
    config = load_config(config_path)
    source_path, document = read_source(config, config_path)
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_prompt = _render(config["prompt_template"], document=document)
    manifest: dict[str, Any] = {
        "kind": "structured_document_bakeoff/v1",
        "config_path": str(config_path),
        "source_path": str(source_path),
        "source_frontmatter_stripped": bool(config["source"].get("strip_yaml_frontmatter", False)),
        "models": [],
    }
    judge = config.get("judge")
    judge_client = None
    if judge is not None:
        judge_client = _client(client_factory, judge["model"], judge.get("provider_routing"))
        manifest["judge_model"] = judge["model"]

    for index, model_spec in enumerate(config["models"], start=1):
        model = model_spec["id"]
        record: dict[str, Any] = {"model": model, "generation": {}}
        started = clock()
        try:
            client = _client(client_factory, model, model_spec.get("provider_routing"))
            response = client.structured_output(
                [{"role": "user", "content": generation_prompt}],
                config["response_schema"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            )
            _validate_payload(response.value, config["response_schema"], "generation payload")
            record["generation"] = {
                "elapsed_seconds": clock() - started,
                "usage": _usage_payload(response.usage),
                "payload": response.value,
                "error": None,
            }
        except Exception as exc:
            record["generation"] = {
                "elapsed_seconds": clock() - started,
                "usage": None,
                "payload": None,
                "error": _error_payload(exc),
            }

        if judge is not None and record["generation"]["payload"] is not None:
            _judge_record(
                record,
                document=document,
                judge=judge,
                judge_client=judge_client,
                clock=clock,
            )

        output_path = _result_path(output_dir, index, model)
        _write_json(output_path, record)
        manifest["models"].append({"model": model, "result_path": str(output_path), "success": record["generation"]["error"] is None})

    manifest_path = output_dir / "manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    _write_json(manifest_path, manifest)
    return manifest


def judge_existing(
    config_path: Path,
    results_dir: Path,
    *,
    client_factory: ClientFactory = create_client,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, int | str]:
    """Retry missing or failed judge results without regenerating candidates.

    Existing successful judgments are retained, while records with a successful
    generation payload and no successful judge are updated in place.
    """
    config_path = config_path.resolve()
    config = load_config(config_path)
    judge = config.get("judge")
    if judge is None:
        raise ValueError("judge-existing requires a judge configuration")
    _, document = read_source(config, config_path)
    results_dir = results_dir.resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results directory not found: {results_dir}")

    counts: dict[str, int | str] = {
        "results_dir": str(results_dir),
        "files": 0,
        "attempted": 0,
        "judged": 0,
        "judge_failed": 0,
        "skipped_generation_failure": 0,
        "skipped_existing_judgment": 0,
        "invalid_records": 0,
    }
    judge_client = None
    for result_path in sorted(results_dir.glob("*.json")):
        if result_path.name == "manifest.json":
            continue
        counts["files"] += 1
        try:
            record = _mapping(json.loads(result_path.read_text(encoding="utf-8")), f"result {result_path}")
            generation = _mapping(record.get("generation"), f"result {result_path}.generation")
            record["generation"] = generation
        except (OSError, json.JSONDecodeError, ValueError):
            counts["invalid_records"] += 1
            continue
        if generation.get("payload") is None:
            counts["skipped_generation_failure"] += 1
            continue
        try:
            _validate_payload(generation["payload"], config["response_schema"], "generation payload")
        except ValueError as exc:
            generation["error"] = _error_payload(exc)
            counts["invalid_records"] += 1
            _write_json(result_path, record)
            continue
        prior_judge = record.get("judge")
        if isinstance(prior_judge, dict) and prior_judge.get("payload") is not None and prior_judge.get("error") is None:
            try:
                _validate_payload(prior_judge["payload"], judge["response_schema"], "judge payload")
            except ValueError:
                pass
            else:
                counts["skipped_existing_judgment"] += 1
                continue
        if judge_client is None:
            judge_client = _client(client_factory, judge["model"], judge.get("provider_routing"))
        counts["attempted"] += 1
        _judge_record(record, document=document, judge=judge, judge_client=judge_client, clock=clock)
        if record["judge"]["error"] is None:
            counts["judged"] += 1
        else:
            counts["judge_failed"] += 1
        _write_json(result_path, record)
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="YAML bakeoff configuration")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and source without model calls")
    parser.add_argument("--judge-existing", type=Path, metavar="RESULTS_DIR", help="Retry failed or missing judge calls from existing result JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        source_path, _ = read_source(config, args.config.resolve())
        if args.dry_run:
            if args.judge_existing is not None:
                raise ValueError("--dry-run cannot be combined with --judge-existing")
            print(json.dumps({"source_path": str(source_path), "models": [item["id"] for item in config["models"]], "judge": config.get("judge", {}).get("model")}, indent=2))
            return 0
        if args.judge_existing is not None:
            print(json.dumps(judge_existing(args.config, args.judge_existing), indent=2))
            return 0
        manifest = run_bakeoff(args.config)
        print(json.dumps(manifest, indent=2))
        return 0
    except Exception as exc:
        print(f"structured document bakeoff failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
