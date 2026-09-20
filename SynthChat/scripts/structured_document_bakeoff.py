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
No model call is made by ``--dry-run``.  Provider transport remains in
``shared.llm``; this runner owns config compilation, durable state, and local
schema admission.

For non-urgent asynchronous work, replace ``source`` with an explicit bounded
request list and use the three durable batch verbs::

    documents:
      - id: stable-document-id
        source_path: private/document.md
        strip_yaml_frontmatter: true
        metadata: {kind: optional-caller-data}
        expected_output: {metadata: {document_id: stable-document-id}}
    batch: {max_requests: 5}

    python -m SynthChat.scripts.structured_document_bakeoff --config batch.yaml --batch-submit
    python -m SynthChat.scripts.structured_document_bakeoff --config batch.yaml --batch-observe
    python -m SynthChat.scripts.structured_document_bakeoff --config batch.yaml --batch-collect
    python -m SynthChat.scripts.structured_document_bakeoff --config batch.yaml --batch-judge

``{metadata}`` is available to the prompt as canonical JSON.  Batch mode uses
the base model id and the real OpenRouter asynchronous Batch API; it does not
send a ``:batch`` model through synchronous chat completions.

When present, ``expected_output`` is a local admission contract.  The decoded
generation must contain every configured mapping key/value after passing the
shared JSON Schema.  Mappings match recursively; arrays and scalars match
exactly.  Booleans remain distinct from numbers, while integer and floating
representations compare as the same JSON number.  The contract is never sent
to the provider unless the caller also includes it in the prompt metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import jsonschema
import yaml

from shared.llm import create_client
from shared.llm.providers.openrouter import (
    OpenRouterBatchRejectedError,
    OpenRouterBatchSubmissionAmbiguousError,
)
from shared.llm.usage import usage_from_openai_block


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
        source = path.read_text(encoding="utf-8")
        _reject_expected_output_yaml_duplicates(source)
        loaded = yaml.safe_load(source)
    except FileNotFoundError:
        raise FileNotFoundError(f"Bakeoff config not found: {path}") from None
    config = _mapping(loaded, "bakeoff config")

    if "source" in config:
        source = _mapping(config.get("source"), "source")
        _text(source.get("path"), "source.path")
        if "strip_yaml_frontmatter" in source and not isinstance(source["strip_yaml_frontmatter"], bool):
            raise ValueError("source.strip_yaml_frontmatter must be a boolean")
        config["source"] = source
    if "documents" in config:
        if not isinstance(config["documents"], list) or not config["documents"]:
            raise ValueError("documents must be a non-empty list")
        config["documents"] = [_document_spec(item, index) for index, item in enumerate(config["documents"])]
        document_ids = [item["id"] for item in config["documents"]]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("documents ids must be unique")
    if "source" not in config and "documents" not in config:
        raise ValueError("bakeoff config requires source or documents")

    _template(config.get("prompt_template"), "prompt_template", "{document}")
    config["response_schema"] = _schema(config.get("response_schema"), "response_schema")
    if not isinstance(config.get("models"), list) or not config["models"]:
        raise ValueError("models must be a non-empty list")
    config["models"] = [_model_spec(item, index) for index, item in enumerate(config["models"])]
    model_ids = [item["id"] for item in config["models"]]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("models ids must be unique")
    config["temperature"] = _number(config.get("temperature"), "temperature")
    config["max_tokens"] = _positive_int(config.get("max_tokens"), "max_tokens")
    config["output_dir"] = _text(config.get("output_dir"), "output_dir")

    batch = config.get("batch")
    if batch is not None:
        batch = _mapping(batch, "batch")
        unknown = set(batch) - {"max_requests"}
        if unknown:
            raise ValueError(f"batch has unsupported fields: {', '.join(sorted(unknown))}")
        batch["max_requests"] = _positive_int(batch.get("max_requests"), "batch.max_requests")
        config["batch"] = batch

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


def _reject_expected_output_yaml_duplicates(source: str) -> None:
    """Reject duplicate keys within configured expected_output YAML mappings."""
    root = yaml.compose(source, Loader=yaml.SafeLoader)
    if not isinstance(root, yaml.MappingNode):
        return

    def scalar_key(node: yaml.Node) -> str | None:
        return node.value if isinstance(node, yaml.ScalarNode) else None

    visiting: set[int] = set()
    nodes_seen = 0

    def validate_contract(node: yaml.Node, depth: int = 0) -> None:
        nonlocal nodes_seen
        nodes_seen += 1
        identity = id(node)
        if nodes_seen > 10_000 or depth > 128 or identity in visiting:
            raise ValueError("expected_output YAML structure is cyclic or too large")
        visiting.add(identity)
        if isinstance(node, yaml.MappingNode):
            seen: set[tuple[str, str]] = set()
            for key_node, value_node in node.value:
                identity = (key_node.tag, key_node.value) if isinstance(key_node, yaml.ScalarNode) else None
                if identity is None or identity in seen:
                    raise ValueError("expected_output contains duplicate or non-scalar YAML keys")
                seen.add(identity)
                validate_contract(value_node, depth + 1)
        elif isinstance(node, yaml.SequenceNode):
            for child in node.value:
                validate_contract(child, depth + 1)
        visiting.remove(id(node))

    documents_node = next(
        (value for key, value in root.value if scalar_key(key) == "documents"),
        None,
    )
    if not isinstance(documents_node, yaml.SequenceNode):
        return
    for document_node in documents_node.value:
        if not isinstance(document_node, yaml.MappingNode):
            continue
        expected_nodes = [
            value for key, value in document_node.value if scalar_key(key) == "expected_output"
        ]
        if len(expected_nodes) > 1:
            raise ValueError("documents entry contains duplicate expected_output fields")
        if expected_nodes:
            validate_contract(expected_nodes[0])


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


def _json_exact_equal(actual: object, expected: object) -> bool:
    """Compare JSON values exactly, including scalar types and mapping keys."""
    if (
        type(actual) in (int, float)
        and type(expected) in (int, float)
    ):
        return actual == expected
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (
            actual.keys() == expected.keys()
            and all(_json_exact_equal(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_exact_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


def _safe_expected_path(parts: list[object]) -> str:
    """Render a diagnostic path without echoing provider/config-controlled data."""
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        elif (
            isinstance(part, str)
            and len(part) <= 64
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part)
        ):
            path += f".{part}"
        else:
            path += "[*]"
    return path


def _expected_output_mismatch_path(
    actual: object,
    expected: Mapping[str, Any],
    path: list[object] | None = None,
) -> list[object] | None:
    """Return the first path that violates partial deep equality."""
    current = [] if path is None else path
    if not isinstance(actual, dict):
        return current
    for key in sorted(expected):
        expected_value = expected[key]
        child_path = [*current, key]
        if key not in actual:
            return child_path
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            mismatch = _expected_output_mismatch_path(actual_value, expected_value, child_path)
            if mismatch is not None:
                return mismatch
        elif not _json_exact_equal(actual_value, expected_value):
            return child_path
    return None


def _validate_expected_output(payload: object, expected: Mapping[str, Any]) -> None:
    mismatch = _expected_output_mismatch_path(payload, expected)
    if mismatch is not None:
        location = _safe_expected_path(mismatch)
        raise ValueError(f"generation payload failed expected-output admission at {location}")


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


def _document_spec(value: object, index: int) -> dict[str, Any]:
    document = _mapping(value, f"documents[{index}]")
    allowed = {"id", "source_path", "strip_yaml_frontmatter", "metadata", "expected_output"}
    unknown = set(document) - allowed
    if unknown:
        raise ValueError(f"documents[{index}] has unsupported fields: {', '.join(sorted(unknown))}")
    document["id"] = _text(document.get("id"), f"documents[{index}].id")
    document["source_path"] = _text(document.get("source_path"), f"documents[{index}].source_path")
    if "strip_yaml_frontmatter" in document and not isinstance(document["strip_yaml_frontmatter"], bool):
        raise ValueError(f"documents[{index}].strip_yaml_frontmatter must be a boolean")
    if "metadata" in document:
        document["metadata"] = _mapping(document["metadata"], f"documents[{index}].metadata")
    else:
        document["metadata"] = {}
    try:
        json.dumps(document["metadata"], ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError(f"documents[{index}].metadata must be finite JSON data") from None
    if "expected_output" in document:
        document["expected_output"] = _mapping(
            document["expected_output"],
            f"documents[{index}].expected_output",
        )
        _require_expected_output_object(
            document["expected_output"],
            f"documents[{index}].expected_output",
        )
    return document


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
    if "source" not in config:
        raise ValueError("synchronous bakeoff requires source; documents is for batch mode")
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


def read_documents(config: Mapping[str, Any], config_path: Path) -> list[dict[str, Any]]:
    """Read explicitly configured batch documents without discovering files."""
    specs = config.get("documents")
    if not isinstance(specs, list) or not specs:
        raise ValueError("batch mode requires a non-empty documents list")
    documents: list[dict[str, Any]] = []
    for index, raw in enumerate(specs):
        spec = _mapping(raw, f"documents[{index}]")
        source_path = Path(_text(spec["source_path"], f"documents[{index}].source_path"))
        if not source_path.is_absolute():
            source_path = (config_path.parent / source_path).resolve()
        text = source_path.read_text(encoding="utf-8")
        if spec.get("strip_yaml_frontmatter", False):
            text = _FRONTMATTER.sub("", text, count=1)
        if not text.strip():
            raise ValueError(f"batch document is empty after preprocessing: {source_path}")
        documents.append(
            {
                "id": spec["id"],
                "source_path": source_path,
                "text": text,
                "metadata": dict(spec.get("metadata", {})),
                "frontmatter_stripped": bool(spec.get("strip_yaml_frontmatter", False)),
                **(
                    {"expected_output": dict(spec["expected_output"])}
                    if "expected_output" in spec
                    else {}
                ),
            }
        )
    return documents


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
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(encoded)
    try:
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json_immutable(path: Path, payload: Mapping[str, Any]) -> bool:
    """Create immutable canonical JSON; identical recollection is a no-op."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_bytes(payload)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise ValueError(f"cannot verify immutable artifact: {path}") from exc
            if existing == encoded:
                return False
            raise ValueError(f"immutable artifact already exists with different canonical bytes: {path}")
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return True


def _verify_immutable_compatible(path: Path, payload: Mapping[str, Any]) -> None:
    """Fail before collection writes when an existing immutable artifact differs."""
    encoded = _json_bytes(payload)
    if not path.exists():
        return
    try:
        existing = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot verify immutable artifact: {path}") from exc
    if existing != encoded:
        raise ValueError(f"immutable artifact already exists with different canonical bytes: {path}")


_BATCH_STATE_KIND = "structured_document_bakeoff/openrouter_batch_state/v1"
_BATCH_ENDPOINT = "/v1/chat/completions"
_BATCH_TERMINAL = {"completed", "failed", "cancelled", "expired"}
_BATCH_PROVIDER_STATUSES = {
    "validating", "in_progress", "finalizing", "completed", "failed",
    "cancelling", "cancelled", "expired",
}
_BATCH_SUBMISSION_STATUSES = {"pending", "submitting", "submitted", "rejected"}
_BATCH_JUDGMENT_KIND = "structured_document_bakeoff/batch_judgment/v1"
_BATCH_JUDGE_MANIFEST_KIND = "structured_document_bakeoff/batch_judge_manifest/v1"
_BATCH_TIMESTAMP_FIELDS = {
    "created_at", "in_progress_at", "finalizing_at", "completed_at",
    "failed_at", "cancelled_at", "expired_at",
}
_BATCH_COUNT_FIELDS = {"total", "completed", "failed"}


def _strict_json_loads(text: str, *, name: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON number: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{name} contains duplicate object keys")
            value[key] = item
        return value

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


def _require_finite_json(value: object, name: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain only finite JSON data") from None


def _require_expected_output_object(value: object, name: str) -> None:
    """Require strict JSON data rather than merely json.dumps-compatible data."""
    visiting: set[int] = set()
    nodes_seen = 0

    def is_json_data(item: object, depth: int = 0) -> bool:
        nonlocal nodes_seen
        nodes_seen += 1
        if nodes_seen > 10_000 or depth > 128:
            return False
        if item is None or isinstance(item, (str, bool)):
            return True
        if type(item) is int:
            return True
        if type(item) is float:
            try:
                json.dumps(item, allow_nan=False)
            except ValueError:
                return False
            return True
        if isinstance(item, list):
            identity = id(item)
            if identity in visiting:
                return False
            visiting.add(identity)
            valid = all(is_json_data(child, depth + 1) for child in item)
            visiting.remove(identity)
            return valid
        if isinstance(item, dict):
            identity = id(item)
            if identity in visiting:
                return False
            visiting.add(identity)
            valid = all(
                isinstance(key, str) and is_json_data(child, depth + 1)
                for key, child in item.items()
            )
            visiting.remove(identity)
            return valid
        return False

    if not isinstance(value, dict) or not is_json_data(value):
        raise ValueError(f"{name} must be a finite JSON object")


def _expected_output_digest(expected_output: Mapping[str, Any]) -> str:
    return _digest(
        {
            "kind": "structured_document_bakeoff/expected_output/v1",
            "expected_output": dict(expected_output),
        }
    )


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@contextmanager
def _exclusive_batch_claim(state_path: Path, spec_digest: str):
    """Hold an OS-wide O_EXCL claim; stale claims are never auto-stolen."""
    claim_path = state_path.with_name(f"{state_path.name}.lock")
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise RuntimeError(
            f"batch state is exclusively claimed; inspect the lock before manual recovery: {claim_path}"
        ) from None
    identity = os.fstat(descriptor)
    try:
        claim = _json_bytes(
            {
                "kind": "structured_document_bakeoff/openrouter_batch_claim/v1",
                "pid": os.getpid(),
                "spec_digest": spec_digest,
            }
        )
        os.write(descriptor, claim)
        os.fsync(descriptor)
        yield claim_path
    finally:
        os.close(descriptor)
        try:
            current = claim_path.stat()
        except FileNotFoundError:
            current = None
        if current is not None and (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
            claim_path.unlink()


@contextmanager
def _exclusive_batch_judge_claim(claim_path: Path, binding_digest: str):
    """Hold a judge-wide O_EXCL claim; existing claims require manual recovery."""
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise RuntimeError(
            f"batch judging is exclusively claimed; inspect the lock before manual recovery: {claim_path}"
        ) from None
    identity = os.fstat(descriptor)
    try:
        claim = _json_bytes(
            {
                "kind": "structured_document_bakeoff/batch_judge_claim/v1",
                "pid": os.getpid(),
                "binding_digest": binding_digest,
            }
        )
        os.write(descriptor, claim)
        os.fsync(descriptor)
        yield claim_path
    finally:
        os.close(descriptor)
        try:
            current = claim_path.stat()
        except FileNotFoundError:
            current = None
        if current is not None and (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
            claim_path.unlink()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(now: Callable[[], datetime]) -> str:
    value = now()
    if not isinstance(value, datetime):
        raise TypeError("batch clock must return datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _output_dir(config: Mapping[str, Any], config_path: Path) -> Path:
    output_dir = Path(_text(config["output_dir"], "output_dir"))
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    return output_dir


def _batch_state_path(config: Mapping[str, Any], config_path: Path, state_path: Path | None) -> Path:
    if state_path is not None:
        if not state_path.is_absolute():
            state_path = (config_path.parent / state_path).resolve()
        return state_path
    return _output_dir(config, config_path) / "openrouter-batch-state.json"


def _batch_custom_id(document_id: str, model: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", document_id).strip("._-")[:36] or "document"
    binding = hashlib.sha256(f"{document_id}\0{model}".encode("utf-8")).hexdigest()[:20]
    return f"{readable}-{binding}"


def _batch_result_path(output_dir: Path, document_id: str, model: str) -> Path:
    document_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", document_id).strip("._-")[:50] or "document"
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-")[:50] or "model"
    suffix = hashlib.sha256(f"{document_id}\0{model}".encode("utf-8")).hexdigest()[:10]
    return output_dir / f"{document_slug}--{model_slug}--{suffix}.json"


def _batch_request_body(
    *,
    model_spec: Mapping[str, Any],
    prompt: str,
    response_schema: Mapping[str, Any],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_spec["id"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": response_schema.get("name", "response"),
                "strict": True,
                "schema": dict(response_schema),
            },
        },
    }
    provider_routing = model_spec.get("provider_routing")
    if provider_routing is not None:
        body["provider"] = dict(provider_routing)
    return body


def _build_batch_plan(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    batch_config = config.get("batch")
    if not isinstance(batch_config, dict):
        raise ValueError("batch mode requires batch.max_requests")
    documents = read_documents(config, config_path)
    request_count = len(documents) * len(config["models"])
    if request_count > batch_config["max_requests"]:
        raise ValueError(
            f"batch request count {request_count} exceeds configured max_requests {batch_config['max_requests']}"
        )

    output_dir = _output_dir(config, config_path)
    batches: list[dict[str, Any]] = []
    custom_ids: set[str] = set()
    for model_spec in config["models"]:
        model = model_spec["id"]
        if model.endswith(":batch"):
            raise ValueError(
                f"batch model must use the base model id, not a synchronous :batch slug: {model}"
            )
        provider_items: list[dict[str, Any]] = []
        mappings: dict[str, dict[str, Any]] = {}
        expected_outputs: dict[str, dict[str, Any]] = {}
        for document in documents:
            custom_id = _batch_custom_id(document["id"], model)
            if custom_id in custom_ids:
                raise ValueError(f"generated duplicate batch custom_id: {custom_id}")
            custom_ids.add(custom_id)
            metadata_text = json.dumps(document["metadata"], ensure_ascii=False, sort_keys=True)
            prompt = _render(config["prompt_template"], document=document["text"], metadata=metadata_text)
            body = _batch_request_body(
                model_spec=model_spec,
                prompt=prompt,
                response_schema=config["response_schema"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            )
            result_path = _batch_result_path(output_dir, document["id"], model)
            provider_items.append({"custom_id": custom_id, "body": body})
            mappings[custom_id] = {
                "document_id": document["id"],
                "model": model,
                "source_path": str(document["source_path"]),
                "source_frontmatter_stripped": document["frontmatter_stripped"],
                "metadata": document["metadata"],
                "result_path": str(result_path),
                "request_digest": _digest(body),
                **(
                    {
                        "expected_output_digest": _expected_output_digest(
                            document["expected_output"]
                        )
                    }
                    if "expected_output" in document
                    else {}
                ),
            }
            if "expected_output" in document:
                expected_outputs[custom_id] = document["expected_output"]
        provider_spec = {"endpoint": _BATCH_ENDPOINT, "model": model, "requests": provider_items}
        digest_spec = provider_spec
        expected_output_digests = {
            custom_id: mapping["expected_output_digest"]
            for custom_id, mapping in mappings.items()
            if "expected_output_digest" in mapping
        }
        if expected_output_digests:
            digest_spec = {
                **provider_spec,
                "expected_output_digests": expected_output_digests,
            }
        batches.append(
            {
                "model": model,
                "provider_routing": model_spec.get("provider_routing"),
                "provider_spec": provider_spec,
                "digest_spec": digest_spec,
                "spec_digest": _digest(digest_spec),
                "custom_ids": mappings,
                "expected_outputs": expected_outputs,
            }
        )
    overall_spec = {
        "endpoint": _BATCH_ENDPOINT,
        "batches": [batch["digest_spec"] for batch in batches],
    }
    return config, batches, _digest(overall_spec)


def _new_batch_state(
    config_path: Path,
    batches: list[dict[str, Any]],
    spec_digest: str,
    *,
    now: Callable[[], datetime],
) -> dict[str, Any]:
    created_at = _timestamp(now)
    return {
        "kind": _BATCH_STATE_KIND,
        "config_path": str(config_path.resolve()),
        "endpoint": _BATCH_ENDPOINT,
        "spec_digest": spec_digest,
        "created_at": created_at,
        "updated_at": created_at,
        "batches": [
            {
                "model": batch["model"],
                "spec_digest": batch["spec_digest"],
                "submission_status": "pending",
                "provider_batch_id": None,
                "provider_status": None,
                "submitted_at": None,
                "observed_at": None,
                "provider_timestamps": {},
                "request_counts": None,
                "rejection_history": [],
                "custom_ids": batch["custom_ids"],
            }
            for batch in batches
        ],
    }


def _validate_batch_state(state: Mapping[str, Any]) -> None:
    allowed_top = {
        "kind", "config_path", "endpoint", "spec_digest", "created_at", "updated_at", "batches",
    }
    if set(state) != allowed_top:
        raise ValueError("batch state top-level fields are invalid")
    if state.get("kind") != _BATCH_STATE_KIND:
        raise ValueError("batch state kind is unsupported")
    if not isinstance(state.get("config_path"), str) or not state["config_path"]:
        raise ValueError("batch state config_path must be a non-empty string")
    if state.get("endpoint") != _BATCH_ENDPOINT:
        raise ValueError("batch state endpoint is unsupported")
    if not _is_digest(state.get("spec_digest")):
        raise ValueError("batch state spec_digest is invalid")
    for field in ("created_at", "updated_at"):
        if not isinstance(state.get(field), str) or not state[field]:
            raise ValueError(f"batch state {field} must be a non-empty string")
    batches = state.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("batch state batches must be a non-empty list")

    models: set[str] = set()
    for index, raw_batch in enumerate(batches):
        batch = _mapping(raw_batch, f"batch state batches[{index}]")
        allowed_batch = {
            "model", "spec_digest", "submission_status", "provider_batch_id", "provider_status",
            "submitted_at", "observed_at", "provider_timestamps", "request_counts", "custom_ids",
            "submission_attempted_at", "rejection_history", "ambiguity",
        }
        required_batch = {
            "model", "spec_digest", "submission_status", "provider_batch_id", "provider_status",
            "submitted_at", "observed_at", "provider_timestamps", "request_counts", "custom_ids",
            "rejection_history",
        }
        unknown_batch = set(batch) - allowed_batch
        if unknown_batch or not required_batch.issubset(batch):
            raise ValueError(
                f"batch state batches[{index}] fields are invalid"
            )
        model = batch.get("model")
        if not isinstance(model, str) or not model or model in models:
            raise ValueError(f"batch state batches[{index}].model is invalid or duplicated")
        models.add(model)
        if not _is_digest(batch.get("spec_digest")):
            raise ValueError(f"batch state batches[{index}].spec_digest is invalid")
        submission_status = batch.get("submission_status")
        if submission_status not in _BATCH_SUBMISSION_STATUSES:
            raise ValueError(f"batch state batches[{index}].submission_status is invalid")
        provider_batch_id = batch.get("provider_batch_id")
        if provider_batch_id is not None and (
            not isinstance(provider_batch_id, str)
            or re.fullmatch(r"[A-Za-z0-9._-]+", provider_batch_id) is None
        ):
            raise ValueError(f"batch state batches[{index}].provider_batch_id is invalid")
        submitted_at = batch.get("submitted_at")
        attempted_at = batch.get("submission_attempted_at")
        rejection_history = batch.get("rejection_history")
        if not isinstance(rejection_history, list):
            raise ValueError(f"batch state batches[{index}].rejection_history is invalid")
        for history_index, rejection in enumerate(rejection_history):
            _validate_closed_diagnostic(
                rejection,
                f"batch state batches[{index}].rejection_history[{history_index}]",
                rejected=True,
            )
        ambiguity = batch.get("ambiguity")
        if submission_status == "pending":
            if {"submission_attempted_at", "ambiguity"} & set(batch) or rejection_history:
                raise ValueError(f"pending batch state batches[{index}] has invalid optional fields")
            if provider_batch_id is not None or submitted_at is not None or attempted_at is not None:
                raise ValueError(f"pending batch state batches[{index}] has submission evidence")
            if ambiguity is not None:
                raise ValueError(f"pending batch state batches[{index}] has terminal diagnostics")
        elif submission_status == "submitting":
            if "submission_attempted_at" not in batch:
                raise ValueError(f"submitting batch state batches[{index}] has invalid optional fields")
            if provider_batch_id is not None or submitted_at is not None:
                raise ValueError(f"submitting batch state batches[{index}] has provider identity")
            if not isinstance(attempted_at, str) or not attempted_at:
                raise ValueError(f"submitting batch state batches[{index}] lacks attempted timestamp")
            if ambiguity is not None:
                _validate_closed_diagnostic(ambiguity, f"batch state batches[{index}].ambiguity")
        elif submission_status == "submitted":
            if "submission_attempted_at" not in batch or "ambiguity" in batch:
                raise ValueError(f"submitted batch state batches[{index}] has invalid optional fields")
            if not isinstance(provider_batch_id, str) or not provider_batch_id:
                raise ValueError(f"submitted batch state batches[{index}] lacks provider id")
            if not isinstance(submitted_at, str) or not submitted_at:
                raise ValueError(f"submitted batch state batches[{index}] lacks submitted timestamp")
            if not isinstance(attempted_at, str) or not attempted_at:
                raise ValueError(f"submitted batch state batches[{index}] lacks attempted timestamp")
            if ambiguity is not None:
                raise ValueError(f"submitted batch state batches[{index}] has incompatible diagnostics")
        else:  # rejected
            if "submission_attempted_at" not in batch or not rejection_history or "ambiguity" in batch:
                raise ValueError(f"rejected batch state batches[{index}] has invalid optional fields")
            if provider_batch_id is not None or submitted_at is not None:
                raise ValueError(f"rejected batch state batches[{index}] has provider identity")
            if not isinstance(attempted_at, str) or not attempted_at:
                raise ValueError(f"rejected batch state batches[{index}] lacks attempted timestamp")

        provider_status = batch.get("provider_status")
        if provider_status is not None and provider_status not in _BATCH_PROVIDER_STATUSES:
            raise ValueError(f"batch state batches[{index}].provider_status is invalid")
        if submission_status != "submitted" and provider_status is not None:
            raise ValueError(f"batch state batches[{index}] has provider status without provider id")
        observed_at = batch.get("observed_at")
        if observed_at is not None and (not isinstance(observed_at, str) or not observed_at):
            raise ValueError(f"batch state batches[{index}].observed_at is invalid")
        if submission_status != "submitted" and observed_at is not None:
            raise ValueError(f"batch state batches[{index}] was observed before submission")
        if observed_at is not None and provider_status is None:
            raise ValueError(f"batch state batches[{index}] has observation time without provider status")

        timestamps = batch.get("provider_timestamps")
        if not isinstance(timestamps, dict) or set(timestamps) - _BATCH_TIMESTAMP_FIELDS:
            raise ValueError(f"batch state batches[{index}].provider_timestamps is invalid")
        if not all(
            value is None or type(value) is int or isinstance(value, str)
            for value in timestamps.values()
        ):
            raise ValueError(f"batch state batches[{index}].provider_timestamps values are invalid")
        _require_finite_json(timestamps, f"batch state batches[{index}].provider_timestamps")
        counts = batch.get("request_counts")
        if counts is not None:
            if (
                not isinstance(counts, dict)
                or set(counts) - _BATCH_COUNT_FIELDS
                or not all(
                isinstance(key, str) and type(value) is int and value >= 0 for key, value in counts.items()
                )
            ):
                raise ValueError(f"batch state batches[{index}].request_counts is invalid")
            total = counts.get("total")
            if total is not None and counts.get("completed", 0) + counts.get("failed", 0) > total:
                raise ValueError(f"batch state batches[{index}].request_counts is inconsistent")
        if submission_status != "submitted" and (timestamps or counts is not None):
            raise ValueError(f"batch state batches[{index}] has provider evidence before submission")

        mappings = batch.get("custom_ids")
        if not isinstance(mappings, dict) or not mappings:
            raise ValueError(f"batch state batches[{index}].custom_ids must be a non-empty mapping")
        for custom_id, raw_mapping in mappings.items():
            if not isinstance(custom_id, str) or not custom_id:
                raise ValueError(f"batch state batches[{index}] has invalid custom_id")
            mapping = _mapping(raw_mapping, f"batch state custom_id {custom_id}")
            allowed_mapping = {
                "document_id", "model", "source_path", "source_frontmatter_stripped",
                "metadata", "result_path", "request_digest", "expected_output_digest",
            }
            required_mapping = allowed_mapping - {"expected_output_digest"}
            if set(mapping) - allowed_mapping or not required_mapping.issubset(mapping):
                raise ValueError(f"batch state custom_id {custom_id} mapping fields are invalid")
            for field in ("document_id", "model", "source_path", "result_path"):
                if not isinstance(mapping.get(field), str) or not mapping[field]:
                    raise ValueError(f"batch state custom_id {custom_id}.{field} is invalid")
            if mapping["model"] != model or _batch_custom_id(mapping["document_id"], model) != custom_id:
                raise ValueError(f"batch state custom_id {custom_id} binding is invalid")
            if not isinstance(mapping.get("source_frontmatter_stripped"), bool):
                raise ValueError(f"batch state custom_id {custom_id} frontmatter flag is invalid")
            if not isinstance(mapping.get("metadata"), dict):
                raise ValueError(f"batch state custom_id {custom_id} metadata is invalid")
            _require_finite_json(mapping["metadata"], f"batch state custom_id {custom_id} metadata")
            if "expected_output_digest" in mapping and not _is_digest(
                mapping["expected_output_digest"]
            ):
                raise ValueError(
                    f"batch state custom_id {custom_id} expected_output_digest is invalid"
                )
            if not _is_digest(mapping.get("request_digest")):
                raise ValueError(f"batch state custom_id {custom_id} request_digest is invalid")
        if counts is not None and counts.get("total") is not None and counts["total"] != len(mappings):
            raise ValueError(f"batch state batches[{index}].request_counts total is inconsistent")
    _require_finite_json(state, "batch state")


def _validate_closed_diagnostic(value: object, name: str, *, rejected: bool = False) -> None:
    diagnostic = _mapping(value, name)
    allowed = {"type", "code", "status_code", "rejected_at"} if rejected else {"type", "code"}
    if set(diagnostic) - allowed or not isinstance(diagnostic.get("type"), str):
        raise ValueError(f"{name} is invalid")
    if "code" in diagnostic and diagnostic["code"] is not None and _safe_provider_identifier(diagnostic["code"]) is None:
        raise ValueError(f"{name}.code is invalid")
    if rejected and (type(diagnostic.get("status_code")) is not int or not 400 <= diagnostic["status_code"] < 500):
        raise ValueError(f"{name}.status_code is invalid")
    if rejected and (not isinstance(diagnostic.get("rejected_at"), str) or not diagnostic["rejected_at"]):
        raise ValueError(f"{name}.rejected_at is invalid")


def _write_batch_state(path: Path, state: Mapping[str, Any]) -> None:
    _validate_batch_state(state)
    _write_json(path, state)


def _load_batch_state(path: Path, *, spec_digest: str) -> dict[str, Any]:
    try:
        state = _mapping(_strict_json_loads(path.read_text(encoding="utf-8"), name="batch state"), "batch state")
    except FileNotFoundError:
        raise FileNotFoundError(f"batch state not found: {path}") from None
    except json.JSONDecodeError:
        raise ValueError(f"batch state is not valid JSON: {path}") from None
    _validate_batch_state(state)
    if state.get("spec_digest") != spec_digest:
        raise ValueError("batch state spec digest does not match the current rendered requests")
    return state


def _provider_batch_fields(observation: Mapping[str, Any]) -> dict[str, Any]:
    _require_finite_json(observation, "OpenRouter batch observation")
    status = observation.get("status")
    if status not in _BATCH_PROVIDER_STATUSES:
        raise ValueError("OpenRouter batch observation status is invalid")
    timestamps = {
        key: observation[key]
        for key in ("created_at", "in_progress_at", "finalizing_at", "completed_at", "failed_at", "cancelled_at", "expired_at")
        if key in observation
    }
    counts = observation.get("request_counts")
    if counts is not None and (
        not isinstance(counts, dict)
        or bool(set(counts) - _BATCH_COUNT_FIELDS)
        or not all(isinstance(key, str) and type(value) is int and value >= 0 for key, value in counts.items())
    ):
        raise ValueError("OpenRouter batch observation request_counts is invalid")
    return {
        "provider_status": status,
        "provider_timestamps": timestamps,
        "request_counts": counts if isinstance(counts, dict) else None,
    }


def _validate_state_plan(
    state: Mapping[str, Any],
    plans: list[dict[str, Any]],
    *,
    config_path: Path,
) -> None:
    if state["config_path"] != str(config_path.resolve()):
        raise ValueError("batch state config_path does not match the current config")
    if len(state["batches"]) != len(plans):
        raise ValueError("batch state model count does not match the current plan")
    for batch_state, plan in zip(state["batches"], plans):
        if batch_state.get("model") != plan["model"] or batch_state.get("spec_digest") != plan["spec_digest"]:
            raise ValueError("batch state model plan does not match the current plan")
        if batch_state.get("custom_ids") != plan["custom_ids"]:
            raise ValueError("batch state request mapping does not match the current plan")


def _persist_ambiguity(
    path: Path,
    state: dict[str, Any],
    batch_state: dict[str, Any],
    exc: Exception,
    *,
    now: Callable[[], datetime],
) -> None:
    batch_state["ambiguity"] = {"type": type(exc).__name__, "code": None}
    state["updated_at"] = _timestamp(now)
    try:
        _write_batch_state(path, state)
    except Exception:
        # The already-durable ``submitting`` marker remains the authority if
        # annotating it fails.  Never replace the original ambiguous outcome.
        pass


def submit_openrouter_batch(
    config_path: Path,
    *,
    state_path: Path | None = None,
    client_factory: ClientFactory = create_client,
    now: Callable[[], datetime] = _utc_now,
) -> dict[str, Any]:
    """Durably submit configured document/model requests to OpenRouter batches."""
    config_path = config_path.resolve()
    config, plans, spec_digest = _build_batch_plan(config_path)
    resolved_state_path = _batch_state_path(config, config_path, state_path)
    with _exclusive_batch_claim(resolved_state_path, spec_digest):
        if resolved_state_path.exists():
            state = _load_batch_state(resolved_state_path, spec_digest=spec_digest)
        else:
            state = _new_batch_state(config_path, plans, spec_digest, now=now)
            _write_batch_state(resolved_state_path, state)
        _validate_state_plan(state, plans, config_path=config_path)

        pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for batch_state, plan in zip(state["batches"], plans):
            status = batch_state["submission_status"]
            if status == "submitted":
                continue
            if status == "submitting":
                raise RuntimeError(
                    f"batch submission for {plan['model']} has an ambiguous prior outcome; reconcile it manually before retrying"
                )
            # Re-entering this explicit submit command authorizes a retry only
            # for a provider-confirmed rejection under this exact spec digest.
            pending.append((batch_state, plan))

        # Construction and capability validation happen before any durable
        # ``submitting`` transition.  A local setup failure remains retryable.
        prepared: list[tuple[dict[str, Any], dict[str, Any], Callable[..., Any]]] = []
        for batch_state, plan in pending:
            client = _client(client_factory, plan["model"], plan.get("provider_routing"))
            submit = getattr(client, "submit_batch", None)
            if not callable(submit):
                raise TypeError("configured client does not support OpenRouter batch submission")
            prepared.append((batch_state, plan, submit))

        for batch_state, plan, submit in prepared:
            attempted_at = _timestamp(now)
            batch_state["submission_status"] = "submitting"
            batch_state["submission_attempted_at"] = attempted_at
            state["updated_at"] = attempted_at
            _write_batch_state(resolved_state_path, state)

            try:
                observation = submit(plan["provider_spec"]["requests"], endpoint=_BATCH_ENDPOINT)
            except OpenRouterBatchRejectedError as exc:
                batch_state["submission_status"] = "rejected"
                batch_state["rejection_history"].append(
                    {
                        "type": type(exc).__name__,
                        "status_code": exc.status_code,
                        "code": exc.code,
                        "rejected_at": _timestamp(now),
                    }
                )
                state["updated_at"] = _timestamp(now)
                _write_batch_state(resolved_state_path, state)
                raise
            except Exception as exc:
                _persist_ambiguity(resolved_state_path, state, batch_state, exc, now=now)
                raise

            try:
                observation = _mapping(observation, "OpenRouter batch submission response")
                _require_finite_json(observation, "OpenRouter batch submission response")
                provider_batch_id = observation.get("id")
                if not isinstance(provider_batch_id, str) or not provider_batch_id:
                    raise OpenRouterBatchSubmissionAmbiguousError(
                        "OpenRouter batch submission returned no provider batch id"
                    )
                fields = _provider_batch_fields(observation)
            except Exception as exc:
                ambiguous = exc if isinstance(exc, OpenRouterBatchSubmissionAmbiguousError) else OpenRouterBatchSubmissionAmbiguousError(
                    "OpenRouter batch submission returned an invalid success object"
                )
                _persist_ambiguity(resolved_state_path, state, batch_state, ambiguous, now=now)
                raise ambiguous from exc

            batch_state["provider_batch_id"] = provider_batch_id
            batch_state["submission_status"] = "submitted"
            batch_state["submitted_at"] = _timestamp(now)
            batch_state.update(fields)
            state["updated_at"] = batch_state["submitted_at"]
            # If this replace fails after POST, the prior durable marker stays
            # ``submitting`` and therefore prevents a blind retry.
            _write_batch_state(resolved_state_path, state)
    state["state_path"] = str(resolved_state_path)
    return state


def _observe_openrouter_batches(
    config_path: Path,
    *,
    state_path: Path | None,
    client_factory: ClientFactory,
    now: Callable[[], datetime],
) -> tuple[dict[str, Any], Path, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    config_path = config_path.resolve()
    config, plans, spec_digest = _build_batch_plan(config_path)
    resolved_state_path = _batch_state_path(config, config_path, state_path)
    observations: dict[str, dict[str, Any]] = {}
    with _exclusive_batch_claim(resolved_state_path, spec_digest):
        state = _load_batch_state(resolved_state_path, spec_digest=spec_digest)
        _validate_state_plan(state, plans, config_path=config_path)
        prepared: list[tuple[dict[str, Any], dict[str, Any], str, Callable[..., Any]]] = []
        for batch_state, plan in zip(state["batches"], plans):
            provider_batch_id = batch_state.get("provider_batch_id")
            if batch_state.get("submission_status") != "submitted" or not isinstance(provider_batch_id, str):
                raise RuntimeError(f"batch for {plan['model']} has no safely persisted provider id")
            client = _client(client_factory, plan["model"], plan.get("provider_routing"))
            observe = getattr(client, "observe_batch", None)
            if not callable(observe):
                raise TypeError("configured client does not support OpenRouter batch observation")
            prepared.append((batch_state, plan, provider_batch_id, observe))
        for batch_state, plan, provider_batch_id, observe in prepared:
            observation = _mapping(observe(provider_batch_id), "OpenRouter batch observation")
            _require_finite_json(observation, "OpenRouter batch observation")
            if observation.get("id") != provider_batch_id:
                raise ValueError(f"OpenRouter batch observation id changed for {plan['model']}")
            observed_at = _timestamp(now)
            batch_state.update(_provider_batch_fields(observation))
            batch_state["observed_at"] = observed_at
            state["updated_at"] = observed_at
            _write_batch_state(resolved_state_path, state)
            observations[provider_batch_id] = observation
    state["state_path"] = str(resolved_state_path)
    return state, resolved_state_path, plans, observations


def observe_openrouter_batch(
    config_path: Path,
    *,
    state_path: Path | None = None,
    client_factory: ClientFactory = create_client,
    now: Callable[[], datetime] = _utc_now,
) -> dict[str, Any]:
    """Refresh and persist provider status for every submitted model batch."""
    state, _, _, _ = _observe_openrouter_batches(
        config_path,
        state_path=state_path,
        client_factory=client_factory,
        now=now,
    )
    return state


def _reconcile_batch_results(
    results: object,
    expected: Mapping[str, Any],
    *,
    model: str,
) -> dict[str, dict[str, Any]]:
    _require_finite_json(results, f"OpenRouter batch results for {model}")
    if not isinstance(results, list):
        raise ValueError(f"completed OpenRouter batch for {model} omitted results")
    indexed: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(f"OpenRouter batch result {index} for {model} is not a mapping")
        custom_id = item.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError(f"OpenRouter batch result {index} for {model} omitted custom_id")
        if custom_id in indexed:
            raise ValueError(f"OpenRouter batch for {model} returned duplicate custom_id: {custom_id}")
        indexed[custom_id] = item
    expected_ids = set(expected)
    actual_ids = set(indexed)
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    if missing or unexpected:
        raise ValueError(
            f"OpenRouter batch result ids did not reconcile for {model}: missing={missing}, unexpected={unexpected}"
        )
    return indexed


def _closed_provider_error(value: object) -> dict[str, Any]:
    """Retain machine-actionable fields without provider-controlled messages."""
    _require_finite_json(value, "OpenRouter provider error")
    if not isinstance(value, dict):
        return {"type": "provider_error", "code": None, "status_code": None}
    code = value.get("code")
    status_code = value.get("status_code")
    error_type = value.get("type")
    return {
        "type": _safe_provider_identifier(error_type) or "provider_error",
        "code": _safe_provider_identifier(code),
        "status_code": status_code if type(status_code) is int else None,
    }


def _safe_provider_identifier(value: object) -> str | None:
    if not isinstance(value, (str, int)):
        return None
    normalized = str(value)
    if len(normalized) > 64 or re.fullmatch(r"[A-Za-z0-9._-]+", normalized) is None:
        return None
    return normalized


def _batch_generation_record(
    item: Mapping[str, Any],
    *,
    response_schema: Mapping[str, Any],
    expected_output: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_finite_json(item, "OpenRouter batch result item")
    provider_error = item.get("error")
    if provider_error is not None:
        return {
            "usage": None,
            "payload": None,
            "error": {"type": "OpenRouterBatchItemError", "message": "provider returned an item-level error"},
            "provider_error": _closed_provider_error(provider_error),
        }
    response = item.get("response")
    if not isinstance(response, dict):
        raise ValueError("OpenRouter batch result has neither response nor error")
    status_code = response.get("status_code")
    if type(status_code) is not int or not 200 <= status_code < 300:
        return {
            "usage": None,
            "payload": None,
            "error": {
                "type": "OpenRouterBatchItemHTTPError",
                "message": f"provider item returned HTTP {status_code}",
            },
            "provider_error": _closed_provider_error(response.get("body")),
        }
    body = _mapping(response.get("body"), "OpenRouter batch response body")
    usage = usage_from_openai_block(body.get("usage"))
    try:
        choices = body["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("choices must be a non-empty list")
        message = _mapping(_mapping(choices[0], "choice").get("message"), "choice.message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("structured response content must be a non-empty string")
        payload = _strict_json_loads(content, name="structured response content")
        if not isinstance(payload, dict):
            raise ValueError("structured response content must decode to an object")
        _validate_payload(payload, response_schema, "generation payload")
        if expected_output is not None:
            _validate_expected_output(payload, expected_output)
    except (KeyError, json.JSONDecodeError, ValueError) as exc:
        return {
            "usage": _usage_payload(usage),
            "payload": None,
            "error": _error_payload(exc),
            "provider_error": None,
        }
    return {
        "usage": _usage_payload(usage),
        "payload": payload,
        "error": None,
        "provider_error": None,
    }


def collect_openrouter_batch(
    config_path: Path,
    *,
    state_path: Path | None = None,
    client_factory: ClientFactory = create_client,
    now: Callable[[], datetime] = _utc_now,
) -> dict[str, Any]:
    """Collect completed batches, reconcile IDs, and locally admit results."""
    config_path = config_path.resolve()
    config = load_config(config_path)
    state, resolved_state_path, plans, observations = _observe_openrouter_batches(
        config_path,
        state_path=state_path,
        client_factory=client_factory,
        now=now,
    )
    summary: dict[str, Any] = {
        "state_path": str(resolved_state_path),
        "ready": True,
        "pending_batches": [],
        "terminal_failures": [],
        "written": 0,
        "succeeded": 0,
        "item_failed": 0,
        "results": [],
    }
    for batch_state, plan in zip(state["batches"], plans):
        provider_batch_id = batch_state["provider_batch_id"]
        status = observations[provider_batch_id].get("status")
        if status not in _BATCH_TERMINAL:
            summary["ready"] = False
            summary["pending_batches"].append(
                {"model": plan["model"], "provider_batch_id": provider_batch_id, "status": status}
            )
        elif status != "completed":
            summary["ready"] = False
            summary["terminal_failures"].append(
                {"model": plan["model"], "provider_batch_id": provider_batch_id, "status": status}
            )
    if summary["pending_batches"]:
        # A nonterminal poll is observation only.  Reserving any canonical
        # final path here would make the later completed collection conflict
        # with immutable-artifact rules.
        return summary

    output_dir = _output_dir(config, config_path)
    artifacts: list[tuple[Path, Mapping[str, Any]]] = []
    for batch_state, plan in zip(state["batches"], plans):
        provider_batch_id = batch_state["provider_batch_id"]
        observation = observations[provider_batch_id]
        status = observation.get("status")
        if status != "completed":
            continue
        reconciled = _reconcile_batch_results(
            observation.get("results"),
            batch_state["custom_ids"],
            model=plan["model"],
        )
        for custom_id, mapping in batch_state["custom_ids"].items():
            generation = _batch_generation_record(
                reconciled[custom_id],
                response_schema=config["response_schema"],
                expected_output=plan["expected_outputs"].get(custom_id),
            )
            record = {
                "model": mapping["model"],
                "document_id": mapping["document_id"],
                "source_path": mapping["source_path"],
                "source_frontmatter_stripped": mapping["source_frontmatter_stripped"],
                "document_metadata": mapping["metadata"],
                "provider_batch_id": provider_batch_id,
                "custom_id": custom_id,
                "request_digest": mapping["request_digest"],
                "generation": generation,
                **(
                    {"expected_output_digest": mapping["expected_output_digest"]}
                    if "expected_output_digest" in mapping
                    else {}
                ),
            }
            result_path = Path(mapping["result_path"])
            artifacts.append((result_path, record))
            success = generation["error"] is None
            summary["written"] += 1
            summary["succeeded" if success else "item_failed"] += 1
            summary["results"].append(
                {
                    "custom_id": custom_id,
                    "document_id": mapping["document_id"],
                    "model": mapping["model"],
                    "result_path": str(result_path),
                    "success": success,
                }
            )
    manifest_path = output_dir / "batch-manifest.json"
    summary["manifest_path"] = str(manifest_path)
    artifacts.append(
        (
            manifest_path,
            {"kind": "structured_document_bakeoff/openrouter_batch_manifest/v1", **summary},
        )
    )
    for artifact_path, payload in artifacts:
        _verify_immutable_compatible(artifact_path, payload)
    for artifact_path, payload in artifacts:
        _write_json_immutable(artifact_path, payload)
    return summary


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_canonical_json_object(path: Path, name: str) -> tuple[dict[str, Any], bytes]:
    try:
        encoded = path.read_bytes()
    except FileNotFoundError:
        raise FileNotFoundError(f"{name} not found: {path}") from None
    try:
        value = _strict_json_loads(encoded.decode("utf-8"), name=name)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError(f"{name} is not canonical finite JSON: {path}") from None
    payload = _mapping(value, name)
    _require_finite_json(payload, name)
    if encoded != _json_bytes(payload):
        raise ValueError(f"{name} does not use canonical JSON bytes: {path}")
    return payload, encoded


def _batch_judgment_path(output_dir: Path, result_path: Path) -> Path:
    return output_dir / "judgments" / f"{result_path.stem}.judgment.json"


def _batch_judge_binding(
    *,
    document_id: str,
    generation_model: str,
    generation_result_bytes: bytes,
    document: str,
    judge: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "generation_model": generation_model,
        "generation_result_sha256": _sha256_bytes(generation_result_bytes),
        "source_sha256": _sha256_bytes(document.encode("utf-8")),
        "judge_model": judge["model"],
        "judge_config_sha256": _digest(dict(judge)),
        "judge_prompt_template_sha256": _sha256_bytes(judge["prompt_template"].encode("utf-8")),
        "judge_response_schema_sha256": _digest(judge["response_schema"]),
    }


def _closed_judge_failure(exc: Exception, *, stage: str) -> dict[str, str]:
    error_type = type(exc).__name__
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", error_type) is None:
        error_type = "Exception"
    return {"stage": stage, "type": error_type}


def _load_existing_batch_judgment(
    path: Path,
    *,
    binding: Mapping[str, Any],
    judge: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    artifact, encoded = _read_canonical_json_object(path, "batch judgment artifact")
    if set(artifact) != {"kind", "binding", "judgment"} or artifact.get("kind") != _BATCH_JUDGMENT_KIND:
        raise ValueError(f"batch judgment artifact fields are invalid: {path}")
    if artifact.get("binding") != dict(binding):
        raise ValueError(
            f"batch judgment binding conflicts with the current generation, source, or judge config: {path}"
        )
    judgment = _mapping(artifact.get("judgment"), "batch judgment")
    if set(judgment) != {"model", "usage", "payload"} or judgment.get("model") != judge["model"]:
        raise ValueError(f"batch judgment payload fields are invalid: {path}")
    _require_finite_json(judgment, "batch judgment")
    try:
        _validate_payload(judgment.get("payload"), judge["response_schema"], "judge payload")
    except ValueError:
        raise ValueError(f"batch judgment payload does not pass the configured schema: {path}") from None
    return artifact, encoded


def _validate_batch_collection_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "kind", "state_path", "ready", "pending_batches", "terminal_failures",
        "written", "succeeded", "item_failed", "results", "manifest_path",
    }
    if set(manifest) != required:
        raise ValueError("batch collection manifest top-level fields are invalid")
    if manifest.get("kind") != "structured_document_bakeoff/openrouter_batch_manifest/v1":
        raise ValueError("batch collection manifest kind is unsupported")
    for field in ("state_path", "manifest_path"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"batch collection manifest {field} is invalid")
    if type(manifest.get("ready")) is not bool:
        raise ValueError("batch collection manifest ready must be a boolean")
    for field in ("pending_batches", "terminal_failures", "results"):
        if not isinstance(manifest.get(field), list):
            raise ValueError(f"batch collection manifest {field} must be a list")
    for field in ("written", "succeeded", "item_failed"):
        if type(manifest.get(field)) is not int or manifest[field] < 0:
            raise ValueError(f"batch collection manifest {field} must be a nonnegative integer")

    result_fields = {"custom_id", "document_id", "model", "result_path", "success"}
    succeeded = 0
    failed = 0
    for index, raw in enumerate(manifest["results"]):
        item = _mapping(raw, f"batch collection manifest results[{index}]")
        if set(item) != result_fields:
            raise ValueError("batch collection manifest result fields are invalid")
        for field in ("custom_id", "document_id", "model", "result_path"):
            if not isinstance(item.get(field), str) or not item[field]:
                raise ValueError(f"batch collection manifest result {field} is invalid")
        if type(item.get("success")) is not bool:
            raise ValueError("batch collection manifest result success must be a boolean")
        if item["success"]:
            succeeded += 1
        else:
            failed += 1
    if (
        manifest["written"] != len(manifest["results"])
        or manifest["succeeded"] != succeeded
        or manifest["item_failed"] != failed
        or manifest["succeeded"] + manifest["item_failed"] != manifest["written"]
    ):
        raise ValueError("batch collection manifest counts are inconsistent")


def _load_batch_judge_rows(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], bytes, list[dict[str, Any]]]:
    """Load and reconcile a completed collection against the current config."""
    config, plans, _ = _build_batch_plan(config_path)
    judge = config.get("judge")
    if not isinstance(judge, dict):
        raise ValueError("batch-judge requires a judge configuration")
    output_dir = _output_dir(config, config_path)
    manifest_path = output_dir / "batch-manifest.json"
    manifest, manifest_bytes = _read_canonical_json_object(manifest_path, "batch collection manifest")
    _validate_batch_collection_manifest(manifest)
    if (
        manifest.get("ready") is not True
        or manifest.get("pending_batches") != []
        or manifest.get("terminal_failures") != []
    ):
        raise ValueError("batch collection manifest is not complete")
    manifest_results = manifest["results"]

    documents = {item["id"]: item for item in read_documents(config, config_path)}
    model_specs = {item["id"]: item for item in config["models"]}
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    expected_contracts: dict[tuple[str, str], dict[str, Any]] = {}
    for plan in plans:
        for custom_id, mapping in plan["custom_ids"].items():
            key = (mapping["document_id"], mapping["model"])
            expected[key] = {"custom_id": custom_id, **mapping}
            if custom_id in plan["expected_outputs"]:
                expected_contracts[key] = plan["expected_outputs"][custom_id]

    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw in enumerate(manifest_results):
        item = _mapping(raw, f"batch collection manifest results[{index}]")
        document_id = item.get("document_id")
        model = item.get("model")
        if not isinstance(document_id, str) or not isinstance(model, str):
            raise ValueError("batch collection manifest result identity is invalid")
        key = (document_id, model)
        if key in indexed:
            raise ValueError("batch collection manifest contains a duplicate result identity")
        indexed[key] = item
    if set(indexed) != set(expected):
        raise ValueError("batch collection manifest does not match the configured document/model set")
    if manifest.get("manifest_path") != str(manifest_path):
        raise ValueError("batch collection manifest path binding is invalid")

    rows: list[dict[str, Any]] = []
    for key, expected_item in expected.items():
        item = indexed[key]
        result_path = Path(expected_item["result_path"])
        document = documents[expected_item["document_id"]]
        current_prompt = _render(
            config["prompt_template"],
            document=document["text"],
            metadata=json.dumps(document["metadata"], ensure_ascii=False, sort_keys=True),
        )
        current_request_digest = _digest(
            _batch_request_body(
                model_spec=model_specs[expected_item["model"]],
                prompt=current_prompt,
                response_schema=config["response_schema"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            )
        )
        if current_request_digest != expected_item["request_digest"]:
            raise ValueError("configured batch source changed while reconciling the collection")
        if (
            item.get("custom_id") != expected_item["custom_id"]
            or item.get("result_path") != str(result_path)
            or type(item.get("success")) is not bool
        ):
            raise ValueError("batch collection manifest result binding is invalid")
        record, result_bytes = _read_canonical_json_object(result_path, "batch generation result")
        required_record = {
            "model", "document_id", "source_path", "source_frontmatter_stripped",
            "document_metadata", "provider_batch_id", "custom_id", "request_digest", "generation",
        }
        allowed_record = required_record | {"expected_output_digest"}
        if set(record) - allowed_record or not required_record.issubset(record):
            raise ValueError(f"batch generation result fields are invalid: {result_path}")
        expected_contract_present = "expected_output_digest" in expected_item
        if (
            ("expected_output_digest" in record) is not expected_contract_present
            or (
                expected_contract_present
                and record.get("expected_output_digest")
                != expected_item["expected_output_digest"]
            )
        ):
            raise ValueError(f"batch generation result expected-output binding is invalid: {result_path}")
        if (
            record.get("model") != expected_item["model"]
            or record.get("document_id") != expected_item["document_id"]
            or record.get("source_path") != expected_item["source_path"]
            or record.get("source_frontmatter_stripped") != expected_item["source_frontmatter_stripped"]
            or record.get("document_metadata") != expected_item["metadata"]
            or record.get("custom_id") != expected_item["custom_id"]
            or record.get("request_digest") != expected_item["request_digest"]
        ):
            raise ValueError(f"batch generation result binding is invalid: {result_path}")
        generation = _mapping(record.get("generation"), "batch generation result generation")
        if set(generation) != {"usage", "payload", "error", "provider_error"}:
            raise ValueError(f"batch generation result generation fields are invalid: {result_path}")
        successful = generation.get("payload") is not None and generation.get("error") is None
        if successful != item["success"]:
            raise ValueError(f"batch generation result success binding is invalid: {result_path}")
        if successful:
            _require_finite_json(generation["payload"], "generation payload")
            try:
                _validate_payload(generation["payload"], config["response_schema"], "generation payload")
                if expected_contract_present:
                    _validate_expected_output(
                        generation["payload"],
                        expected_contracts[key],
                    )
            except ValueError:
                raise ValueError(
                    f"batch generation payload does not pass configured admission: {result_path}"
                ) from None
        elif generation.get("payload") is not None or generation.get("error") is None:
            raise ValueError(f"failed batch generation result is inconsistent: {result_path}")
        rows.append(
            {
                "document": document["text"],
                "document_id": expected_item["document_id"],
                "model": expected_item["model"],
                "result_path": result_path,
                "result_bytes": result_bytes,
                "payload": generation.get("payload"),
                "eligible": successful,
            }
        )
    return config, manifest, manifest_bytes, rows


def judge_collected_batch(
    config_path: Path,
    *,
    client_factory: ClientFactory = create_client,
) -> dict[str, Any]:
    """Judge locally admitted batch results without mutating generation artifacts."""
    config_path = config_path.resolve()
    config, _, batch_manifest_bytes, rows = _load_batch_judge_rows(config_path)
    judge = _mapping(config.get("judge"), "judge")
    output_dir = _output_dir(config, config_path)
    judgments_dir = output_dir / "judgments"
    final_manifest_path = judgments_dir / "manifest.json"
    claim_path = judgments_dir / "batch-judge.lock"
    claim_binding = _digest(
        {
            "batch_manifest_sha256": _sha256_bytes(batch_manifest_bytes),
            "judge_config_sha256": _digest(judge),
        }
    )
    with _exclusive_batch_judge_claim(claim_path, claim_binding):
        return _judge_collected_batch_claimed(
            rows=rows,
            batch_manifest_bytes=batch_manifest_bytes,
            judge=judge,
            output_dir=output_dir,
            judgments_dir=judgments_dir,
            final_manifest_path=final_manifest_path,
            client_factory=client_factory,
        )


def _judge_collected_batch_claimed(
    *,
    rows: list[dict[str, Any]],
    batch_manifest_bytes: bytes,
    judge: Mapping[str, Any],
    output_dir: Path,
    judgments_dir: Path,
    final_manifest_path: Path,
    client_factory: ClientFactory,
) -> dict[str, Any]:
    """Discover, call, and persist judgments while the caller holds the claim."""
    summary: dict[str, Any] = {
        "ready": False,
        "batch_manifest_path": str(output_dir / "batch-manifest.json"),
        "judgments_dir": str(judgments_dir),
        "manifest_path": str(final_manifest_path),
        "judge_model": judge["model"],
        "eligible": sum(row["eligible"] for row in rows),
        "attempted": 0,
        "judged": 0,
        "judge_failed": 0,
        "skipped_generation_failure": sum(not row["eligible"] for row in rows),
        "skipped_existing_judgment": 0,
        "failures": [],
        "judgments": [],
    }

    pending: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    for row in rows:
        if not row["eligible"]:
            continue
        judgment_path = _batch_judgment_path(output_dir, row["result_path"])
        binding = _batch_judge_binding(
            document_id=row["document_id"],
            generation_model=row["model"],
            generation_result_bytes=row["result_bytes"],
            document=row["document"],
            judge=judge,
        )
        if judgment_path.exists():
            _, judgment_bytes = _load_existing_batch_judgment(
                judgment_path,
                binding=binding,
                judge=judge,
            )
            summary["skipped_existing_judgment"] += 1
            summary["judgments"].append(
                {
                    "document_id": row["document_id"],
                    "model": row["model"],
                    "generation_result_sha256": binding["generation_result_sha256"],
                    "judgment_path": str(judgment_path),
                    "judgment_sha256": _sha256_bytes(judgment_bytes),
                }
            )
        else:
            pending.append((row, judgment_path, binding))

    if final_manifest_path.exists() and pending:
        raise ValueError("final batch judge manifest exists but a bound judgment artifact is missing")

    judge_client = None
    for row, judgment_path, binding in pending:
        summary["attempted"] += 1
        if judge_client is None:
            try:
                judge_client = _client(client_factory, judge["model"], judge.get("provider_routing"))
            except Exception as exc:
                summary["judge_failed"] += 1
                summary["failures"].append(
                    {
                        "document_id": row["document_id"],
                        "model": row["model"],
                        "error": _closed_judge_failure(exc, stage="client_construction"),
                    }
                )
                continue
        candidate = json.dumps(row["payload"], ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        judge_prompt = _render(judge["prompt_template"], document=row["document"], candidate=candidate)
        try:
            response = judge_client.structured_output(
                [{"role": "user", "content": judge_prompt}],
                judge["response_schema"],
                temperature=judge["temperature"],
                max_tokens=judge["max_tokens"],
            )
        except Exception as exc:
            summary["judge_failed"] += 1
            summary["failures"].append(
                {
                    "document_id": row["document_id"],
                    "model": row["model"],
                    "error": _closed_judge_failure(exc, stage="judge_call"),
                }
            )
            continue
        try:
            _require_finite_json(response.value, "judge payload")
            _validate_payload(response.value, judge["response_schema"], "judge payload")
            usage = _usage_payload(response.usage)
            _require_finite_json(usage, "judge usage")
        except Exception as exc:
            summary["judge_failed"] += 1
            summary["failures"].append(
                {
                    "document_id": row["document_id"],
                    "model": row["model"],
                    "error": _closed_judge_failure(exc, stage="judge_validation"),
                }
            )
            continue
        artifact = {
            "kind": _BATCH_JUDGMENT_KIND,
            "binding": binding,
            "judgment": {"model": judge["model"], "usage": usage, "payload": response.value},
        }
        _write_json_immutable(judgment_path, artifact)
        judgment_bytes = _json_bytes(artifact)
        summary["judged"] += 1
        summary["judgments"].append(
            {
                "document_id": row["document_id"],
                "model": row["model"],
                "generation_result_sha256": binding["generation_result_sha256"],
                "judgment_path": str(judgment_path),
                "judgment_sha256": _sha256_bytes(judgment_bytes),
            }
        )

    summary["judgments"].sort(key=lambda item: (item["model"], item["document_id"]))
    if summary["judge_failed"]:
        return summary
    if len(summary["judgments"]) != summary["eligible"]:
        raise ValueError("batch judgment reconciliation is incomplete")
    final_manifest = {
        "kind": _BATCH_JUDGE_MANIFEST_KIND,
        "batch_manifest_path": summary["batch_manifest_path"],
        "batch_manifest_sha256": _sha256_bytes(batch_manifest_bytes),
        "judge_model": judge["model"],
        "judge_config_sha256": _digest(judge),
        "eligible": summary["eligible"],
        "skipped_generation_failure": summary["skipped_generation_failure"],
        "judgments": summary["judgments"],
    }
    _verify_immutable_compatible(final_manifest_path, final_manifest)
    _write_json_immutable(final_manifest_path, final_manifest)
    summary["ready"] = True
    return summary


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
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dry-run", action="store_true", help="Validate configuration and source without model calls")
    mode_group.add_argument("--judge-existing", type=Path, metavar="RESULTS_DIR", help="Retry failed or missing judge calls from existing result JSON")
    mode_group.add_argument("--batch-submit", action="store_true", help="Submit configured documents through the OpenRouter Batch API")
    mode_group.add_argument("--batch-observe", action="store_true", help="Refresh durable OpenRouter batch state")
    mode_group.add_argument("--batch-collect", action="store_true", help="Collect and locally validate completed OpenRouter batch results")
    mode_group.add_argument("--batch-judge", action="store_true", help="Judge completed locally admitted batch results")
    parser.add_argument("--batch-state", type=Path, help="Optional durable batch-state path (defaults under output_dir)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        if args.batch_state is not None and not (args.batch_submit or args.batch_observe or args.batch_collect):
            raise ValueError("--batch-state requires a batch execution mode")
        if args.dry_run:
            if "documents" in config:
                _, plans, spec_digest = _build_batch_plan(args.config)
                payload = {
                    "documents": [item["id"] for item in config["documents"]],
                    "models": [item["id"] for item in config["models"]],
                    "requests": sum(len(item["provider_spec"]["requests"]) for item in plans),
                    "spec_digest": spec_digest,
                    "judge": config.get("judge", {}).get("model"),
                }
            else:
                source_path, _ = read_source(config, args.config.resolve())
                payload = {
                    "source_path": str(source_path),
                    "models": [item["id"] for item in config["models"]],
                    "judge": config.get("judge", {}).get("model"),
                }
            print(json.dumps(payload, indent=2))
            return 0
        if args.batch_submit:
            print(json.dumps(submit_openrouter_batch(args.config, state_path=args.batch_state), indent=2))
            return 0
        if args.batch_observe:
            print(json.dumps(observe_openrouter_batch(args.config, state_path=args.batch_state), indent=2))
            return 0
        if args.batch_collect:
            summary = collect_openrouter_batch(args.config, state_path=args.batch_state)
            print(json.dumps(summary, indent=2))
            return 0 if summary["ready"] and not summary["terminal_failures"] and summary["item_failed"] == 0 else 3
        if args.batch_judge:
            summary = judge_collected_batch(args.config)
            print(json.dumps(summary, indent=2))
            return 0 if summary["ready"] else 3
        if args.judge_existing is not None:
            print(json.dumps(judge_existing(args.config, args.judge_existing), indent=2))
            return 0
        manifest = run_bakeoff(args.config)
        print(json.dumps(manifest, indent=2))
        return 0
    except Exception as exc:
        if args.batch_submit or args.batch_observe or args.batch_collect or args.batch_judge:
            print(
                json.dumps(
                    {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
        else:
            print(f"structured document bakeoff failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
