#!/usr/bin/env python3
"""Render validated structured-document batch artifacts into create-only Markdown.

This is deliberately a small renderer, not a general template engine.  All
identity-to-path mappings and all selected fields are declared in configuration.
No provider calls are made by any lifecycle verb.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml
from yaml.tokens import AliasToken, AnchorToken

from SynthChat.scripts.structured_document_bakeoff import load_validated_batch_artifacts


_CONFIG_KIND = "structured_document_export/v1"
_MANIFEST_KIND = "structured_document_export/manifest/v1"
_DISPOSITIONS = {"accept", "review", "reject"}
_FORMATS = {"text", "json", "yaml", "markdown"}
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)), "com¹", "com²", "com³",
    "lpt¹", "lpt²", "lpt³", "clock$", "conin$", "conout$",
}
_TOKEN = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_WINDOWS_INVALID_COMPONENT = re.compile(r'[<>:"|?*]')


class _NoDuplicateSafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError:
            raise ValueError("export config contains a non-scalar mapping key") from None
        if duplicate:
            raise ValueError("export config contains a duplicate mapping key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_NoDuplicateSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _exact(mapping: Mapping[str, Any], keys: set[str], name: str) -> None:
    if set(mapping) != keys:
        raise ValueError(f"{name} fields are invalid")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _absolute_from(base: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return Path(os.path.abspath(path))


def _relative_output(value: object, name: str) -> str:
    text = _text(value, name)
    if "\\" in text or _CONTROL.search(text):
        raise ValueError(f"{name} is not a safe POSIX relative path")
    raw_parts = text.split("/")
    if text.startswith("/") or not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"{name} is not a safe POSIX relative path")
    for part in raw_parts:
        if (
            len(part) > 255
            or _WINDOWS_INVALID_COMPONENT.search(part)
            or part.endswith((".", " "))
            or part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED
        ):
            raise ValueError(f"{name} is not portable to Windows")
    return "/".join(raw_parts)


def _reject_path_topology_conflicts(paths: list[str]) -> None:
    """Reject exact and file-vs-directory collisions under Windows case folding."""
    folded = sorted(tuple(part.casefold() for part in path.split("/")) for path in paths)
    for previous, current in zip(folded, folded[1:]):
        shared = min(len(previous), len(current))
        if previous[:shared] == current[:shared]:
            raise ValueError("destination paths have a case-folded topology conflict")


def _validate_pointer(pointer: object, name: str) -> str:
    pointer = _text(pointer, name)
    if not pointer.startswith("/"):
        raise ValueError(f"{name} must be an RFC 6901 JSON pointer")
    for token in pointer.split("/")[1:]:
        if re.search(r"~(?:[^01]|$)", token):
            raise ValueError(f"{name} contains an invalid JSON pointer escape")
    return pointer


def _select(root: object, pointer: str) -> object:
    value = root
    for encoded in pointer.split("/")[1:]:
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            if token not in value:
                raise ValueError("configured selection is absent")
            value = value[token]
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", token):
            index = int(token)
            if index >= len(value):
                raise ValueError("configured selection is absent")
            value = value[index]
        else:
            raise ValueError("configured selection is absent")
    return value


def _validate_render_node(value: object, name: str, *, root: bool = False) -> object:
    if isinstance(value, dict):
        if set(value) == {"$literal"}:
            json.dumps(value["$literal"], ensure_ascii=False, allow_nan=False)
            return {"$literal": value["$literal"]}
        if set(value) == {"$select"}:
            return {"$select": _validate_pointer(value["$select"], f"{name}.$select")}
        if not root and any(str(key).startswith("$") for key in value):
            raise ValueError(f"{name} has an unsupported render directive")
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key or _CONTROL.search(key):
                raise ValueError(f"{name} has an invalid mapping key")
            result[key] = _validate_render_node(child, f"{name}.{key}")
        return result
    if isinstance(value, list):
        return [_validate_render_node(child, f"{name}[]") for child in value]
    raise ValueError(f"{name} leaves must use $literal or $select")


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    encoded = path.read_bytes()
    if len(encoded) > 16 * 1024 * 1024:
        raise ValueError("export config exceeds the byte limit")
    try:
        source = encoded.decode("utf-8")
        if any(isinstance(token, (AliasToken, AnchorToken)) for token in yaml.scan(source)):
            raise ValueError("export config cannot contain YAML anchors or aliases")
        loaded = yaml.load(source, Loader=_NoDuplicateSafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError):
        raise ValueError("export config is not valid UTF-8 YAML") from None
    config = _mapping(loaded, "export config")
    _exact(config, {"kind", "input", "destination", "outputs", "render", "limits"}, "export config")
    if config.get("kind") != _CONFIG_KIND:
        raise ValueError("export config kind is unsupported")

    input_spec = _mapping(config["input"], "input")
    _exact(input_spec, {"bakeoff_config", "judgments"}, "input")
    input_spec["bakeoff_config"] = _text(input_spec["bakeoff_config"], "input.bakeoff_config")
    judgments = _mapping(input_spec["judgments"], "input.judgments")
    _exact(judgments, {"mode", "manifest_path", "disposition_pointer", "disposition_map", "when_absent"}, "input.judgments")
    if judgments.get("mode") not in {"required", "optional", "none"}:
        raise ValueError("input.judgments.mode is invalid")
    manifest_path = judgments.get("manifest_path")
    if manifest_path is not None:
        judgments["manifest_path"] = _text(manifest_path, "input.judgments.manifest_path")
    if judgments["mode"] == "required" and manifest_path is None:
        raise ValueError("required judgments need manifest_path")
    if judgments["mode"] == "none" and manifest_path is not None:
        raise ValueError("none judgments cannot set manifest_path")
    judgments["disposition_pointer"] = _validate_pointer(judgments["disposition_pointer"], "input.judgments.disposition_pointer")
    disposition_map = _mapping(judgments["disposition_map"], "input.judgments.disposition_map")
    if not disposition_map or any(not isinstance(key, str) or value not in _DISPOSITIONS for key, value in disposition_map.items()):
        raise ValueError("input.judgments.disposition_map is invalid")
    judgments["disposition_map"] = disposition_map
    if judgments.get("when_absent") not in _DISPOSITIONS:
        raise ValueError("input.judgments.when_absent is invalid")
    input_spec["judgments"] = judgments
    config["input"] = input_spec

    destination = _mapping(config["destination"], "destination")
    _exact(destination, {"root", "manifest", "emit"}, "destination")
    destination["root"] = _text(destination["root"], "destination.root")
    destination["manifest"] = _relative_output(destination["manifest"], "destination.manifest")
    emit = _mapping(destination["emit"], "destination.emit")
    _exact(emit, _DISPOSITIONS, "destination.emit")
    if any(type(value) is not bool for value in emit.values()):
        raise ValueError("destination.emit values must be booleans")
    destination["emit"] = emit
    config["destination"] = destination

    if not isinstance(config["outputs"], list) or not config["outputs"]:
        raise ValueError("outputs must be a non-empty list")
    outputs: list[dict[str, str]] = []
    identities: set[tuple[str, str]] = set()
    folded_paths: set[str] = set()
    for index, raw in enumerate(config["outputs"]):
        item = _mapping(raw, f"outputs[{index}]")
        _exact(item, {"document_id", "model", "relative_path"}, f"outputs[{index}]")
        normalized = {
            "document_id": _text(item["document_id"], f"outputs[{index}].document_id"),
            "model": _text(item["model"], f"outputs[{index}].model"),
            "relative_path": _relative_output(item["relative_path"], f"outputs[{index}].relative_path"),
        }
        identity = (normalized["document_id"], normalized["model"])
        folded = normalized["relative_path"].casefold()
        if identity in identities or folded in folded_paths:
            raise ValueError("outputs contain a duplicate identity or case-folded path")
        identities.add(identity)
        folded_paths.add(folded)
        outputs.append(normalized)
    config["outputs"] = outputs
    _reject_path_topology_conflicts(
        [item["relative_path"] for item in outputs] + [destination["manifest"]]
    )
    claim_name = ".structured-document-export.lock".casefold()
    if any(PurePosixPath(item["relative_path"]).parts[0].casefold() == claim_name for item in outputs):
        raise ValueError("an output path collides with the exporter claim")
    if PurePosixPath(destination["manifest"]).parts[0].casefold() == claim_name:
        raise ValueError("destination manifest collides with the exporter claim")

    render = _mapping(config["render"], "render")
    _exact(render, {"frontmatter", "body_template", "variables"}, "render")
    render["frontmatter"] = _validate_render_node(render["frontmatter"], "render.frontmatter", root=True)
    render["body_template"] = _text(render["body_template"], "render.body_template")
    variables = _mapping(render["variables"], "render.variables")
    validated_variables: dict[str, dict[str, str]] = {}
    for name, raw in variables.items():
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ValueError("render variable name is invalid")
        variable = _mapping(raw, f"render.variables.{name}")
        _exact(variable, {"select", "format"}, f"render.variables.{name}")
        pointer = _validate_pointer(variable["select"], f"render.variables.{name}.select")
        if variable.get("format") not in _FORMATS:
            raise ValueError(f"render.variables.{name}.format is invalid")
        validated_variables[name] = {"select": pointer, "format": variable["format"]}
    tokens = _TOKEN.findall(render["body_template"])
    residue = _TOKEN.sub("", render["body_template"])
    if "{{" in residue or "}}" in residue or set(tokens) != set(validated_variables):
        raise ValueError("body_template tokens must exactly match variables")
    render["variables"] = validated_variables
    config["render"] = render

    limits = _mapping(config["limits"], "limits")
    _exact(limits, {"max_documents", "max_output_bytes_per_document", "max_total_output_bytes"}, "limits")
    for key in list(limits):
        limits[key] = _positive_int(limits[key], f"limits.{key}")
    config["limits"] = limits
    if len(outputs) > limits["max_documents"]:
        raise ValueError("outputs exceed max_documents")
    return config, encoded


def _render_node(spec: object, context: Mapping[str, Any]) -> object:
    if isinstance(spec, dict):
        if set(spec) == {"$literal"}:
            return spec["$literal"]
        if set(spec) == {"$select"}:
            return _select(context, spec["$select"])
        return {key: _render_node(value, context) for key, value in spec.items()}
    return [_render_node(value, context) for value in spec]  # type: ignore[union-attr]


def _markdown(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return str(value).lower() if isinstance(value, bool) else ("null" if value is None else str(value))
    if isinstance(value, list):
        return "\n".join(f"- {_markdown(item).replace(chr(10), chr(10) + '  ')}" for item in value)
    if isinstance(value, dict):
        sections: list[str] = []
        for key, child in value.items():
            rendered = _markdown(child)
            sections.append(f"## {key}\n\n{rendered}" if rendered else f"## {key}")
        return "\n\n".join(sections)
    raise ValueError("markdown selection is not finite JSON")


def _format(value: object, format_name: str) -> str:
    if format_name == "text":
        if not isinstance(value, str):
            raise ValueError("text format requires a string")
        return value
    if format_name == "json":
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
    if format_name == "yaml":
        return yaml.safe_dump(value, allow_unicode=True, sort_keys=False, default_flow_style=False).rstrip("\n")
    return _markdown(value)


def _context(row: Mapping[str, Any], disposition: str) -> dict[str, Any]:
    judgment = row.get("judgment")
    judgment_payload = None
    if isinstance(judgment, dict):
        judgment_payload = judgment["judgment"]["payload"]
    return {
        "identity": {"document_id": row["document_id"], "model": row["model"]},
        "source": {"metadata": row["source_metadata"]},
        "generation": {"payload": row["payload"]},
        "judgment": {"payload": judgment_payload},
        "quality": {"disposition": disposition},
        "hashes": {
            "source_sha256": row["source_sha256"],
            "generation_result_sha256": row["generation_result_sha256"],
            "judgment_sha256": row.get("judgment_sha256"),
        },
    }


def _disposition(row: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    if "judgment" not in row:
        return spec["when_absent"]
    provisional = _context(row, spec["when_absent"])
    selected = _select(provisional, spec["disposition_pointer"])
    if isinstance(selected, bool):
        key = "true" if selected else "false"
    elif selected is None:
        key = "null"
    elif isinstance(selected, (str, int, float)):
        key = str(selected)
    else:
        raise ValueError("judgment disposition is not scalar")
    if key not in spec["disposition_map"]:
        raise ValueError("judgment disposition is not mapped")
    return spec["disposition_map"][key]


def _render_document(context: Mapping[str, Any], render: Mapping[str, Any]) -> bytes:
    frontmatter = _render_node(render["frontmatter"], context)
    if not isinstance(frontmatter, dict):
        raise ValueError("rendered frontmatter must be a mapping")
    yaml_text = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False, default_flow_style=False, width=1_000_000).rstrip("\n")
    variables = {
        name: _format(_select(context, spec["select"]), spec["format"])
        for name, spec in render["variables"].items()
    }
    body = _TOKEN.sub(lambda match: variables[match.group(1)], render["body_template"])
    body = body.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    return f"---\n{yaml_text}\n---\n\n{body}\n".encode("utf-8")


def _is_reparse(path: Path) -> bool:
    info = path.lstat()
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _check_existing_ancestors(path: Path) -> None:
    current = path
    existing: list[Path] = []
    while True:
        if current.exists() or current.is_symlink():
            existing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for item in reversed(existing):
        if _is_reparse(item):
            raise ValueError("destination has a reparse or symlink ancestor")
        if not item.is_dir():
            raise ValueError("destination directory ancestry contains a non-directory")


def _check_leaf(path: Path) -> None:
    _check_existing_ancestors(path.parent)
    if path.exists() or path.is_symlink():
        if _is_reparse(path) or not path.is_file():
            raise ValueError("destination leaf is not a regular file")


def _same_bytes(path: Path, expected: bytes) -> bool:
    try:
        before = path.lstat()
        if _is_reparse(path) or not stat.S_ISREG(before.st_mode):
            raise ValueError("destination leaf is not a regular file")
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise ValueError("destination leaf changed while opening")
            if opened.st_size != len(expected):
                return False
            actual = handle.read(len(expected) + 1)
            after = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (after.st_dev, after.st_ino, after.st_size):
            raise ValueError("destination leaf changed while reading")
        return actual == expected
    except OSError:
        raise ValueError("destination leaf cannot be verified") from None


def _compile(config_path: Path, *, inspect_destination: bool) -> dict[str, Any]:
    config_path = config_path.resolve()
    config, config_bytes = _load_config(config_path)
    base = config_path.parent
    bakeoff_path = _absolute_from(base, config["input"]["bakeoff_config"])
    judgment_spec = config["input"]["judgments"]
    judgment_manifest = None
    if judgment_spec["manifest_path"] is not None:
        candidate = _absolute_from(base, judgment_spec["manifest_path"])
        if candidate.exists():
            judgment_manifest = candidate
        elif judgment_spec["mode"] == "required":
            raise ValueError("required judgment manifest is absent")
    loaded = load_validated_batch_artifacts(
        bakeoff_path,
        judgment_manifest_path=judgment_manifest,
    )
    if judgment_spec["mode"] == "required" and judgment_manifest is None:
        raise ValueError("required judgment manifest is absent")

    rows = {(row["document_id"], row["model"]): row for row in loaded["rows"] if row["eligible"]}
    output_index = {(item["document_id"], item["model"]): item for item in config["outputs"]}
    if set(output_index) != set(rows):
        raise ValueError("outputs must exactly map every eligible generation result")

    root = _absolute_from(base, config["destination"]["root"])
    manifest_path = root.joinpath(*PurePosixPath(config["destination"]["manifest"]).parts)
    planned: list[dict[str, Any]] = []
    total_bytes = 0
    for identity in sorted(rows):
        row = rows[identity]
        disposition = _disposition(row, judgment_spec)
        output_spec = output_index[identity]
        encoded = _render_document(_context(row, disposition), config["render"])
        if len(encoded) > config["limits"]["max_output_bytes_per_document"]:
            raise ValueError("rendered document exceeds per-document byte limit")
        total_bytes += len(encoded)
        if total_bytes > config["limits"]["max_total_output_bytes"]:
            raise ValueError("rendered documents exceed total byte limit")
        relative_path = output_spec["relative_path"]
        output_path = root.joinpath(*PurePosixPath(relative_path).parts)
        planned.append({
            "document_id": identity[0], "model": identity[1],
            "relative_path": relative_path, "output_path": output_path,
            "bytes": encoded, "disposition": disposition,
            "emit": config["destination"]["emit"][disposition],
            "source_sha256": row["source_sha256"],
            "generation_result_sha256": row["generation_result_sha256"],
            "judgment_sha256": row.get("judgment_sha256"),
            "output_sha256": _sha256(encoded),
        })
    emitted = [item for item in planned if item["emit"]]
    manifest_header = {
        "type": "header", "kind": _MANIFEST_KIND,
        "config_sha256": _sha256(config_bytes),
        "collection_manifest_sha256": loaded["collection_manifest_sha256"],
        "judgment_manifest_sha256": loaded.get("judgment_manifest_sha256"),
        "output_count": len(emitted),
    }
    manifest_items = [
        {key: item[key] for key in (
            "document_id", "model", "relative_path", "disposition", "source_sha256",
            "generation_result_sha256", "judgment_sha256", "output_sha256",
        )} | {"type": "item"}
        for item in emitted
    ]
    manifest_bytes = b"".join(_canonical_json(item) for item in [manifest_header, *manifest_items])
    plan = {
        "config": config, "root": root, "manifest_path": manifest_path,
        "manifest_bytes": manifest_bytes, "planned": planned, "emitted": emitted,
        "config_sha256": _sha256(config_bytes),
    }
    if inspect_destination:
        _check_existing_ancestors(root)
        for item in emitted:
            _check_leaf(item["output_path"])
            if item["output_path"].exists() and not _same_bytes(item["output_path"], item["bytes"]):
                raise ValueError("destination output exists with different bytes")
        _check_leaf(manifest_path)
        if manifest_path.exists() and not _same_bytes(manifest_path, manifest_bytes):
            raise ValueError("destination manifest exists with different bytes")
    return plan


def _summary(plan: Mapping[str, Any], operation: str) -> dict[str, Any]:
    return {
        "ok": True, "operation": operation,
        "config_sha256": plan["config_sha256"],
        "documents": len(plan["planned"]), "emitted": len(plan["emitted"]),
        "manifest_sha256": _sha256(plan["manifest_bytes"]),
    }


def preflight_export(config_path: Path) -> dict[str, Any]:
    return _summary(_compile(config_path, inspect_destination=False), "preflight")


def dry_run_export(config_path: Path) -> dict[str, Any]:
    return _summary(_compile(config_path, inspect_destination=True), "dry-run")


@contextmanager
def _claim(root: Path, binding: str):
    _check_existing_ancestors(root)
    root.mkdir(parents=True, exist_ok=True)
    _check_existing_ancestors(root)
    claim_path = root / ".structured-document-export.lock"
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise RuntimeError("destination is exclusively claimed") from None
    identity = os.fstat(descriptor)
    try:
        os.write(descriptor, _canonical_json({"kind": "structured_document_export/claim/v1", "binding": binding, "pid": os.getpid()}))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        try:
            current = claim_path.stat()
        except FileNotFoundError:
            current = None
        if current is not None and (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
            claim_path.unlink()


def _publish(path: Path, encoded: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_existing_ancestors(path.parent)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, path)
            return True
        except FileExistsError:
            _check_leaf(path)
            if _same_bytes(path, encoded):
                return False
            raise ValueError("destination changed during create-only publication") from None
    finally:
        if temporary.exists():
            temporary.unlink()


def execute_export(config_path: Path) -> dict[str, Any]:
    initial = _compile(config_path, inspect_destination=True)
    binding = _sha256(initial["manifest_bytes"])
    with _claim(initial["root"], binding):
        plan = _compile(config_path, inspect_destination=True)
        if _sha256(plan["manifest_bytes"]) != binding:
            raise ValueError("export inputs changed before publication")
        created = 0
        for item in plan["emitted"]:
            created += int(_publish(item["output_path"], item["bytes"]))
        manifest_created = _publish(plan["manifest_path"], plan["manifest_bytes"])
    summary = _summary(plan, "execute")
    summary.update({"created": created, "existing_identical": len(plan["emitted"]) - created, "manifest_created": manifest_created})
    return summary


def verify_export(config_path: Path) -> dict[str, Any]:
    plan = _compile(config_path, inspect_destination=True)
    for item in plan["emitted"]:
        if not item["output_path"].is_file() or not _same_bytes(item["output_path"], item["bytes"]):
            raise ValueError("published output is absent or changed")
    if not plan["manifest_path"].is_file() or not _same_bytes(plan["manifest_path"], plan["manifest_bytes"]):
        raise ValueError("published manifest is absent or changed")
    return _summary(plan, "verify")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    operation = "preflight" if args.preflight else "dry-run" if args.dry_run else "execute" if args.execute else "verify"
    try:
        result = {
            "preflight": preflight_export,
            "dry-run": dry_run_export,
            "execute": execute_export,
            "verify": verify_export,
        }[operation](args.config)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        error_type = type(exc).__name__ if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", type(exc).__name__) else "Exception"
        print(json.dumps({"ok": False, "operation": operation, "error": {"type": error_type}}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
