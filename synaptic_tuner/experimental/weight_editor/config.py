"""Config-first run policy and hash-frozen input capsule.

Everything scientific or model-specific lives in the host-owned config JSON.
The core never branches on a model name. Unknown keys are errors so that a
config cannot silently carry authority the validator does not understand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .canon import is_finite_nonneg, load_strict_file, sha256_file, sha256_hex
from .errors import CapsuleError, ConfigError

CONFIG_SCHEMA = "weight_editor_config/v1"
CAPSULE_SCHEMA = "weight_editor_capsule/v1"

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
OP_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
RESOURCE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")

# Names that denote host authority. Neither a model request nor an operation declaration may use them
# as an argument name: a model cannot be handed a path/command/policy knob.
AUTHORITY_NAMES = frozenset(
    {
        "path", "paths", "file", "files", "command", "cmd", "shell", "exec", "import", "module",
        "url", "uri", "policy", "budget", "limits", "output_root", "env", "cwd", "executor",
        "adapter", "model", "split", "heldout", "held_out", "write", "target", "capsule",
    }
)
BUILTIN_METERS = ("requests", "model_turns", "dispatches")
ARG_TYPES = ("ref", "ref_list", "int", "number", "enum", "text", "artifact_name")


def _strict(obj: Any, required: set[str], optional: set[str], where: str) -> dict:
    if not isinstance(obj, dict):
        raise ConfigError(f"{where} must be an object")
    missing = required - obj.keys()
    if missing:
        raise ConfigError(f"{where} missing keys: {sorted(missing)}")
    extra = obj.keys() - required - optional
    if extra:
        raise ConfigError(f"{where} has unknown keys: {sorted(extra)}")
    return obj


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    settings: Mapping[str, Any]
    allowed_returned: tuple[str, ...]

    def requested(self) -> dict:
        return {"provider": self.provider, "model": self.model, "settings": dict(self.settings)}


@dataclass(frozen=True)
class ArgSpec:
    name: str
    type: str
    required: bool
    params: Mapping[str, Any]


@dataclass(frozen=True)
class OperationSpec:
    name: str
    description: str
    args: Mapping[str, ArgSpec]
    cost: Mapping[str, Any]
    selectable: bool


@dataclass(frozen=True)
class CapsuleObject:
    id: str
    kind: str
    rel_path: str
    sha256: str
    size: int
    abs_path: Path


@dataclass(frozen=True)
class Capsule:
    root: Path
    manifest_bytes: bytes
    manifest_sha256: str
    capsule_id: str
    objects: Mapping[str, CapsuleObject]

    def listing(self) -> list[dict]:
        """Opaque model-visible view: no paths."""
        return [
            {"id": o.id, "kind": o.kind, "sha256": o.sha256, "size": o.size}
            for o in sorted(self.objects.values(), key=lambda o: o.id)
        ]

    def verify_objects(self, ids=None) -> list[str]:
        """Return ids whose on-disk bytes no longer match the frozen hash."""
        bad = []
        for oid in sorted(ids if ids is not None else self.objects):
            obj = self.objects[oid]
            try:
                if (
                    not obj.abs_path.is_file()
                    or obj.abs_path.resolve() != obj.abs_path
                    or sha256_file(obj.abs_path) != obj.sha256
                ):
                    bad.append(oid)
            except OSError:
                bad.append(oid)
        return bad


@dataclass(frozen=True)
class EditorConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    model: ModelSpec
    capsule: Capsule
    instructions: tuple[Mapping[str, Any], ...]
    operations: Mapping[str, OperationSpec]
    output_root: Path
    read_only_roots: tuple[Path, ...]
    limits: Mapping[str, float]
    allow_no_edit: bool
    withheld_digests: frozenset[str]
    max_text_chars: int
    data: Mapping[str, Any] = field(repr=False, default_factory=dict)


# ---------------------------------------------------------------- capsule


def load_capsule(manifest_path: Path) -> Capsule:
    manifest_path = Path(manifest_path)
    data, raw = _load(manifest_path, "capsule manifest", CapsuleError)
    _strict_c(data, {"schema", "capsule_id", "objects"}, set(), "capsule manifest")
    if data["schema"] != CAPSULE_SCHEMA:
        raise CapsuleError(f"capsule schema must be {CAPSULE_SCHEMA}")
    if not isinstance(data["capsule_id"], str) or not ID_RE.match(data["capsule_id"]):
        raise CapsuleError("capsule_id malformed")
    root = manifest_path.resolve().parent
    objects: dict[str, CapsuleObject] = {}
    if not isinstance(data["objects"], list) or not data["objects"]:
        raise CapsuleError("capsule objects must be a non-empty list")
    for i, item in enumerate(data["objects"]):
        where = f"capsule object[{i}]"
        _strict_c(item, {"id", "kind", "path", "sha256", "split"}, set(), where)
        oid, kind, rel = item["id"], item["kind"], item["path"]
        if not isinstance(oid, str) or not ID_RE.match(oid):
            raise CapsuleError(f"{where} id malformed")
        if oid in objects:
            raise CapsuleError(f"duplicate capsule object id {oid!r}")
        if not isinstance(kind, str) or not RESOURCE_RE.match(kind):
            raise CapsuleError(f"{where} kind malformed")
        if item["split"] != "development":
            raise CapsuleError(
                f"{where} split must be 'development': capsule holds registered development objects only",
                code="non_development_object",
            )
        if not isinstance(item["sha256"], str) or not SHA_RE.match(item["sha256"]):
            raise CapsuleError(f"{where} sha256 malformed")
        abs_path = _contained_file(root, rel, where)
        if sha256_file(abs_path) != item["sha256"]:
            raise CapsuleError(f"{where} bytes do not match frozen sha256", code="capsule_hash_mismatch")
        objects[oid] = CapsuleObject(oid, kind, rel, item["sha256"], abs_path.stat().st_size, abs_path)
    return Capsule(root, raw, sha256_hex(raw), data["capsule_id"], objects)


def _strict_c(obj, required, optional, where):
    try:
        return _strict(obj, required, optional, where)
    except ConfigError as exc:
        raise CapsuleError(str(exc)) from exc


def _load(path: Path, what: str, exc_type):
    try:
        return load_strict_file(path, what=what)
    except ConfigError as exc:
        if exc_type is ConfigError:
            raise
        raise exc_type(str(exc), code=exc.code) from exc


def _contained_file(root: Path, rel: Any, where: str) -> Path:
    if not isinstance(rel, str) or not rel:
        raise CapsuleError(f"{where} path must be a non-empty string")
    if rel.startswith(("/", "~")) or "\\" in rel or "\x00" in rel or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", rel):
        raise CapsuleError(f"{where} path must be a relative capsule path", code="path_not_relative")
    if ".." in Path(rel).parts:
        raise CapsuleError(f"{where} path traversal", code="path_traversal")
    candidate = root / rel
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CapsuleError(f"{where} path does not exist", code="path_missing") from exc
    if not resolved.is_relative_to(root):
        raise CapsuleError(f"{where} escapes capsule root (symlink or traversal)", code="path_escape")
    if not resolved.is_file():
        raise CapsuleError(f"{where} is not a regular file", code="path_not_file")
    return resolved


# ----------------------------------------------------------------- config


def load_config(path: Path) -> EditorConfig:
    path = Path(path)
    data, raw = load_strict_file(path, what="config")
    _strict(
        data,
        {"schema", "model", "capsule", "operations", "output_root"},
        {"instructions", "read_only_roots", "resources", "final_choice", "withheld_id_digests"},
        "config",
    )
    if data["schema"] != CONFIG_SCHEMA:
        raise ConfigError(f"config schema must be {CONFIG_SCHEMA}")
    base = path.resolve().parent

    m = _strict(data["model"], {"provider", "model", "settings"}, {"allowed_returned"}, "model")
    for key in ("provider", "model"):
        if not isinstance(m[key], str) or not m[key].strip():
            raise ConfigError(f"model.{key} must be a non-empty string; no default/fallback exists")
    if not isinstance(m["settings"], dict):
        raise ConfigError("model.settings must be an object")
    allowed = m.get("allowed_returned", [])
    if not isinstance(allowed, list) or not all(isinstance(x, str) and x for x in allowed):
        raise ConfigError("model.allowed_returned must be a list of strings")
    model = ModelSpec(m["provider"], m["model"], m["settings"], tuple(allowed))

    cap = _strict(data["capsule"], {"manifest"}, set(), "capsule")
    if not isinstance(cap["manifest"], str):
        raise ConfigError("capsule.manifest must be a path string")
    capsule = load_capsule(_host_path(base, cap["manifest"]))

    instructions = []
    for i, item in enumerate(data.get("instructions", [])):
        where = f"instructions[{i}]"
        if not isinstance(item, dict) or item.get("role") not in ("system", "user"):
            raise ConfigError(f"{where} needs role system|user")
        if set(item) == {"role", "text"} and isinstance(item["text"], str):
            instructions.append({"role": item["role"], "text": item["text"]})
        elif set(item) == {"role", "object"} and isinstance(item["object"], str):
            obj = capsule.objects.get(item["object"])
            if obj is None or obj.kind != "instructions":
                raise ConfigError(f"{where} object must be a capsule object of kind 'instructions'")
            instructions.append({"role": item["role"], "object": item["object"]})
        else:
            raise ConfigError(f"{where} must be exactly role+text or role+object")

    operations = _parse_operations(data["operations"])

    out_root = _host_path(base, data["output_root"]) if isinstance(data["output_root"], str) else None
    if out_root is None:
        raise ConfigError("output_root must be a path string")
    out_root = out_root.resolve()
    ro = []
    for item in data.get("read_only_roots", []):
        if not isinstance(item, str):
            raise ConfigError("read_only_roots entries must be path strings")
        ro.append(_host_path(base, item).resolve())
    for guarded in (capsule.root, *ro):
        if out_root.is_relative_to(guarded) or guarded.is_relative_to(out_root):
            raise ConfigError(
                f"output_root {out_root} overlaps source/base root {guarded}", code="output_root_overlap"
            )

    res = _strict(data.get("resources", {}), set(), {"limits"}, "resources")
    limits: dict[str, float] = {}
    lim = res.get("limits", {})
    if not isinstance(lim, dict):
        raise ConfigError("resources.limits must be an object")
    for name, val in lim.items():
        if not RESOURCE_RE.match(name) or not is_finite_nonneg(val):
            raise ConfigError(f"resource limit {name!r} must be a finite non-negative number")
        limits[name] = val
    for op in operations.values():
        for rname in op.cost:
            if not RESOURCE_RE.match(rname):
                raise ConfigError(f"operation {op.name} cost resource {rname!r} malformed")
    measured = {"requests", "model_turns", "dispatches", "dispatch_attempts"}
    measured.update(r for op in operations.values() for r in op.cost)
    unmeasured = set(limits) - measured
    if unmeasured:
        raise ConfigError(f"limits have no configured meter: {sorted(unmeasured)}", code="unmeasured_resource")

    fc = _strict(data.get("final_choice", {}), set(), {"allow_no_edit", "max_rationale_chars"}, "final_choice")
    allow_no_edit = fc.get("allow_no_edit", True)
    if not isinstance(allow_no_edit, bool):
        raise ConfigError("final_choice.allow_no_edit must be boolean")
    max_text = fc.get("max_rationale_chars", 20000)
    if not isinstance(max_text, int) or isinstance(max_text, bool) or max_text <= 0:
        raise ConfigError("final_choice.max_rationale_chars must be a positive integer")

    withheld = data.get("withheld_id_digests", [])
    if not isinstance(withheld, list) or not all(isinstance(x, str) and SHA_RE.match(x) for x in withheld):
        raise ConfigError("withheld_id_digests must be a list of sha256 hex digests")

    return EditorConfig(
        path=path.resolve(),
        raw_bytes=raw,
        sha256=sha256_hex(raw),
        model=model,
        capsule=capsule,
        instructions=tuple(instructions),
        operations=operations,
        output_root=out_root,
        read_only_roots=tuple(ro),
        limits=limits,
        allow_no_edit=allow_no_edit,
        withheld_digests=frozenset(withheld),
        max_text_chars=max_text,
        data=data,
    )


def _host_path(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def _parse_operations(ops: Any) -> dict[str, OperationSpec]:
    if not isinstance(ops, dict) or not ops:
        raise ConfigError("operations must be a non-empty object (no operation is admitted by default)")
    out: dict[str, OperationSpec] = {}
    for name, spec in ops.items():
        if not OP_RE.match(name):
            raise ConfigError(f"operation name {name!r} malformed")
        where = f"operations.{name}"
        spec = _strict(spec, {"arguments"}, {"description", "cost", "selectable", "required"}, where)
        args: dict[str, ArgSpec] = {}
        required = spec.get("required", [])
        if not isinstance(required, list) or not all(isinstance(x, str) for x in required):
            raise ConfigError(f"{where}.required must be a list of argument names")
        if not isinstance(spec["arguments"], dict):
            raise ConfigError(f"{where}.arguments must be an object")
        for an, aspec in spec["arguments"].items():
            if not ID_RE.match(an):
                raise ConfigError(f"{where} argument name {an!r} malformed")
            if an in AUTHORITY_NAMES:
                raise ConfigError(f"{where} argument name {an!r} denotes host authority and cannot be model-settable",
                                  code="authority_argument_name")
            args[an] = _parse_arg(an, aspec, an in required, f"{where}.arguments.{an}")
        for r in required:
            if r not in args:
                raise ConfigError(f"{where}.required names undeclared argument {r!r}")
        cost = spec.get("cost", {})
        if not isinstance(cost, dict):
            raise ConfigError(f"{where}.cost must be an object")
        for rname, c in cost.items():
            _validate_cost(c, args, f"{where}.cost.{rname}")
        sel = spec.get("selectable", False)
        if not isinstance(sel, bool):
            raise ConfigError(f"{where}.selectable must be boolean")
        desc = spec.get("description", "")
        if not isinstance(desc, str):
            raise ConfigError(f"{where}.description must be a string")
        out[name] = OperationSpec(name, desc, args, cost, sel)
    return out


def _parse_arg(name: str, spec: Any, required: bool, where: str) -> ArgSpec:
    if not isinstance(spec, dict) or spec.get("type") not in ARG_TYPES:
        raise ConfigError(f"{where}.type must be one of {ARG_TYPES}")
    t = spec["type"]
    allowed = {
        "ref": ({"type", "kind"}, {"kind"}),
        "ref_list": ({"type", "kind", "max_items"}, {"kind", "max_items"}),
        "int": ({"type", "min", "max"}, {"min", "max"}),
        "number": ({"type", "min", "max"}, {"min", "max"}),
        "enum": ({"type", "values"}, {"values"}),
        "text": ({"type", "max_length"}, {"max_length"}),
        "artifact_name": ({"type"}, set()),
    }[t]
    extra = spec.keys() - allowed[0]
    missing = allowed[1] - spec.keys()
    if extra or missing:
        raise ConfigError(f"{where} bad keys (extra={sorted(extra)} missing={sorted(missing)})")
    params = {k: v for k, v in spec.items() if k != "type"}
    if "kind" in params and not (isinstance(params["kind"], str) and RESOURCE_RE.match(params["kind"])):
        raise ConfigError(f"{where}.kind malformed")
    for key in ("max_items", "max_length"):
        if key in params and not (isinstance(params[key], int) and not isinstance(params[key], bool) and params[key] > 0):
            raise ConfigError(f"{where}.{key} must be a positive integer")
    for key in ("min", "max"):
        if key in params:
            v = params[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) == float("inf"):
                raise ConfigError(f"{where}.{key} must be finite")
            if t == "int" and not isinstance(v, int):
                raise ConfigError(f"{where}.{key} must be an integer")
    if "min" in params and params["min"] > params["max"]:
        raise ConfigError(f"{where} min > max")
    if t == "enum" and not (
        isinstance(params["values"], list) and params["values"] and all(isinstance(x, str) for x in params["values"])
    ):
        raise ConfigError(f"{where}.values must be a non-empty list of strings")
    return ArgSpec(name, t, required, params)


def _validate_cost(c: Any, args: Mapping[str, ArgSpec], where: str) -> None:
    if is_finite_nonneg(c):
        return
    if isinstance(c, dict) and set(c) == {"from_arg"}:
        a = args.get(c["from_arg"])
        if a is None or a.type not in ("int", "number") or "min" not in a.params or a.params["min"] < 0:
            raise ConfigError(f"{where}.from_arg must name a bounded non-negative numeric argument")
        return
    if isinstance(c, dict) and set(c) == {"len_of", "per_item"}:
        a = args.get(c["len_of"])
        if a is None or a.type != "ref_list" or not is_finite_nonneg(c["per_item"]):
            raise ConfigError(f"{where}.len_of must name a ref_list argument with finite per_item")
        return
    raise ConfigError(f"{where} must be a finite non-negative number, {{from_arg}} or {{len_of, per_item}}")
