"""Declarative ``mechinterp-weight-patch/v1`` and ``mechinterp-weights-inspect/v1``.

A patch spec is a YAML/JSON document that binds a patch to exact base bytes and
declares every restriction up front:

  base      source identity: declared revision, optional index/file hashes, and
            the dtype/shape/byte-hash of EVERY tensor an operation targets.
  budget    finite limits: allowed tensor patterns, max modified tensors and
            entries, per-tensor relative Frobenius norm, payload size.
  payload   a safetensors file (+ sha256) holding all numeric arrays; the spec
            itself carries no floating-point data except scalar scales.
  operations the edits (``add_rank_one``, ``project``, ``replace_rows``,
            ``replace_cols``); indices inline or from integer payload arrays.
  export    optional destination for ``weights-patch --export``.

This module only validates *structure*. Byte-level checks happen in
checkpoint.py / patch.py. No tensor name, shape or value is hardcoded here.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from MechInterp.weights.errors import WeightPatchError
from MechInterp.weights.safetensors_io import FLOAT_TORCH_NAMES

PATCH_SCHEMA = "mechinterp-weight-patch/v1"
INSPECT_SCHEMA = "mechinterp-weights-inspect/v1"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REV40 = re.compile(r"^[0-9a-f]{40}$")
_MAX_SPEC_BYTES = 8 * 1024 * 1024

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Hex64 = Annotated[str, Field(pattern=_HEX64.pattern)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaseTensorBinding(_Strict):
    """Exact identity of one targeted base tensor."""

    dtype: StrictStr
    shape: list[StrictInt] = Field(..., min_length=1)
    sha256: Hex64

    @model_validator(mode="after")
    def _check(self) -> "BaseTensorBinding":
        if self.dtype not in FLOAT_TORCH_NAMES:
            raise ValueError(
                f"dtype {self.dtype!r} is not patchable; allowed {sorted(FLOAT_TORCH_NAMES)}"
            )
        if any(d < 0 for d in self.shape):
            raise ValueError("shape must be non-negative integers")
        return self


class RuntimePrefixRule(_Strict):
    """Maps checkpoint tensor-name prefixes to in-memory parameter-name prefixes.

    Only needed when the loaded module names differ from checkpoint file names
    (some wrapper architectures). First matching rule wins; no rule = identity.
    """

    checkpoint_prefix: StrictStr
    runtime_prefix: StrictStr


class BaseBinding(_Strict):
    model_id: Optional[StrictStr] = None
    revision: StrictStr = Field(..., description="Full 40-hex commit/revision id")
    checkpoint: Optional[StrictStr] = Field(
        default=None,
        description="LOCAL safetensors directory or file; required by weights-patch.",
    )
    index_sha256: Optional[Hex64] = Field(
        default=None,
        description="sha256 of model.safetensors.index.json (sharded) or of the single file.",
    )
    files_sha256: Optional[dict[StrictStr, Hex64]] = Field(
        default=None, description="Whole-file sha256 per weight file name."
    )
    allow_symlinked_files: bool = False
    runtime_prefix_rules: list[RuntimePrefixRule] = Field(default_factory=list)
    tensors: dict[StrictStr, BaseTensorBinding] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "BaseBinding":
        if not _REV40.match(self.revision):
            raise ValueError("base.revision must be a full 40-character lowercase hex id")
        for name in self.tensors:
            if not name or len(name) > 512:
                raise ValueError("tensor names must be 1..512 characters")
        if self.files_sha256 is not None:
            for fname in self.files_sha256:
                if "/" in fname or "\\" in fname or fname in {"", ".", ".."}:
                    raise ValueError(f"files_sha256 key {fname!r} must be a plain file name")
        return self


class Budget(_Strict):
    allowed_tensors: list[StrictStr] = Field(..., min_length=1)
    max_modified_tensors: StrictInt = Field(..., ge=1, le=100_000)
    max_modified_entries: StrictInt = Field(..., ge=1)
    max_relative_frobenius: FiniteFloat = Field(..., ge=0.0)
    max_payload_bytes: StrictInt = Field(default=256 * 1024 * 1024, ge=1, le=4 * 1024**3)
    max_index_count: StrictInt = Field(default=1_000_000, ge=1, le=100_000_000)


class PayloadRef(_Strict):
    path: StrictStr = Field(..., min_length=1)
    sha256: Hex64


OpName = Literal["add_rank_one", "project", "replace_rows", "replace_cols"]

# role -> required?  (per op)
_ROLES: dict[str, dict[str, bool]] = {
    "add_rank_one": {"u": True, "v": True},
    "project": {"direction": True},
    "replace_rows": {"values": True},
    "replace_cols": {"values": True},
}
_INDEX_ROLES = {"rows", "cols"}


class Operation(_Strict):
    """One edit of a 2-D tensor ``W`` of shape ``[out, in]``.

    add_rank_one   ``W[R][:, C] += scale * outer(u, v)`` with optional row set R
                   (``rows``) and column set C (``cols``); ``len(u)==|R|``,
                   ``len(v)==|C|`` (omitted set = every index).
    project        remove (scaled) component along unit ``direction``:
                   ``axis: output`` -> ``W[:, C] -= scale * d (d^T W[:, C])``
                   (d has length ``out``; optional ``cols``);
                   ``axis: input``  -> ``W[R] -= scale * (W[R] d) d^T``
                   (d has length ``in``; optional ``rows``).
    replace_rows   ``W[R] = values`` (``values`` is ``[|R|, in]``; ``rows`` required).
    replace_cols   ``W[:, C] = values`` (``values`` is ``[out, |C|]``; ``cols`` required).

    ``arrays`` maps each role to a payload tensor name. ``rows``/``cols`` are
    either inline integer lists or payload integer arrays via
    ``arrays: {rows: <name>}`` (never both).
    """

    id: StrictStr = Field(..., min_length=1, max_length=128)
    tensor: StrictStr = Field(..., min_length=1, max_length=512)
    op: OpName
    scale: Optional[FiniteFloat] = None
    axis: Optional[Literal["output", "input"]] = None
    normalize: bool = False
    rows: Optional[list[StrictInt]] = None
    cols: Optional[list[StrictInt]] = None
    arrays: dict[StrictStr, StrictStr] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "Operation":
        roles = _ROLES[self.op]
        allowed = set(roles) | _INDEX_ROLES
        for role, name in self.arrays.items():
            if role not in allowed:
                raise ValueError(f"op {self.id}: unknown array role {role!r} for {self.op}")
            if not name:
                raise ValueError(f"op {self.id}: empty payload name for role {role!r}")
        for role, required in roles.items():
            if required and role not in self.arrays:
                raise ValueError(f"op {self.id}: {self.op} requires arrays.{role}")
        for role in _INDEX_ROLES:
            inline = getattr(self, role)
            if inline is not None and role in self.arrays:
                raise ValueError(f"op {self.id}: {role} given both inline and as an array")
        has_rows = self.rows is not None or "rows" in self.arrays
        has_cols = self.cols is not None or "cols" in self.arrays

        if self.op in {"add_rank_one", "project"}:
            if self.scale is None:
                self.scale = 1.0
        elif self.scale is not None:
            raise ValueError(f"op {self.id}: scale is not valid for {self.op}")

        if self.op == "project":
            if self.axis is None:
                raise ValueError(f"op {self.id}: project requires axis")
            if self.axis == "output" and has_rows:
                raise ValueError(f"op {self.id}: axis=output cannot restrict rows")
            if self.axis == "input" and has_cols:
                raise ValueError(f"op {self.id}: axis=input cannot restrict cols")
        else:
            if self.axis is not None or self.normalize:
                raise ValueError(f"op {self.id}: axis/normalize are only valid for project")
        if self.op == "replace_rows" and (not has_rows or has_cols):
            raise ValueError(f"op {self.id}: replace_rows needs rows and no cols")
        if self.op == "replace_cols" and (not has_cols or has_rows):
            raise ValueError(f"op {self.id}: replace_cols needs cols and no rows")
        return self


class ExportConfig(_Strict):
    output_dir: StrictStr = Field(..., min_length=1)
    sidecar_files: list[StrictStr] = Field(default_factory=lambda: ["*.json", "*.model", "*.jinja", "*.txt", "*.tiktoken", "*.py"])
    max_sidecar_bytes: StrictInt = Field(default=64 * 1024 * 1024, ge=1)


class WeightPatchSpec(_Strict):
    schema_version: Literal["mechinterp-weight-patch/v1"]
    description: Optional[StrictStr] = None
    base: BaseBinding
    budget: Budget
    payload: PayloadRef
    operations: list[Operation] = Field(..., min_length=1, max_length=256)
    export: Optional[ExportConfig] = None

    @model_validator(mode="after")
    def _check(self) -> "WeightPatchSpec":
        ids = [op.id for op in self.operations]
        if len(ids) != len(set(ids)):
            raise ValueError("operation ids must be unique")
        targeted = []
        for op in self.operations:
            if op.tensor not in self.base.tensors:
                raise ValueError(
                    f"op {op.id}: tensor {op.tensor!r} has no entry in base.tensors"
                )
            if not any(fnmatch.fnmatchcase(op.tensor, pat) for pat in self.budget.allowed_tensors):
                raise ValueError(
                    f"op {op.id}: tensor {op.tensor!r} is not matched by budget.allowed_tensors"
                )
            if op.tensor not in targeted:
                targeted.append(op.tensor)
        if len(targeted) > self.budget.max_modified_tensors:
            raise ValueError(
                f"{len(targeted)} targeted tensors exceed budget.max_modified_tensors"
            )
        unused = set(self.base.tensors) - set(targeted)
        if unused:
            raise ValueError(
                f"base.tensors lists tensors no operation targets: {sorted(unused)[:3]}"
            )
        return self

    def targeted_tensors(self) -> list[str]:
        out: list[str] = []
        for op in self.operations:
            if op.tensor not in out:
                out.append(op.tensor)
        return out


class VectorFile(_Strict):
    """A 1-D float vector stored in a safetensors file (hash-pinned)."""

    path: StrictStr = Field(..., min_length=1)
    key: StrictStr = Field(..., min_length=1)
    sha256: Hex64


class DirectionFile(_Strict):
    """A ``mechinterp-direction/v1`` JSON (its ``vector`` field), hash-pinned."""

    path: StrictStr = Field(..., min_length=1)
    sha256: Hex64


class VectorSource(_Strict):
    """Exactly one of: inline numbers, a safetensors vector, a direction JSON."""

    inline: Optional[list[FiniteFloat]] = Field(default=None, min_length=1, max_length=65536)
    safetensors: Optional[VectorFile] = None
    direction_json: Optional[DirectionFile] = None

    @model_validator(mode="after")
    def _one(self) -> "VectorSource":
        given = [
            n
            for n in ("inline", "safetensors", "direction_json")
            if getattr(self, n) is not None
        ]
        if len(given) != 1:
            raise ValueError("vector needs exactly one of inline / safetensors / direction_json")
        return self


class ProjectionRequest(_Strict):
    """A bounded numeric read of one 2-D tensor ``W``.

    ``axis`` picks which lines are selected: ``cols`` -> columns ``W[:, j]`` (e.g.
    the output vector one hidden unit writes), ``rows`` -> rows ``W[i, :]``.
    ``indices`` selects lines (omitted = every line). ``other_indices`` restricts
    positions along each line (omitted = the whole line).

    kind ``dot``     value per selected line: ``line . vector``
         ``cosine``  ``line . vector / (|line| |vector|)`` (null for a zero line)
         ``slice``   the raw values, ``[lines, positions]`` (no vector)

    All arithmetic is float64 on the stored values; output size is capped by the
    request's ``max_output_values``.
    """

    id: StrictStr = Field(..., min_length=1, max_length=128)
    tensor: StrictStr = Field(..., min_length=1, max_length=512)
    kind: Literal["dot", "cosine", "slice"] = "dot"
    axis: Literal["cols", "rows"]
    indices: Optional[list[StrictInt]] = Field(default=None, min_length=1)
    other_indices: Optional[list[StrictInt]] = Field(default=None, min_length=1)
    vector: Optional[VectorSource] = None
    include_norms: bool = False

    @model_validator(mode="after")
    def _check(self) -> "ProjectionRequest":
        if self.kind == "slice":
            if self.vector is not None or self.include_norms:
                raise ValueError(f"projection {self.id}: slice takes no vector/include_norms")
        elif self.vector is None:
            raise ValueError(f"projection {self.id}: {self.kind} requires a vector")
        return self


class InspectRequest(_Strict):
    schema_version: Literal["mechinterp-weights-inspect/v1"]
    checkpoint: StrictStr = Field(..., min_length=1)
    allow_symlinked_files: bool = False
    hash_files: bool = False
    tensors: list[StrictStr] = Field(
        default_factory=list, description="fnmatch patterns of tensors to list"
    )
    hash_tensors: bool = False
    max_listed_tensors: StrictInt = Field(default=512, ge=1, le=100_000)
    projections: list[ProjectionRequest] = Field(default_factory=list, max_length=64)
    max_output_values: StrictInt = Field(
        default=16384,
        ge=1,
        le=1_000_000,
        description="Total numeric values all projections may emit.",
    )
    max_tensor_bytes: StrictInt = Field(
        default=2 * 1024**3,
        ge=1,
        le=16 * 1024**3,
        description="Largest single tensor a projection may read into memory.",
    )

    @model_validator(mode="after")
    def _unique_projection_ids(self) -> "InspectRequest":
        ids = [p.id for p in self.projections]
        if len(ids) != len(set(ids)):
            raise ValueError("projection ids must be unique")
        return self


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


class _DupSafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    keys = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in keys:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        keys.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_DupSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def read_document(path: str | Path) -> tuple[dict, str]:
    """Read a YAML/JSON document (bounded, duplicate-key-safe).

    Returns ``(data, sha256_of_file_bytes)``.
    """
    p = Path(path)
    try:
        with open(p, "rb") as fh:
            raw = fh.read(_MAX_SPEC_BYTES + 1)
    except OSError as exc:
        raise WeightPatchError("path", f"cannot read {p.name}: {exc.strerror}") from None
    if len(raw) > _MAX_SPEC_BYTES:
        raise WeightPatchError("schema", f"{p.name} exceeds {_MAX_SPEC_BYTES} bytes")
    try:
        data = yaml.load(raw.decode("utf-8"), Loader=_DupSafeLoader)  # noqa: S506 - safe subclass
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise WeightPatchError("schema", f"{p.name}: invalid document: {exc}") from None
    if not isinstance(data, dict):
        raise WeightPatchError("schema", f"{p.name}: document must be a mapping")
    return data, hashlib.sha256(raw).hexdigest()


def _wrap_validation(model, data, label: str):
    from pydantic import ValidationError

    try:
        return model(**data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in exc.errors()[:8]
        )
        raise WeightPatchError("schema", f"{label}: {details}") from None


def parse_patch_spec(data: dict) -> WeightPatchSpec:
    return _wrap_validation(WeightPatchSpec, data, "weight patch spec")


def parse_inspect_request(data: dict) -> InspectRequest:
    return _wrap_validation(InspectRequest, data, "weights inspect request")


def canonical_sha256(model: BaseModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_patch_spec(
    path: str | Path, *, context=None
) -> tuple[WeightPatchSpec, str]:
    """Load a patch spec; resolve its paths through the project context if given.

    Returns ``(spec, spec_file_sha256)``. The hash is of the on-disk bytes, which
    is what ``surface.weight_patch.sha256`` pins.
    """
    declaring = Path(path).resolve()
    data, digest = read_document(declaring)
    spec = parse_patch_spec(data)
    if context is not None:
        spec = resolve_spec_paths(spec, context, declaring)
    else:
        spec = _resolve_plain(spec, declaring)
    return spec, digest


def _resolve_plain(spec: WeightPatchSpec, declaring_file: Path) -> WeightPatchSpec:
    """Without a project context, relative paths are relative to the spec file."""

    def res(value: str) -> str:
        p = Path(value)
        return str(p if p.is_absolute() else (declaring_file.parent / p).resolve(strict=False))

    update: dict = {"payload": spec.payload.model_copy(update={"path": res(spec.payload.path)})}
    if spec.base.checkpoint:
        update["base"] = spec.base.model_copy(update={"checkpoint": res(spec.base.checkpoint)})
    if spec.export is not None:
        update["export"] = spec.export.model_copy(
            update={"output_dir": res(spec.export.output_dir)}
        )
    return spec.model_copy(update=update)


def _resolve_in_document(value: str, context, declaring_file: Path, access: str = "read") -> str:
    """Resolve a path declared inside a weights document.

    Scheme references (``project://``, ``config://``...) and strict project mode
    follow the project path rules. A plain relative path in legacy (standalone)
    mode is relative to the declaring document -- the same meaning it has in
    strict mode -- rather than to the process's working directory.
    """
    from tuner.project import PathRef, resolve_path

    if context.path_mode == "legacy" and PathRef.parse(value).scheme is None:
        p = Path(value)
        return str(p if p.is_absolute() else (declaring_file.parent / p).resolve(strict=False))
    return str(resolve_path(value, context, declaring_file=declaring_file, access=access))


def resolve_spec_paths(spec: WeightPatchSpec, context, declaring_file: Path) -> WeightPatchSpec:
    def res(value: str, access: str = "read") -> str:
        return _resolve_in_document(value, context, declaring_file, access)

    base_update = {}
    if spec.base.checkpoint:
        base_update["checkpoint"] = res(spec.base.checkpoint)
    update: dict = {
        "base": spec.base.model_copy(update=base_update),
        "payload": spec.payload.model_copy(update={"path": res(spec.payload.path)}),
    }
    if spec.export is not None:
        update["export"] = spec.export.model_copy(
            update={"output_dir": res(spec.export.output_dir, "write")}
        )
    return spec.model_copy(update=update)


def load_inspect_request(path: str | Path, *, context=None) -> InspectRequest:
    declaring = Path(path).resolve()
    data, _ = read_document(declaring)
    req = parse_inspect_request(data)
    def res(value: str) -> str:
        if context is not None:
            return _resolve_in_document(value, context, declaring)
        p = Path(value)
        return str(p if p.is_absolute() else (declaring.parent / p).resolve(strict=False))

    projections = []
    for proj in req.projections:
        if proj.vector is not None and proj.vector.safetensors is not None:
            vf = proj.vector.safetensors.model_copy(update={"path": res(proj.vector.safetensors.path)})
            proj = proj.model_copy(update={"vector": proj.vector.model_copy(update={"safetensors": vf})})
        elif proj.vector is not None and proj.vector.direction_json is not None:
            df = proj.vector.direction_json.model_copy(
                update={"path": res(proj.vector.direction_json.path)}
            )
            proj = proj.model_copy(update={"vector": proj.vector.model_copy(update={"direction_json": df})})
        projections.append(proj)
    return req.model_copy(update={"checkpoint": res(req.checkpoint), "projections": projections})
