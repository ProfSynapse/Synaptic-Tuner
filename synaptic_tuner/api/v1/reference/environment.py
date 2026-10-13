"""Bounded input admission for the disabled C1 environment execution seam.

Only selected files are committed. Catalog discovery is freshly checked for
ambiguity; unselected contents are not committed. Captured bytes are retained
from the same reads that derive identity. Execution remains disabled; later
adapters must consume these bytes rather than reopen the original paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import math
from contextlib import ExitStack
from pathlib import Path
from dataclasses import dataclass, field

from jsonschema import Draft202012Validator
from referencing import Registry

from ..data_facade import DataMode, DataOperationCode, DataOperationError, DataScenarioTarget
from .._contract import digest_text
from ..environment_execution import (
    EnvironmentBindingV1, EnvironmentDependencyV1, MAX_DEPENDENCY_BYTES,
    MAX_DEPENDENCIES, MAX_DEPENDENCY_TOTAL_BYTES, logical_ref,
)


def _invalid():
    return DataOperationError(DataOperationCode.ENVIRONMENT_INVALID)


@dataclass(frozen=True, slots=True)
class AdmittedEnvironmentInputs:
    """Private, process-local input custody; never a session or restart grant.

    No raw bytes appear in public plans or durable records. Only fresh
    re-admission may supply this capture; it is not reconstructed from a plan.
    """

    targets: tuple[DataScenarioTarget, ...]
    rows_requested: int
    source_digest: str
    binding: EnvironmentBindingV1
    documents: tuple[tuple[str, str, bytes], ...] = field(repr=False)
    _binding_digest: str = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.binding) is not EnvironmentBindingV1:
            raise TypeError("exact environment binding required")
        binding = EnvironmentBindingV1.from_dict(self.binding.to_dict())
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "_binding_digest", binding.binding_digest)
        if type(self.targets) is not tuple or any(type(item) is not DataScenarioTarget for item in self.targets):
            raise TypeError("exact scenario targets required")
        object.__setattr__(self, "targets", tuple(DataScenarioTarget.from_dict(item.to_dict()) for item in self.targets))
        if (not self.targets or type(self.rows_requested) is not int or
                not 1 <= self.rows_requested <= binding.bounds.max_rows or
                self.rows_requested != sum(item.rows for item in self.targets)):
            raise ValueError("rows must match captured targets")
        digest_text(self.source_digest, "source_digest")
        self._check_documents()

    def _check_documents(self):
        if self.binding.binding_digest != self._binding_digest:
            raise _invalid()
        if type(self.documents) is not tuple or len(self.documents) != len(self.binding.dependencies):
            raise _invalid()
        expected = {dependency.ref: dependency for dependency in self.binding.dependencies}
        seen = set()
        for entry in self.documents:
            if type(entry) is not tuple or len(entry) != 3:
                raise _invalid()
            role, ref, raw = entry
            if type(role) is not str or type(ref) is not str or type(raw) is not bytes:
                raise _invalid()
            dependency = expected.get(ref)
            if dependency is None or ref in seen or role != dependency.role:
                raise _invalid()
            if len(raw) != dependency.size_bytes or hashlib.sha256(raw).hexdigest() != dependency.sha256:
                raise _invalid()
            seen.add(ref)

    def document_bytes(self, role: str, ref: str) -> bytes:
        self._check_documents()
        for item_role, item_ref, raw in self.documents:
            if (item_role, item_ref) == (role, ref):
                return raw
        raise _invalid()

    def workflow_documents(self) -> tuple[tuple[str, str, bytes], ...]:
        """The exact common WorkflowInputSnapshot input, excluding host/dataset bytes."""
        self._check_documents()
        roles = {"validation", "privacy", "registry", "scenario", "rubric"}
        return tuple(entry for entry in self.documents if entry[0] in roles)


def _read(path: Path) -> bytes:
    """Read a bounded regular leaf while retaining and checking its ancestors."""
    path = Path(os.path.abspath(path))
    with ExitStack() as stack:
        parent = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, parent)
        guards = []
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            stack.callback(os.close, child)
            guards.append((parent, component, os.fstat(child)))
            parent = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        stack.callback(os.close, fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_DEPENDENCY_BYTES:
            raise _invalid()
        chunks = bytearray()
        while len(chunks) <= MAX_DEPENDENCY_BYTES:
            chunk = os.read(fd, min(65536, MAX_DEPENDENCY_BYTES + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
        after = os.fstat(fd)
        named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        if identity(before) != identity(after) or identity(before) != identity(named) or len(chunks) != before.st_size:
            raise _invalid()
        for ancestor, component, original in guards:
            current = os.stat(component, dir_fd=ancestor, follow_symlinks=False)
            if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
                raise _invalid()
        return bytes(chunks)


def _tree(value, depth=0, budget=None, active=None):
    budget = [20000] if budget is None else budget
    active = set() if active is None else active
    budget[0] -= 1
    if budget[0] < 0 or depth > 32:
        raise _invalid()
    if type(value) in (dict, list):
        if id(value) in active:
            raise _invalid()
        active.add(id(value))
        items = value.values() if type(value) is dict else value
        for item in items:
            _tree(item, depth + 1, budget, active)
        active.remove(id(value))
    elif value is not None and type(value) not in (str, int, float, bool):
        raise _invalid()
    elif type(value) is float and not math.isfinite(value):
        raise _invalid()


def _yaml(raw):
    # The same pure parser will consume retained snapshot documents. This
    # import remains lazy so composing the reference does not load SynthChat.
    from SynthChat.utils.yaml_loader import parse_workflow_yaml
    return parse_workflow_yaml(raw)


def _dependency(role, ref, raw):
    return EnvironmentDependencyV1(role, ref, hashlib.sha256(raw).hexdigest(), len(raw))


def _scan(directory):
    paths = []
    with os.scandir(directory) as entries:
        for count, entry in enumerate(entries):
            if count >= MAX_DEPENDENCIES:
                raise _invalid()
            if entry.name.endswith(".yaml"):
                paths.append(directory / entry.name)
    paths.sort()
    total = 0
    for path in paths:
        raw = _read(path)
        total += len(raw)
        if total > MAX_DEPENDENCY_TOTAL_BYTES:
            raise _invalid()
        yield path, raw, _yaml(raw)


def _no_external_refs(document):
    def visit(value):
        if type(value) is dict:
            for key, item in value.items():
                if key in ("$ref", "$dynamicRef") and (type(item) is not str or not item.startswith("#/")):
                    raise _invalid()
                visit(item)
        elif type(value) is list:
            for item in value:
                visit(item)
    visit(document)


def _schema(document):
    Draft202012Validator.check_schema(document)
    if document.get("type") != "object" or document.get("additionalProperties") is not False:
        raise _invalid()
    _no_external_refs(document)
    def no_remote_resource(uri):
        raise _invalid()
    return Draft202012Validator(document, registry=Registry(retrieve=no_remote_resource))


def admit_environment(request, ports):
    """Resolve only; no LLM creation, session opening, claims or writes."""
    try:
        definitions = {}
        selected_files = {}
        if request.mode is DataMode.GENERATE:
            selected = {target.scenario_ref for target in request.scenarios}
            for path, raw, document in _scan(ports.scenarios_dir):
                scenarios = document.get("scenarios", {})
                if type(scenarios) is not dict:
                    raise _invalid()
                for ref in selected & scenarios.keys():
                    if ref in definitions or type(scenarios[ref]) is not dict:
                        raise _invalid()
                    definitions[ref] = scenarios[ref]
                    selected_files[path] = raw
            if set(definitions) != selected:
                raise _invalid()
        for definition in definitions.values():
            if "environment_execution" in definition and (
                type(definition["environment_execution"]) is not str or
                definition["environment_execution"] not in ("disabled", "required")
            ):
                raise _invalid()
            environment = definition.get("environment", {})
            disabled = definition.get("environment_execution") == "disabled" or (
                type(environment) is dict and environment.get("enabled") is False
            )
            if disabled and (
                request.environment_selection is not None or definition.get("environment_execution") == "required"
            ):
                raise _invalid()
        if request.environment_selection is None:
            for definition in definitions.values():
                environment = definition.get("environment", {})
                declaration = definition.get("environment_execution")
                if declaration is not None and declaration not in ("disabled", "required"):
                    raise _invalid()
                nonexecuting = declaration == "disabled" or (
                    type(environment) is dict and environment.get("enabled") is False
                )
                if declaration == "required" or (
                    not nonexecuting and (environment or definition.get("environment_generation"))
                ):
                    raise DataOperationError(DataOperationCode.ENVIRONMENT_REQUIRED)
            return None
        port = ports.environments
        if port is None or not callable(getattr(port, "resolve", None)) or not callable(getattr(port, "read_dependency", None)):
            raise DataOperationError(DataOperationCode.ENVIRONMENT_REQUIRED)
        binding = port.resolve(request.environment_selection, project_ref=request.project_ref)
        if type(binding) is not EnvironmentBindingV1 or binding.selection != request.environment_selection:
            raise _invalid()
        binding = EnvironmentBindingV1.from_dict(binding.to_dict())
        if request.max_iterations > binding.bounds.max_turns:
            raise _invalid()
        host = {}
        captured_documents = []
        for dependency in binding.dependencies:
            if dependency.role not in {"tool_schema", "execution_policy", "fixture_schema"} or dependency.role in host:
                raise _invalid()
            raw = port.read_dependency(dependency.ref, maximum_bytes=dependency.size_bytes)
            if type(raw) is not bytes or len(raw) != dependency.size_bytes or hashlib.sha256(raw).hexdigest() != dependency.sha256:
                raise _invalid()
            document = _yaml(raw)
            if not document:
                raise _invalid()
            _no_external_refs(document)
            host[dependency.role] = document
            captured_documents.append((dependency.role, dependency.ref, raw))
        if set(host) != {"tool_schema", "execution_policy", "fixture_schema"}:
            raise _invalid()
        policy = {
            "backend_ref": binding.backend_ref, "backend_revision": binding.backend_revision,
            "scope_ref": binding.scope_ref, "mode": binding.selection.mode.value,
            "bounds": binding.bounds.to_dict(), "allowed_actions": list(binding.allowed_actions),
        }
        if host["execution_policy"] != policy:
            raise _invalid()
        fixture_validator = _schema(host["fixture_schema"])
        for definition in definitions.values():
            system = definition.get("system")
            if type(system) is dict and "template" in system:
                # The legacy generator opens this arbitrary file directly.
                # C2 must add snapshot-aware consumption before admitting it.
                raise _invalid()
            _no_external_refs(definition)
            if definition.get("environment_execution") == "required" and binding.selection.mode.value != "execute":
                raise _invalid()
            configuration = definition.get("environment", {})
            if type(configuration) is not dict:
                raise _invalid()
            # These are host policy, never fixture or scenario data.
            if set(configuration) & set(policy):
                raise _invalid()
            fixture_validator.validate(configuration)
        dependencies = list(binding.dependencies)
        def capture(role, ref, raw):
            dependencies.append(_dependency(role, ref, raw))
            captured_documents.append((role, ref, raw))
        for path, raw in selected_files.items():
            logical_ref(path.stem, "scenario file")
            capture("scenario", "scenario:" + path.stem, raw)
        rubric_refs = set(request.rubric_refs)
        engine = request.mode is DataMode.IMPROVE or request.max_iterations > 1
        if engine:
            for definition in definitions.values():
                configured = definition.get("rubrics", {})
                if type(configured) is not dict:
                    raise _invalid()
                for refs in configured.values():
                    if type(refs) is not list or any(type(ref) is not str for ref in refs):
                        raise _invalid()
                    rubric_refs.update(refs)
            raw = _read(ports.config_dir / "validation.yaml")
            capture("validation", "config:validation", raw)
            _no_external_refs(_yaml(raw))
            discovered = {path.stem: (raw, document) for path, raw, document in _scan(ports.rubrics_dir) if path.name != "quality_labels.yaml"}
            for ref in sorted(rubric_refs):
                logical_ref(ref, "rubric")
                if ref not in discovered:
                    raise _invalid()
                raw, document = discovered[ref]
                _no_external_refs(document)
                capture("rubric", "rubric:" + ref, raw)
        if request.mode is DataMode.GENERATE:
            from SynthChat.config.format_resolver import _CONFIG_DIR
            for name in ("tool_call_formats", "workspace_formats", "label_mappings"):
                raw = _read(_CONFIG_DIR / (name + ".yaml"))
                _no_external_refs(_yaml(raw))
                capture("registry", "registry:" + name, raw)
            raw = _read(ports.config_dir / "privacy_profiles.yaml")
            _no_external_refs(_yaml(raw))
            capture("privacy", "config:privacy_profiles", raw)
        else:
            raw = _read(Path(request.input_dataset_ref))
            capture("dataset", "input:dataset", raw)
            dataset_rows = []
            for line in raw.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                _tree(row)
                if type(row) is not dict or "_meta" in row:
                    raise _invalid()
                dataset_rows.append(row)
                if len(dataset_rows) > binding.bounds.max_rows:
                    raise _invalid()
            if not dataset_rows:
                raise _invalid()
        resolved = EnvironmentBindingV1(
            binding.schema_version, binding.selection, binding.backend_ref, binding.backend_revision,
            binding.scope_ref, binding.bounds, binding.allowed_actions, tuple(dependencies),
        )
        # Legacy-compatible source identity, derived from these admitted reads.
        from tuner.execution.foundation_v2.canonical import domain_digest, canonical_bytes
        if request.mode is DataMode.GENERATE:
            targets = request.scenarios
            source = json.dumps(definitions, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        else:
            targets = (DataScenarioTarget(request.input_dataset_ref, len(dataset_rows)),)
            source = canonical_bytes({"input_sha256": hashlib.sha256(raw).hexdigest(), "rubric_refs": list(request.rubric_refs)})
        rows = sum(target.rows for target in targets)
        if rows > binding.bounds.max_rows:
            raise _invalid()
        return AdmittedEnvironmentInputs(
            targets, rows, domain_digest("synaptic-reference-data-source/v1", source), resolved,
            tuple(sorted(captured_documents, key=lambda entry: entry[1])),
        )
    except DataOperationError:
        raise
    except Exception:
        raise _invalid() from None
