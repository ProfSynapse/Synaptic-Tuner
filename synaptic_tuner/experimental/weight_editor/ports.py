"""Trusted ports: ModelAdapter and Executor.

Both are supplied by the host operator (CLI ``--ports module:factory`` or the
Python API). They are never selected by a model request or by the run config.
The broker, not the port, owns admission, budgets, write-ahead and the journal.
"""

from __future__ import annotations

import importlib
import inspect
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from .canon import sha256_hex
from .errors import PortError, WriteBoundaryViolation
from .config import ID_RE

# --------------------------------------------------------------- data types


@dataclass(frozen=True)
class TransportEvent:
    """Exact bytes on a named client stream; never a synthetic provider event.

    ``provider`` is parsed by the adapter. Client input and diagnostics are
    retained for audit without granting them request or final-choice authority.
    """

    data: bytes
    channel: str = "provider"


@dataclass(frozen=True)
class TurnInput:
    run_id: str
    turn_id: str
    turn_index: int
    session_id: str | None
    messages: tuple[Mapping[str, str], ...]  # exact model-visible messages for this turn
    context_digest: str  # chain over every message ever delivered
    resumed: bool  # True when the previous turn was interrupted/failed


EVENT_KINDS = (
    "message",
    "reasoning_summary",
    "request",
    "final_choice",
    "session",
    "usage",
    "finish",
    "handoff",
    "truncation",
    "unknown",
)


@dataclass(frozen=True)
class Receipt:
    status: str  # "ok" is a valid, selectable outcome; anything else is recorded but not selectable
    output: str  # text shown to the model verbatim
    detail: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, Any] = field(default_factory=dict)  # actual resource use, caller-defined names


@dataclass(frozen=True)
class Dispatch:
    dispatch_id: str
    run_id: str
    request_id: str
    operation: str
    arguments: Mapping[str, Any]  # validated model values
    bound: Mapping[str, Any]  # arg name -> {"id","kind","sha256","path"} resolved by the broker
    attempt: int


@dataclass(frozen=True)
class ReconcileResult:
    status: str  # "complete" | "unstarted" | "unknown"
    receipt: Receipt | None = None
    authority: Mapping[str, Any] | None = None  # for "unstarted": {"dispatch_id","basis"}
    note: str = ""


# -------------------------------------------------------------------- ports


class ModelAdapter(ABC):
    @abstractmethod
    def identity(self) -> dict:
        """Requested model identity and declared capabilities.

        Must contain ``provider``, ``model``, ``settings`` equal to the config's
        model block. May contain ``client_version`` and ``capabilities``
        (``raw_events`` must be true; ``captures_hidden_reasoning`` must not be).
        """

    @abstractmethod
    def run_turn(self, turn: TurnInput) -> Iterator[bytes | TransportEvent]:
        """Yield exact provider bytes, optionally with named client streams.

        Channels are ordered as observed by the adapter; no total ordering
        between independent operating-system pipes is claimed.
        """

    @abstractmethod
    def parse_event(self, raw: bytes) -> dict:
        """Normalize one raw event to ``{"kind": <EVENT_KINDS>, ...}``.

        Raise or return an unrecognized kind for anything unsupported; the
        broker retains the raw bytes and marks capture incomplete.
        """

    def close(self) -> None:  # pragma: no cover - optional
        return None


class BoundedWriter:
    """Create-only artifact writer confined to one dispatch's namespace.

    This is a cooperative host API, not OS isolation: an executor that calls
    ``open`` itself is outside this guarantee.
    """

    def __init__(self, root: Path, read_only_roots: tuple[Path, ...]):
        self._root = root
        self._ro = read_only_roots
        self.written: list[dict] = []

    @property
    def root(self) -> Path:
        return self._root

    def write(self, name: str, data: bytes) -> dict:
        if not isinstance(name, str) or not ID_RE.match(name) or name in (".", ".."):
            raise WriteBoundaryViolation(f"artifact name {name!r} is not a bare artifact name")
        if not isinstance(data, (bytes, bytearray)):
            raise WriteBoundaryViolation("artifact data must be bytes")
        self._root.mkdir(parents=True, exist_ok=True)
        resolved_root = self._root.resolve()
        target = resolved_root / name
        if target.parent != resolved_root or any(resolved_root.is_relative_to(r) for r in self._ro):
            raise WriteBoundaryViolation("artifact target outside the run artifact namespace")
        fd = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            view = memoryview(data)
            while view:
                n = os.write(fd, view)
                if n <= 0:
                    raise OSError("artifact write made no progress")
                view = view[n:]
            os.fsync(fd)
        finally:
            os.close(fd)
        rec = {"name": name, "sha256": sha256_hex(bytes(data)), "size": len(data)}
        self.written.append(rec)
        return rec


class Executor(ABC):
    @abstractmethod
    def identity(self) -> dict: ...

    @abstractmethod
    def supported_operations(self) -> frozenset[str]:
        """Operations this executor implements; a configured op outside it is rejected."""

    @abstractmethod
    def validate(self, operation: str, arguments: Mapping[str, Any], bound: Mapping[str, Any]) -> None:
        """Side-effect-free check. Raise ``ExecutorRejection`` to deny."""

    @abstractmethod
    def execute(self, dispatch: Dispatch, writer: BoundedWriter) -> Receipt:
        """Perform the effect. Must be keyed by ``dispatch.dispatch_id``."""

    @abstractmethod
    def reconcile(self, dispatch: Dispatch) -> ReconcileResult:
        """Report what happened to ``dispatch_id`` without causing a new effect."""

    def close(self) -> None:  # pragma: no cover - optional
        return None


@dataclass
class PortSet:
    adapter: ModelAdapter
    executor: Executor


@dataclass(frozen=True)
class PortContext:
    args: Mapping[str, str]
    model: Mapping[str, Any]  # the config's requested model block (no fallback is derived from it)


# ----------------------------------------------------------------- identity


def source_package(path: str | None) -> dict:
    if path is None or not Path(path).is_file():
        raise PortError("port source must be an inspectable Python package", code="unbound_port_source")
    root = Path(path).resolve().parent
    if not (root / "__init__.py").is_file():
        raise PortError("plain-module ports require packaging before use", code="unbound_port_source")
    while (root.parent / "__init__.py").is_file():
        root = root.parent
    sources = {}
    for file in sorted(root.rglob("*.py")):
        if file.is_symlink():
            raise PortError("port source tree contains a symlink")
        sources[str(file.relative_to(root))] = sha256_hex(file.read_bytes())
    return sources


def describe_port(obj: Any) -> dict:
    cls = type(obj)
    try:
        path = inspect.getsourcefile(cls)
    except (OSError, TypeError) as exc:
        raise PortError("port source cannot be inspected", code="unbound_port_source") from exc
    package_sources = source_package(path)
    return {
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "source_sha256": sha256_hex(Path(path).read_bytes()),
        "package_sources": package_sources,
        "source_scope": "Python package tree; external binaries, dependencies and resources need explicit pins",
        "declared": obj.identity(),
    }


def load_ports(spec: str, ctx: PortContext) -> PortSet:
    """Operator-trusted factory loading. ``spec`` is ``module:attribute``."""
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise PortError("ports spec must be module:factory")
    mod_name, attr = spec.split(":")
    try:
        factory = getattr(importlib.import_module(mod_name), attr)
    except (ImportError, AttributeError) as exc:
        raise PortError(f"cannot load ports factory {spec}: {exc}") from exc
    ports = factory(ctx)
    if not isinstance(ports, PortSet) or not isinstance(ports.adapter, ModelAdapter) or not isinstance(
        ports.executor, Executor
    ):
        raise PortError("ports factory must return PortSet(ModelAdapter, Executor)")
    return ports
