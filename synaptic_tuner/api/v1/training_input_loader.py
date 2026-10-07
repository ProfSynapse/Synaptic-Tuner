"""Cold, provider-neutral loader for the canonical training-input contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Condition, get_ident
from types import ModuleType
from typing import Callable, NoReturn, final

from .training_input import TrainingInputV1


_IDENTITY_SCHEMA = "synaptic-training-input-contract-identity/v1"
_CONTRACT_SCHEMA = "synaptic-training-input/v1"
_MODULE_NAME = "synaptic_tuner.api.v1.training_input"
_TYPE_NAME = "TrainingInputV1"
_PARSER_NAME = "from_json"
_IMPLEMENTATION_DOMAIN = b"synaptic-training-input-implementation/v1\0"
_IDENTITY_DOMAIN = b"synaptic-training-input-contract-identity/v1\0"
_CLOSURE_MODULE_IDS = (
    "synaptic_tuner.api.v1._contract",
    "synaptic_tuner.api.v1.training_input",
    "synaptic_tuner.api.v1.training_input_loader",
)
_MAX_MEMBER_BYTES = 256 * 1024
_MAX_CLOSURE_BYTES = 1024 * 1024
_HEX_DIGEST = frozenset("0123456789abcdef")
_TRAINING_INPUT_TYPE = TrainingInputV1
_TRAINING_INPUT_PARSER = TrainingInputV1.from_json


@final
class TrainingInputContractCodeV1(str, Enum):
    CONTRACT_UNAVAILABLE = "contract_unavailable"
    LOAD_REENTRANT = "load_reentrant"
    LOAD_INTERRUPTED = "load_interrupted"
    INPUT_INVALID = "input_invalid"


@final
class TrainingInputContractErrorV1(Exception):
    __slots__ = ("code",)

    def __init__(self, code: TrainingInputContractCodeV1) -> None:
        if type(code) is not TrainingInputContractCodeV1:
            raise TypeError("training input contract error code is invalid")
        Exception.__init__(self, code.value)
        object.__setattr__(self, "code", code)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("training input contract errors are immutable")


def _fresh_error(code: TrainingInputContractCodeV1) -> TrainingInputContractErrorV1:
    return TrainingInputContractErrorV1(code)


def _canonical_bytes(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _is_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _HEX_DIGEST for character in value)
    )


def _identity_digest(body: dict[str, object]) -> str:
    return hashlib.sha256(_IDENTITY_DOMAIN + _canonical_bytes(body)).hexdigest()


def _identity_body(implementation_digest: str) -> dict[str, object]:
    return {
        "schema_version": _IDENTITY_SCHEMA,
        "contract_schema": _CONTRACT_SCHEMA,
        "module_name": _MODULE_NAME,
        "type_name": _TYPE_NAME,
        "parser_name": _PARSER_NAME,
        "implementation_digest": implementation_digest,
    }


@dataclass(frozen=True, slots=True)
class TrainingInputContractIdentityV1:
    schema_version: str
    contract_schema: str
    module_name: str
    type_name: str
    parser_name: str
    implementation_digest: str
    identity_digest: str

    def __post_init__(self) -> None:
        expected = (
            (self.schema_version, _IDENTITY_SCHEMA),
            (self.contract_schema, _CONTRACT_SCHEMA),
            (self.module_name, _MODULE_NAME),
            (self.type_name, _TYPE_NAME),
            (self.parser_name, _PARSER_NAME),
        )
        if any(type(value) is not str for value, _ in expected):
            raise TypeError("training input contract identity is invalid")
        if any(value != required for value, required in expected):
            raise ValueError("training input contract identity is invalid")
        if not _is_digest(self.implementation_digest):
            raise ValueError("training input implementation digest is invalid")
        if not _is_digest(self.identity_digest):
            raise ValueError("training input identity digest is invalid")
        if self.identity_digest != _identity_digest(
            _identity_body(self.implementation_digest)
        ):
            raise ValueError("training input identity digest is invalid")


@final
class LoadedTrainingInputContractV1:
    __slots__ = ("identity", "input_type", "__parser")

    identity: TrainingInputContractIdentityV1
    input_type: type[TrainingInputV1]

    def __init__(
        self,
        identity: TrainingInputContractIdentityV1,
        input_type: type[TrainingInputV1],
    ) -> None:
        if type(identity) is not TrainingInputContractIdentityV1:
            raise TypeError("training input contract identity is invalid")
        if input_type is not _TRAINING_INPUT_TYPE:
            raise TypeError("training input contract type is invalid")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "input_type", input_type)
        object.__setattr__(
            self, "_LoadedTrainingInputContractV1__parser", _TRAINING_INPUT_PARSER
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("loaded training input contracts are immutable")

    def __repr__(self) -> str:
        return (
            "LoadedTrainingInputContractV1("
            f"identity={self.identity!r}, input_type={self.input_type!r})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is LoadedTrainingInputContractV1
            and self.identity == other.identity
            and self.input_type is other.input_type
        )

    def parse_json(self, value: str) -> TrainingInputV1:
        parser = object.__getattribute__(
            self, "_LoadedTrainingInputContractV1__parser"
        )
        outcome = _invoke_training_input_parser(parser, value)
        del value
        del parser
        del self
        if outcome[0]:
            result = outcome[1]
            del outcome
            return result
        del outcome
        _raise_input_invalid()


def _invoke_training_input_parser(
    parser: Callable[[str], TrainingInputV1], value: object,
) -> tuple[bool, TrainingInputV1 | None]:
    if type(value) is not str:
        return False, None
    result: object | None = None
    failed = False
    try:
        result = parser(value)
    except BaseException:
        failed = True
    if failed or type(result) is not _TRAINING_INPUT_TYPE:
        result = None
        return False, None
    return True, result


def _raise_input_invalid() -> NoReturn:
    raise _fresh_error(TrainingInputContractCodeV1.INPUT_INVALID) from None


@dataclass(frozen=True, slots=True)
class _ReadyTrainingInputContractV1:
    module: ModuleType
    origin: Path
    parser: Callable[[str], TrainingInputV1]
    bundle: LoadedTrainingInputContractV1

    def __post_init__(self) -> None:
        if type(self.module) is not ModuleType or not isinstance(self.origin, Path):
            raise TypeError("training input contract is unavailable")
        if self.parser is not _TRAINING_INPUT_PARSER:
            raise TypeError("training input contract is unavailable")
        if type(self.bundle) is not LoadedTrainingInputContractV1:
            raise TypeError("training input contract is unavailable")


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _read_stable_source_inner(
    path: Path, *, _after_read: Callable[[], None] | None = None,
) -> bytes:
    declared = path.lstat()
    if stat.S_ISLNK(declared.st_mode) or not stat.S_ISREG(declared.st_mode):
        raise RuntimeError("training input contract is unavailable")
    if declared.st_size > _MAX_MEMBER_BYTES:
        raise RuntimeError("training input contract is unavailable")
    resolved = path.resolve(strict=True)
    if resolved != path.absolute():
        raise RuntimeError("training input contract is unavailable")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        size = 0
        while size <= _MAX_MEMBER_BYTES:
            chunk = os.read(descriptor, _MAX_MEMBER_BYTES + 1 - size)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        payload = b"".join(chunks)
        if _after_read is not None:
            _after_read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = path.lstat()
    identities = (
        _stat_identity(declared),
        _stat_identity(before),
        _stat_identity(after),
        _stat_identity(final),
    )
    if any(identity != identities[0] for identity in identities[1:]):
        raise RuntimeError("training input contract is unavailable")
    if len(payload) > _MAX_MEMBER_BYTES or len(payload) != before.st_size:
        raise RuntimeError("training input contract is unavailable")
    return payload


def _read_stable_source(
    path: Path, *, _after_read: Callable[[], None] | None = None,
) -> bytes:
    payload: bytes | None = None
    failed = False
    try:
        payload = _read_stable_source_inner(path, _after_read=_after_read)
    except BaseException:
        failed = True
    if failed or type(payload) is not bytes:
        raise _fresh_error(TrainingInputContractCodeV1.CONTRACT_UNAVAILABLE) from None
    return payload


def _source_origin(
    module_id: str,
    find_spec: Callable[[str], object],
) -> Path:
    spec = find_spec(module_id)
    origin = getattr(spec, "origin", None)
    if type(origin) is not str:
        raise RuntimeError("training input contract is unavailable")
    return Path(origin)


def _implementation_digest_v1(
    *,
    _find_spec: Callable[[str], object] = importlib.util.find_spec,
    _source_reader: Callable[[Path], bytes] = _read_stable_source,
) -> str:
    closure: dict[str, object] = {}
    total = 0
    for module_id in _CLOSURE_MODULE_IDS:
        source = _source_reader(_source_origin(module_id, _find_spec))
        total += len(source)
        if total > _MAX_CLOSURE_BYTES:
            raise RuntimeError("training input contract is unavailable")
        closure[module_id] = hashlib.sha256(source).hexdigest()
    return hashlib.sha256(
        _IMPLEMENTATION_DOMAIN + _canonical_bytes(closure)
    ).hexdigest()


def _build_loaded_contract_v1() -> _ReadyTrainingInputContractV1:
    implementation_digest = _implementation_digest_v1()
    body = _identity_body(implementation_digest)
    identity = TrainingInputContractIdentityV1(
        _IDENTITY_SCHEMA,
        _CONTRACT_SCHEMA,
        _MODULE_NAME,
        _TYPE_NAME,
        _PARSER_NAME,
        implementation_digest,
        _identity_digest(body),
    )
    bundle = LoadedTrainingInputContractV1(identity, _TRAINING_INPUT_TYPE)
    module = sys.modules.get(_MODULE_NAME)
    if type(module) is not ModuleType:
        raise _fresh_error(TrainingInputContractCodeV1.CONTRACT_UNAVAILABLE) from None
    origin = _source_origin(_MODULE_NAME, importlib.util.find_spec).resolve(strict=True)
    return _ReadyTrainingInputContractV1(
        module, origin, _TRAINING_INPUT_PARSER, bundle
    )


def _closed_code(value: BaseException) -> TrainingInputContractCodeV1:
    if type(value) is not TrainingInputContractErrorV1:
        return TrainingInputContractCodeV1.CONTRACT_UNAVAILABLE
    try:
        code = object.__getattribute__(value, "code")
    except BaseException:
        return TrainingInputContractCodeV1.CONTRACT_UNAVAILABLE
    if type(code) is not TrainingInputContractCodeV1:
        return TrainingInputContractCodeV1.CONTRACT_UNAVAILABLE
    return code


def _install_contract_loader_v1(
    builder: Callable[[], _ReadyTrainingInputContractV1], *,
    _condition_factory: Callable[[], Condition] = Condition,
) -> Callable[[], LoadedTrainingInputContractV1]:
    condition = _condition_factory()
    uninitialized = object()
    initializing = object()
    ready = object()
    failed = object()
    state = uninitialized
    initializing_thread: int | None = None
    ready_record: _ReadyTrainingInputContractV1 | None = None
    failure_code: TrainingInputContractCodeV1 | None = None

    def load() -> LoadedTrainingInputContractV1:
        nonlocal state, initializing_thread, ready_record, failure_code
        claimed = False
        immediate_code: TrainingInputContractCodeV1 | None = None
        interrupted = False
        while not claimed and immediate_code is None and not interrupted:
            with condition:
                if state is ready:
                    if type(ready_record) is not _ReadyTrainingInputContractV1:
                        immediate_code = TrainingInputContractCodeV1.CONTRACT_UNAVAILABLE
                    else:
                        return ready_record.bundle
                elif state is failed:
                    immediate_code = (
                        failure_code
                        if type(failure_code) is TrainingInputContractCodeV1
                        else TrainingInputContractCodeV1.CONTRACT_UNAVAILABLE
                    )
                elif state is uninitialized:
                    state = initializing
                    initializing_thread = get_ident()
                    claimed = True
                elif initializing_thread == get_ident():
                    immediate_code = TrainingInputContractCodeV1.LOAD_REENTRANT
                else:
                    try:
                        condition.wait()
                    except BaseException:
                        interrupted = True
        if interrupted:
            raise _fresh_error(TrainingInputContractCodeV1.LOAD_INTERRUPTED) from None
        if immediate_code is not None:
            raise _fresh_error(immediate_code) from None

        built: _ReadyTrainingInputContractV1 | None = None
        caught_code: TrainingInputContractCodeV1 | None = None
        caught: BaseException | None = None
        try:
            built = builder()
        except BaseException as error:
            caught = error
        if caught is not None:
            caught_code = _closed_code(caught)
        if type(built) is not _ReadyTrainingInputContractV1:
            caught_code = (
                caught_code or TrainingInputContractCodeV1.CONTRACT_UNAVAILABLE
            )

        with condition:
            initializing_thread = None
            if caught_code is None:
                ready_record = built
                state = ready
            else:
                failure_code = caught_code
                state = failed
            condition.notify_all()
        caught = None
        if caught_code is not None:
            raise _fresh_error(caught_code) from None
        if type(built) is not _ReadyTrainingInputContractV1:  # pragma: no cover
            raise _fresh_error(TrainingInputContractCodeV1.CONTRACT_UNAVAILABLE) from None
        return built.bundle

    return load


load_training_input_contract_v1 = _install_contract_loader_v1(
    _build_loaded_contract_v1
)
load_training_input_contract_v1.__name__ = "load_training_input_contract_v1"
load_training_input_contract_v1.__qualname__ = "load_training_input_contract_v1"
load_training_input_contract_v1.__doc__ = (
    "Load and cache the exact canonical V1 training-input contract."
)


__all__ = [
    "LoadedTrainingInputContractV1",
    "TrainingInputContractCodeV1",
    "TrainingInputContractErrorV1",
    "TrainingInputContractIdentityV1",
    "load_training_input_contract_v1",
]
