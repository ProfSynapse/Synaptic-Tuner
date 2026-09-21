#!/usr/bin/env python3
"""Create and verify privacy-safe, offline tokenizer-profile evidence."""
from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator


PROFILE_SCHEMA = "syntunia-token-profile/v1"
MANIFEST_SCHEMA = "syntunia-token-profile-manifest/v1"
CLOSURE_ALGORITHM = "tokenizer-load-closure/1"
MESSAGE_ROLE_BUCKETS = ("system", "developer", "user", "assistant", "tool")
PROFILE_FILENAMES = ("manifest.json", "profile.json", "profile.csv")
TOKENIZER_REQUIRED_FILES = frozenset({"tokenizer.json", "tokenizer_config.json"})
TOKENIZER_ALLOWED_FILES = frozenset(
    {"added_tokens.json", "chat_template.jinja", "merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "vocab.txt"}
)
CONFIG_MAX_BYTES = 1_048_576
RUNTIME_LOCK_MAX_BYTES = 1_048_576
RUNTIME_SCHEMA_MAX_BYTES = 2_097_152
ARTIFACT_MEMBER_MAX_BYTES = 16_777_216
MODEL_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
COMPONENT_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
HARD_LIMITS = {
    "max_input_bytes": 1_073_741_824,
    "max_rows": 1_000_000,
    "max_line_bytes": 67_108_864,
    "max_record_bytes": 67_108_864,
    "max_tokens_per_record": 10_000_000,
    "max_components": 64,
    "max_component_label_chars": 128,
    "max_snapshot_files": 64,
    "max_snapshot_file_bytes": 536_870_912,
    "max_snapshot_total_bytes": 1_073_741_824,
}


class ProfilerError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class PublicationUncertain(Exception):
    def __init__(self, profile_id: str, phase: str):
        self.profile_id = profile_id
        self.phase = phase
        super().__init__(phase)


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ProfilerError("INVALID_ARGUMENTS")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_record(value: Any, domain: str) -> str:
    return _sha256_bytes((domain + "\0" + _canonical_json(value)).encode("utf-8"))


def _require_mapping(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProfilerError(code)
    return value


def _require_exact_keys(value: dict[str, Any], allowed: set[str], code: str) -> None:
    if set(value) != allowed:
        raise ProfilerError(code)


def _immutable_revision(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ProfilerError("INVALID_IMMUTABLE_REVISION")
    return value


def _digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
        raise ProfilerError(code)
    return value


def _is_reparse(st: os.stat_result) -> bool:
    return bool(getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _identity(st: os.stat_result) -> tuple[int, int, int]:
    return (st.st_dev, st.st_ino, stat.S_IFMT(st.st_mode))


def _stable_identity(st: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (*_identity(st), st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _stable_metadata(st: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (st.st_dev, st.st_ino, st.st_mode, st.st_nlink, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


@dataclass
class _RetainedMember:
    name: str
    descriptor: int
    metadata: tuple[int, int, int, int, int, int, int]
    byte_count: int = -1
    sha256: str = ""


def _identity_digest(st: os.stat_result) -> str:
    return _digest_record({"device": st.st_dev, "inode": st.st_ino, "kind": stat.S_IFMT(st.st_mode)}, "fs-identity/1")


class _DirectoryGuard:
    def __init__(self, path: Path):
        self.path = path
        self.handle: Any = None
        self.identity: tuple[int, int] | tuple[int, int, int]
        self.binding_metadata: tuple[Any, ...]
        self.member_handles: list[_RetainedMember] = []
        self.capture_members: dict[str, _RetainedMember] = {}
        try:
            if os.name == "nt":
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

                class FileInformation(ctypes.Structure):
                    _fields_ = [
                        ("attributes", ctypes.c_uint32),
                        ("creation_low", ctypes.c_uint32),
                        ("creation_high", ctypes.c_uint32),
                        ("access_low", ctypes.c_uint32),
                        ("access_high", ctypes.c_uint32),
                        ("write_low", ctypes.c_uint32),
                        ("write_high", ctypes.c_uint32),
                        ("volume_serial", ctypes.c_uint32),
                        ("size_high", ctypes.c_uint32),
                        ("size_low", ctypes.c_uint32),
                        ("links", ctypes.c_uint32),
                        ("file_index_high", ctypes.c_uint32),
                        ("file_index_low", ctypes.c_uint32),
                    ]

                self._kernel32, self._information_type = kernel32, FileInformation
                create = kernel32.CreateFileW
                create.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
                create.restype = ctypes.c_void_p
                # Deliberately omit FILE_SHARE_DELETE: the authenticated directory
                # cannot be renamed or replaced while this handle is retained.
                handle = create(str(path), 0, 0x1 | 0x2, None, 3, 0x02000000 | 0x00200000, None)
                if handle == ctypes.c_void_p(-1).value:
                    raise OSError(ctypes.get_last_error(), "directory open failed")
                self.handle = handle
                self.identity = self._windows_identity(handle)
                self.binding_metadata = self._windows_metadata(handle)
            else:
                self.handle = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
                opened = os.fstat(self.handle)
                self.identity = _identity(opened)
                self.binding_metadata = _stable_metadata(opened)
        except OSError as error:
            self.close()
            raise ProfilerError("SNAPSHOT_INVALID") from error

    def _windows_identity(self, handle: Any) -> tuple[int, int]:
        information = self._windows_information(handle)
        if information.attributes & 0x400 or not information.attributes & 0x10:
            raise OSError(errno.ELOOP, "redirected directory")
        return (information.volume_serial, (information.file_index_high << 32) | information.file_index_low)

    def _windows_information(self, handle: Any) -> Any:
        information = self._information_type()
        get_info = self._kernel32.GetFileInformationByHandle
        get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(self._information_type)]
        get_info.restype = ctypes.c_int
        if not get_info(handle, ctypes.byref(information)):
            raise OSError(ctypes.get_last_error(), "directory identity failed")
        return information

    def _windows_metadata(self, handle: Any) -> tuple[int, ...]:
        information = self._windows_information(handle)
        return (
            information.attributes,
            information.write_high,
            information.write_low,
            information.size_high,
            information.size_low,
            information.links,
            information.volume_serial,
            information.file_index_high,
            information.file_index_low,
        )

    def _windows_open_relative(self, name: str, *, create: bool) -> int:
        import msvcrt

        class UnicodeString(ctypes.Structure):
            _fields_ = [("Length", ctypes.c_ushort), ("MaximumLength", ctypes.c_ushort), ("Buffer", ctypes.c_wchar_p)]

        class ObjectAttributes(ctypes.Structure):
            _fields_ = [("Length", ctypes.c_ulong), ("RootDirectory", ctypes.c_void_p), ("ObjectName", ctypes.POINTER(UnicodeString)), ("Attributes", ctypes.c_ulong), ("SecurityDescriptor", ctypes.c_void_p), ("SecurityQualityOfService", ctypes.c_void_p)]

        class IoStatusBlock(ctypes.Structure):
            _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t)]

        buffer = ctypes.create_unicode_buffer(name)
        unicode_name = UnicodeString(len(name.encode("utf-16-le")), len(name.encode("utf-16-le")) + 2, ctypes.cast(buffer, ctypes.c_wchar_p))
        attributes = ObjectAttributes(ctypes.sizeof(ObjectAttributes), ctypes.c_void_p(self.handle), ctypes.pointer(unicode_name), 0x40, None, None)
        io_status = IoStatusBlock()
        native_handle = ctypes.c_void_p()
        nt_create = ctypes.WinDLL("ntdll", use_last_error=True).NtCreateFile
        nt_create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_ulong, ctypes.POINTER(ObjectAttributes), ctypes.POINTER(IoStatusBlock), ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong]
        nt_create.restype = ctypes.c_long
        desired_access = (0x40000000 if create else 0x80000000) | 0x00100000 | 0x80
        share_access = 0 if create else 0x1
        disposition = 2 if create else 1
        options = 0x40 | 0x20 | 0x00200000
        status_code = nt_create(ctypes.byref(native_handle), desired_access, ctypes.byref(attributes), ctypes.byref(io_status), None, 0x80, share_access, disposition, options, None, 0)
        if status_code < 0 or not native_handle.value:
            raise OSError(errno.EACCES, "relative open failed")
        try:
            information = self._windows_information(native_handle.value)
            if information.attributes & 0x400 or information.attributes & 0x10:
                raise OSError(errno.ELOOP, "unsafe member")
            flags = (os.O_WRONLY if create else os.O_RDONLY) | getattr(os, "O_BINARY", 0)
            return msvcrt.open_osfhandle(native_handle.value, flags)
        except Exception:
            self._kernel32.CloseHandle(native_handle)
            raise

    def list_names(self) -> list[str]:
        self.verify_binding()
        try:
            names = os.listdir(self.path if os.name == "nt" else self.handle)
        except OSError as error:
            raise ProfilerError("SNAPSHOT_INVALID") from error
        self.verify_binding()
        return names

    def open_member(self, name: str) -> int:
        if name in {".", ".."} or "/" in name or "\\" in name:
            raise ProfilerError("SNAPSHOT_INVENTORY_INVALID")
        try:
            descriptor = self._windows_open_relative(name, create=False) if os.name == "nt" else os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self.handle)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                os.close(descriptor)
                raise ProfilerError("SNAPSHOT_MEMBER_UNSTABLE")
            return descriptor
        except ProfilerError:
            raise
        except OSError as error:
            raise ProfilerError("SNAPSHOT_MEMBER_UNSTABLE") from error

    def create_member(self, name: str) -> int:
        if name in {".", ".."} or "/" in name or "\\" in name:
            raise ProfilerError("CAPSULE_CREATE_FAILED")
        try:
            if os.name == "nt":
                return self._windows_open_relative(name, create=True)
            return os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), stat.S_IRUSR | stat.S_IWUSR, dir_fd=self.handle)
        except OSError as error:
            raise ProfilerError("CAPSULE_CREATE_FAILED") from error

    def refresh_binding_metadata(self) -> None:
        try:
            self.binding_metadata = self._windows_metadata(self.handle) if os.name == "nt" else _stable_metadata(os.fstat(self.handle))
        except OSError as error:
            raise ProfilerError("SNAPSHOT_UNSTABLE") from error

    def retain_members(self, names: list[str]) -> list[_RetainedMember]:
        retained: list[_RetainedMember] = []
        try:
            for name in names:
                descriptor = self.open_member(name)
                retained.append(_RetainedMember(name, descriptor, _stable_metadata(os.fstat(descriptor))))
        except Exception:
            for member in retained:
                os.close(member.descriptor)
            raise
        self.member_handles.extend(retained)
        return retained

    def loader_path(self) -> str:
        if os.name == "nt":
            return str(self.path)
        if sys.platform.startswith("linux"):
            path = f"/proc/self/fd/{self.handle}"
            if not os.path.isdir(path):
                raise ProfilerError("CAPSULE_AUTHORITY_UNAVAILABLE")
            return path
        raise ProfilerError("CAPSULE_AUTHORITY_UNAVAILABLE")

    def verify_binding(self) -> None:
        try:
            if os.name == "nt":
                if self._windows_identity(self.handle) != self.identity or self._windows_metadata(self.handle) != self.binding_metadata:
                    raise ProfilerError("SNAPSHOT_UNSTABLE")
                current = _DirectoryGuard(self.path)
                try:
                    if current.identity != self.identity or current.binding_metadata != self.binding_metadata:
                        raise ProfilerError("SNAPSHOT_UNSTABLE")
                finally:
                    current.close()
            else:
                retained = os.fstat(self.handle)
                current = os.lstat(self.path)
                if _identity(retained) != self.identity or _identity(current) != self.identity or _stable_metadata(retained) != self.binding_metadata or _stable_metadata(current) != self.binding_metadata:
                    raise ProfilerError("SNAPSHOT_UNSTABLE")
            _assert_no_redirects(self.path, code="SNAPSHOT_UNSTABLE")
        except ProfilerError:
            raise
        except OSError as error:
            raise ProfilerError("SNAPSHOT_UNSTABLE") from error

    def close(self) -> None:
        for member in self.member_handles:
            try:
                os.close(member.descriptor)
            except OSError:
                pass
        self.member_handles.clear()
        self.capture_members.clear()
        if self.handle is None:
            return
        try:
            if os.name == "nt":
                ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(ctypes.c_void_p(self.handle))
            else:
                os.close(self.handle)
        finally:
            self.handle = None


def _assert_no_redirects(path: Path, *, include_leaf: bool = True, code: str = "UNSAFE_PATH") -> None:
    absolute = path.absolute()
    chain = list(reversed(absolute.parents))
    if include_leaf:
        chain.append(absolute)
    for member in chain:
        try:
            observed = os.lstat(member)
        except OSError as error:
            raise ProfilerError(code) from error
        if stat.S_ISLNK(observed.st_mode) or _is_reparse(observed):
            raise ProfilerError(code)


def _open_stable_regular(path: Path, max_bytes: int, code: str) -> tuple[BinaryIO, os.stat_result]:
    _assert_no_redirects(path, code=code)
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or _is_reparse(before) or before.st_size > max_bytes:
            raise ProfilerError(code)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        handle = os.fdopen(descriptor, "rb", closefd=True)
        opened = os.fstat(handle.fileno())
        if _identity(opened) != _identity(before) or not stat.S_ISREG(opened.st_mode):
            handle.close()
            raise ProfilerError(code)
        return handle, opened
    except ProfilerError:
        raise
    except OSError as error:
        raise ProfilerError(code) from error


def _finish_stable_read(path: Path, handle: BinaryIO, opened: os.stat_result, code: str) -> None:
    try:
        after_handle = os.fstat(handle.fileno())
        after_path = os.lstat(path)
    except OSError as error:
        raise ProfilerError(code) from error
    if _stable_identity(after_handle) != _stable_identity(opened) or _identity(after_path) != _identity(opened) or stat.S_ISLNK(after_path.st_mode) or _is_reparse(after_path):
        raise ProfilerError(code)
    _assert_no_redirects(path, code=code)


def _stable_read_file(path: Path, max_bytes: int, code: str) -> tuple[bytes, os.stat_result]:
    handle, opened = _open_stable_regular(path, max_bytes, code)
    try:
        data = handle.read(max_bytes + 1)
        if len(data) > max_bytes or len(data) != opened.st_size:
            raise ProfilerError(code)
        _finish_stable_read(path, handle, opened, code)
        return data, opened
    finally:
        handle.close()


def _stable_read_member(guard: _DirectoryGuard, name: str, max_bytes: int, code: str) -> tuple[bytes, os.stat_result]:
    retained = guard.capture_members.get(name)
    descriptor = retained.descriptor if retained is not None else guard.open_member(name)
    try:
        opened = os.fstat(descriptor)
        if opened.st_size > max_bytes:
            raise ProfilerError(code)
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1_048_576, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(data) > max_bytes or len(data) != opened.st_size or _stable_metadata(after) != _stable_metadata(opened):
            raise ProfilerError(code)
        if retained is not None:
            if _stable_metadata(opened) != retained.metadata:
                raise ProfilerError(code)
            digest = _sha256_bytes(data)
            if retained.byte_count < 0:
                retained.byte_count, retained.sha256 = len(data), digest
            elif (retained.byte_count, retained.sha256) != (len(data), digest):
                raise ProfilerError(code)
        guard.verify_binding()
        return data, opened
    finally:
        if retained is None:
            os.close(descriptor)


def _strict_json_loads(raw: bytes, code: str) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("constant")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ProfilerError(code) from error


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as error:
        raise ProfilerError("YAML_PARSER_UNAVAILABLE") from error
    raw, _ = _stable_read_file(path, CONFIG_MAX_BYTES, "CONFIG_READ_FAILED")
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_unique_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise yaml.constructor.ConstructorError("mapping", node.start_mark, "duplicate key", key_node.start_mark)
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_unique_mapping)
    try:
        value = yaml.load(raw.decode("utf-8"), Loader=UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ProfilerError("CONFIG_READ_FAILED") from error
    return _require_mapping(value, "INVALID_CONFIG")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    _require_exact_keys(config, {"model", "runtime", "input", "budgets", "limits"}, "INVALID_CONFIG")
    source_model = _require_mapping(config["model"], "INVALID_MODEL_CONFIG")
    _require_exact_keys(source_model, {"ref", "model_revision", "tokenizer_revision", "local_tokenizer_path"}, "INVALID_MODEL_CONFIG")
    model_ref = source_model["ref"]
    if not isinstance(model_ref, str) or MODEL_REF_RE.fullmatch(model_ref) is None or any(segment in {".", ".."} for segment in model_ref.split("/")):
        raise ProfilerError("INVALID_MODEL_REF")
    local_path = source_model["local_tokenizer_path"]
    if not isinstance(local_path, str) or not local_path:
        raise ProfilerError("INVALID_MODEL_CONFIG")
    model = {"ref": model_ref, "model_revision": _immutable_revision(source_model["model_revision"]), "tokenizer_revision": _immutable_revision(source_model["tokenizer_revision"]), "local_tokenizer_path": local_path}

    source_runtime = _require_mapping(config["runtime"], "INVALID_RUNTIME_CONFIG")
    _require_exact_keys(source_runtime, {"provider", "lock_path", "expected_lock_sha256"}, "INVALID_RUNTIME_CONFIG")
    if source_runtime["provider"] != "modal" or not isinstance(source_runtime["lock_path"], str) or not source_runtime["lock_path"]:
        raise ProfilerError("INVALID_RUNTIME_CONFIG")
    runtime = {"provider": "modal", "lock_path": source_runtime["lock_path"], "expected_lock_sha256": _digest(source_runtime["expected_lock_sha256"], "INVALID_RUNTIME_CONFIG")}

    source_input = _require_mapping(config["input"], "INVALID_INPUT_CONFIG")
    allowed_input = {"jsonl_path", "mode", "text_field", "messages_field", "components", "use_chat_template", "add_generation_prompt"}
    if set(source_input) - allowed_input:
        raise ProfilerError("INVALID_INPUT_CONFIG")
    jsonl_path, mode = source_input.get("jsonl_path"), source_input.get("mode")
    if not isinstance(jsonl_path, str) or not jsonl_path or mode not in {"text", "messages", "components"}:
        raise ProfilerError("INVALID_INPUT_CONFIG")
    input_config: dict[str, Any] = {"jsonl_path": jsonl_path, "mode": mode}
    if mode == "text":
        if set(source_input) - {"jsonl_path", "mode", "text_field"}:
            raise ProfilerError("INVALID_INPUT_CONFIG")
        field = source_input.get("text_field", "text")
        if not isinstance(field, str) or not field:
            raise ProfilerError("INVALID_INPUT_CONFIG")
        input_config["text_field"] = field
    elif mode == "messages":
        if set(source_input) - {"jsonl_path", "mode", "messages_field", "use_chat_template", "add_generation_prompt"}:
            raise ProfilerError("INVALID_INPUT_CONFIG")
        field = source_input.get("messages_field", "messages")
        use_template, add_prompt = source_input.get("use_chat_template", True), source_input.get("add_generation_prompt", False)
        if not isinstance(field, str) or not field or use_template is not True or not isinstance(add_prompt, bool):
            raise ProfilerError("INVALID_INPUT_CONFIG")
        input_config.update(messages_field=field, use_chat_template=True, add_generation_prompt=add_prompt)
    else:
        if set(source_input) != {"jsonl_path", "mode", "components"}:
            raise ProfilerError("INVALID_INPUT_CONFIG")
        components = source_input.get("components")
        if not isinstance(components, dict) or not components:
            raise ProfilerError("INVALID_COMPONENT_CONFIG")
        input_config["components"] = dict(sorted(components.items()))

    source_budgets = _require_mapping(config["budgets"], "INVALID_BUDGET_CONFIG")
    _require_exact_keys(source_budgets, {"max_sequence_tokens", "completion_reserve_tokens"}, "INVALID_BUDGET_CONFIG")
    budgets: dict[str, int | None] = {}
    for key in ("max_sequence_tokens", "completion_reserve_tokens"):
        value = source_budgets[key]
        if key == "max_sequence_tokens" and value is None:
            budgets[key] = None
        elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProfilerError("INVALID_BUDGET_CONFIG")
        else:
            budgets[key] = value
    if budgets["max_sequence_tokens"] is not None and budgets["completion_reserve_tokens"] > budgets["max_sequence_tokens"]:
        raise ProfilerError("INVALID_BUDGET_CONFIG")

    source_limits = _require_mapping(config["limits"], "INVALID_LIMIT_CONFIG")
    _require_exact_keys(source_limits, set(HARD_LIMITS), "INVALID_LIMIT_CONFIG")
    limits: dict[str, int] = {}
    for key, ceiling in HARD_LIMITS.items():
        value = source_limits[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > ceiling:
            raise ProfilerError("INVALID_LIMIT_CONFIG")
        limits[key] = value
    if limits["max_record_bytes"] > limits["max_line_bytes"] or limits["max_line_bytes"] > limits["max_input_bytes"]:
        raise ProfilerError("INVALID_LIMIT_CONFIG")
    components = input_config.get("components")
    if components is not None:
        if len(components) > limits["max_components"]:
            raise ProfilerError("INVALID_COMPONENT_CONFIG")
        for label, field in components.items():
            if not isinstance(label, str) or len(label) > limits["max_component_label_chars"] or COMPONENT_LABEL_RE.fullmatch(label) is None or not isinstance(field, str) or not field:
                raise ProfilerError("INVALID_COMPONENT_CONFIG")
    return {"model": model, "runtime": runtime, "input": input_config, "budgets": budgets, "limits": limits}


def _runtime_schema_path() -> Path:
    try:
        return Path(__file__).resolve().parents[3] / "schemas" / "synaptic-modal-runtime-lock-v1.schema.json"
    except (OSError, IndexError) as error:
        raise ProfilerError("RUNTIME_SCHEMA_INVALID") from error


def _load_runtime_commitment(runtime_config: dict[str, Any]) -> dict[str, Any]:
    lock_bytes, _ = _stable_read_file(Path(runtime_config["lock_path"]), RUNTIME_LOCK_MAX_BYTES, "RUNTIME_LOCK_INVALID")
    lock_sha = _sha256_bytes(lock_bytes)
    if lock_sha != runtime_config["expected_lock_sha256"]:
        raise ProfilerError("RUNTIME_LOCK_DIGEST_MISMATCH")
    schema_bytes, _ = _stable_read_file(_runtime_schema_path(), RUNTIME_SCHEMA_MAX_BYTES, "RUNTIME_SCHEMA_INVALID")
    lock, schema = _strict_json_loads(lock_bytes, "RUNTIME_LOCK_INVALID"), _strict_json_loads(schema_bytes, "RUNTIME_SCHEMA_INVALID")
    try:
        import jsonschema  # type: ignore

        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(lock)
    except Exception as error:
        raise ProfilerError("RUNTIME_LOCK_SCHEMA_MISMATCH") from error
    lock_map = _require_mapping(lock, "RUNTIME_LOCK_INVALID")
    ml_stack, python = _require_mapping(lock_map.get("ml_stack"), "RUNTIME_LOCK_INVALID"), _require_mapping(lock_map.get("python"), "RUNTIME_LOCK_INVALID")
    registry = lock_map.get("registry_reference")
    if not isinstance(registry, str) or re.fullmatch(r"unsloth/unsloth:[^@]+@sha256:[0-9a-f]{64}", registry) is None:
        raise ProfilerError("RUNTIME_LOCK_INVALID")
    return {"provider": "modal", "runtime_lock_sha256": lock_sha, "runtime_schema_sha256": _sha256_bytes(schema_bytes), "registry_reference": registry, "python_version": python.get("version"), "transformers_version": ml_stack.get("transformers"), "tokenizer_stack_claim": "transformers_exact_image_committed_tokenizers_version_unavailable"}


def _semantic_config(config: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    return {"model": {key: config["model"][key] for key in ("ref", "model_revision", "tokenizer_revision")}, "runtime": runtime, "input": {key: value for key, value in config["input"].items() if key != "jsonl_path"}, "budgets": config["budgets"], "limits": config["limits"], "tokenizer_load_closure_algorithm": CLOSURE_ALGORITHM}


def _verify_retained_members(guard: _DirectoryGuard, retained: list[_RetainedMember], max_bytes: int, code: str) -> None:
    try:
        expected_names = [member.name for member in retained]
        if sorted(guard.list_names()) != expected_names:
            raise ProfilerError(code)
        for member in retained:
            if _stable_metadata(os.fstat(member.descriptor)) != member.metadata:
                raise ProfilerError(code)
            _stable_read_member(guard, member.name, max_bytes, code)
            current_descriptor = guard.open_member(member.name)
            try:
                current_metadata = os.fstat(current_descriptor)
                if _identity(current_metadata) != _identity(os.fstat(member.descriptor)) or _stable_metadata(current_metadata) != member.metadata:
                    raise ProfilerError(code)
                os.lseek(current_descriptor, 0, os.SEEK_SET)
                current_hasher = hashlib.sha256()
                current_bytes = 0
                while True:
                    chunk = os.read(current_descriptor, 1_048_576)
                    if not chunk:
                        break
                    current_bytes += len(chunk)
                    if current_bytes > max_bytes:
                        raise ProfilerError(code)
                    current_hasher.update(chunk)
                if (current_bytes, current_hasher.hexdigest()) != (member.byte_count, member.sha256) or _stable_metadata(os.fstat(current_descriptor)) != _stable_metadata(current_metadata):
                    raise ProfilerError(code)
            finally:
                os.close(current_descriptor)
        guard.verify_binding()
    except ProfilerError as error:
        if error.code == code:
            raise
        raise ProfilerError(code) from error
    except OSError as error:
        raise ProfilerError(code) from error


def _guard_inventory(guard: _DirectoryGuard, limits: dict[str, int]) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    names = guard.list_names()
    if len(names) > limits["max_snapshot_files"]:
        raise ProfilerError("SNAPSHOT_LIMIT_EXCEEDED")
    if len({name.casefold() for name in names}) != len(names):
        raise ProfilerError("SNAPSHOT_CASE_COLLISION")
    if set(names) - TOKENIZER_ALLOWED_FILES or not TOKENIZER_REQUIRED_FILES.issubset(names):
        raise ProfilerError("SNAPSHOT_INVENTORY_INVALID")
    inventory: list[dict[str, Any]] = []
    contents: dict[str, bytes] = {}
    total_bytes = 0
    retained = guard.retain_members(sorted(names))
    guard.capture_members = {member.name: member for member in retained}
    try:
        for name in sorted(names):
            data, _ = _stable_read_member(guard, name, limits["max_snapshot_file_bytes"], "SNAPSHOT_MEMBER_UNSTABLE")
            total_bytes += len(data)
            if total_bytes > limits["max_snapshot_total_bytes"]:
                raise ProfilerError("SNAPSHOT_LIMIT_EXCEEDED")
            inventory.append({"path": name, "bytes": len(data), "sha256": _sha256_bytes(data)})
            contents[name] = data
        _verify_retained_members(guard, retained, limits["max_snapshot_file_bytes"], "SNAPSHOT_MEMBER_UNSTABLE")
        return inventory, contents
    finally:
        guard.capture_members.clear()


def _snapshot_inventory(snapshot: Path, limits: dict[str, int]) -> tuple[list[dict[str, Any]], dict[str, bytes], os.stat_result]:
    _assert_no_redirects(snapshot, code="SNAPSHOT_INVALID")
    guard = _DirectoryGuard(snapshot)
    try:
        root_before = os.lstat(snapshot)
        if not stat.S_ISDIR(root_before.st_mode) or _is_reparse(root_before):
            raise ProfilerError("SNAPSHOT_INVALID")
        inventory, contents = _guard_inventory(guard, limits)
        root_after = os.lstat(snapshot)
        guard.verify_binding()
        if _stable_identity(root_after) != _stable_identity(root_before):
            raise ProfilerError("SNAPSHOT_UNSTABLE")
        return inventory, contents, root_before
    except ProfilerError:
        raise
    except OSError as error:
        raise ProfilerError("SNAPSHOT_INVALID") from error
    finally:
        guard.close()


def _closure_identity(inventory: list[dict[str, Any]]) -> dict[str, Any]:
    return {"algorithm": CLOSURE_ALGORITHM, "sha256": _digest_record({"algorithm": CLOSURE_ALGORITHM, "files": inventory}, "tokenizer-closure/1"), "member_count": len(inventory), "total_bytes": sum(item["bytes"] for item in inventory), "files": inventory}


def _sync_stage_directory(path: Path) -> None:
    if os.name != "nt":
        _fsync_directory(path)


def _parent_durability_supported() -> bool:
    return os.name != "nt"


def _write_capsule(contents: dict[str, bytes]) -> tuple[Path, _DirectoryGuard]:
    capsule: Path | None = None
    guard: _DirectoryGuard | None = None
    try:
        capsule = Path(tempfile.mkdtemp(prefix="tokenizer-profile-capsule-"))
        os.chmod(capsule, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        guard = _DirectoryGuard(capsule)
        for name, data in sorted(contents.items()):
            descriptor = guard.create_member(name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if os.name != "nt":
                os.chmod(name, stat.S_IRUSR, dir_fd=guard.handle, follow_symlinks=False)
        if os.name != "nt":
            os.fsync(guard.handle)
            os.chmod(capsule, stat.S_IRUSR | stat.S_IXUSR)
        guard.refresh_binding_metadata()
        retained = guard.retain_members(sorted(contents))
        guard.capture_members = {member.name: member for member in retained}
        for member in retained:
            member.byte_count = len(contents[member.name])
            member.sha256 = _sha256_bytes(contents[member.name])
        _verify_retained_members(guard, retained, max(len(data) for data in contents.values()), "CAPSULE_MUTATED")
        guard.verify_binding()
        return capsule, guard
    except ProfilerError:
        if guard is not None:
            guard.close()
        if capsule is not None:
            shutil.rmtree(capsule, ignore_errors=True)
        raise
    except OSError as error:
        if guard is not None:
            guard.close()
        if capsule is not None:
            shutil.rmtree(capsule, ignore_errors=True)
        raise ProfilerError("CAPSULE_CREATE_FAILED") from error


def _verify_capsule(guard: _DirectoryGuard, expected_inventory: list[dict[str, Any]], limits: dict[str, int]) -> None:
    retained_inventory = [{"path": member.name, "bytes": member.byte_count, "sha256": member.sha256} for member in guard.member_handles]
    if retained_inventory != expected_inventory:
        raise ProfilerError("CAPSULE_MUTATED")
    _verify_retained_members(guard, guard.member_handles, limits["max_snapshot_file_bytes"], "CAPSULE_MUTATED")


def _load_tokenizer(config: dict[str, Any], runtime: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    inventory, contents, root_stat = _snapshot_inventory(Path(config["model"]["local_tokenizer_path"]), config["limits"])
    capsule, capsule_guard = _write_capsule(contents)
    loader_path = capsule_guard.loader_path()
    old_environment = {key: os.environ.get(key) for key in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN")}
    cleanup_authorized = False
    try:
        os.environ.update(TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
        try:
            import transformers  # type: ignore
            from transformers import AutoTokenizer  # type: ignore
        except Exception as error:
            raise ProfilerError("TRANSFORMERS_UNAVAILABLE") from error
        installed = getattr(transformers, "__version__", None)
        if installed != runtime["transformers_version"]:
            raise ProfilerError("TRANSFORMERS_VERSION_MISMATCH")
        capsule_guard.verify_binding()
        try:
            tokenizer = AutoTokenizer.from_pretrained(loader_path, local_files_only=True, trust_remote_code=False)
        except Exception as error:
            raise ProfilerError("LOCAL_TOKENIZER_LOAD_FAILED") from error
        loaded_path = getattr(tokenizer, "name_or_path", None)
        if loaded_path != loader_path:
            raise ProfilerError("TOKENIZER_LOAD_SOURCE_AMBIGUOUS")
        _verify_capsule(capsule_guard, inventory, config["limits"])
        cleanup_authorized = True
        return tokenizer, {"load_closure": _closure_identity(inventory), "source_root_identity_sha256": _identity_digest(root_stat), "transformers_version": installed}
    finally:
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if not cleanup_authorized:
            try:
                capsule_guard.verify_binding()
                cleanup_authorized = True
            except ProfilerError:
                pass
        capsule_guard.close()
        if cleanup_authorized:
            try:
                if os.name != "nt":
                    os.chmod(capsule, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
                shutil.rmtree(capsule)
            except OSError:
                pass


class Histogram:
    def __init__(self) -> None:
        self.counts: dict[int, int] = defaultdict(int)
        self.n = 0

    def add(self, value: int) -> None:
        self.counts[value] += 1
        self.n += 1

    def distribution(self) -> dict[str, int]:
        if self.n == 0:
            return {"n": 0, "min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
        ordered = sorted(self.counts)

        def nearest_rank(percentile: int) -> int:
            target, seen = math.ceil((percentile / 100) * self.n), 0
            for value in ordered:
                seen += self.counts[value]
                if seen >= target:
                    return value
            raise AssertionError("unreachable")

        return {"n": self.n, "min": ordered[0], "p50": nearest_rank(50), "p90": nearest_rank(90), "p95": nearest_rank(95), "p99": nearest_rank(99), "max": ordered[-1]}


def _token_count(tokenizer: Any, text: str, limit: int) -> int:
    try:
        count = len(tokenizer.encode(text, add_special_tokens=False))
    except Exception as error:
        raise ProfilerError("TOKENIZATION_FAILED") from error
    if isinstance(count, bool) or not isinstance(count, int) or count > limit:
        raise ProfilerError("TOKEN_LIMIT_EXCEEDED")
    return count


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        try:
            return _canonical_json(content)
        except (TypeError, ValueError, RecursionError) as error:
            raise ProfilerError("INVALID_MESSAGES_ROW") from error
    raise ProfilerError("INVALID_MESSAGES_ROW")


def _row_counts(tokenizer: Any, row: dict[str, Any], input_config: dict[str, Any], limits: dict[str, int]) -> tuple[dict[str, int], int]:
    token_limit = limits["max_tokens_per_record"]
    mode = input_config["mode"]
    if mode == "text":
        value = row.get(input_config["text_field"])
        if not isinstance(value, str):
            raise ProfilerError("INVALID_TEXT_ROW")
        count = _token_count(tokenizer, value, token_limit)
        return {"text": count}, count
    if mode == "components":
        counts: dict[str, int] = {}
        for component, field in input_config["components"].items():
            value = row.get(field)
            if not isinstance(value, str):
                raise ProfilerError("INVALID_COMPONENT_ROW")
            counts[component] = _token_count(tokenizer, value, token_limit)
        total = sum(counts.values())
        if total > token_limit:
            raise ProfilerError("TOKEN_LIMIT_EXCEEDED")
        return counts, total
    messages = row.get(input_config["messages_field"])
    if not isinstance(messages, list) or not messages or len(messages) > limits["max_components"]:
        raise ProfilerError("INVALID_MESSAGES_ROW")
    components = {role: 0 for role in MESSAGE_ROLE_BUCKETS}
    normalized: list[dict[str, Any]] = []
    for message in messages:
        mapping = _require_mapping(message, "INVALID_MESSAGES_ROW")
        role = mapping.get("role")
        if role not in MESSAGE_ROLE_BUCKETS:
            raise ProfilerError("INVALID_MESSAGES_ROW")
        content = _message_content(mapping)
        components[role] += _token_count(tokenizer, content, token_limit)
        if components[role] > token_limit:
            raise ProfilerError("TOKEN_LIMIT_EXCEEDED")
        normalized.append(dict(mapping))
    try:
        total = len(tokenizer.apply_chat_template(normalized, tokenize=True, add_generation_prompt=input_config["add_generation_prompt"]))
    except Exception as error:
        raise ProfilerError("CHAT_TEMPLATE_RENDER_FAILED") from error
    if total > token_limit:
        raise ProfilerError("TOKEN_LIMIT_EXCEEDED")
    return components, total


def _stream_rows(path: Path, limits: dict[str, int], hasher: Any, byte_counter: list[int]) -> Iterator[dict[str, Any]]:
    handle, opened = _open_stable_regular(path, limits["max_input_bytes"], "INPUT_READ_FAILED")
    total_bytes = rows = 0
    try:
        while True:
            line = handle.readline(limits["max_line_bytes"] + 1)
            if not line:
                break
            total_bytes += len(line)
            hasher.update(line)
            byte_counter[0] += len(line)
            if len(line) > limits["max_line_bytes"] or total_bytes > limits["max_input_bytes"]:
                raise ProfilerError("INPUT_LIMIT_EXCEEDED")
            record = line[:-1] if line.endswith(b"\n") else line
            if record.endswith(b"\r"):
                record = record[:-1]
            if not record.strip():
                continue
            if len(record) > limits["max_record_bytes"]:
                raise ProfilerError("INPUT_LIMIT_EXCEEDED")
            rows += 1
            if rows > limits["max_rows"]:
                raise ProfilerError("INPUT_LIMIT_EXCEEDED")
            yield _require_mapping(_strict_json_loads(record, "INVALID_JSONL_ROW"), "INVALID_JSONL_ROW")
        if total_bytes != opened.st_size:
            raise ProfilerError("INPUT_UNSTABLE")
        _finish_stable_read(path, handle, opened, "INPUT_UNSTABLE")
    finally:
        handle.close()


def _profile_semantic_payload(profile: dict[str, Any]) -> dict[str, Any]:
    return {key: profile[key] for key in ("schema_version", "model", "runtime", "tokenizer", "config_provenance", "input_provenance", "components", "total", "budget")}


def _compute_profile_id(profile: dict[str, Any]) -> str:
    return _digest_record(_profile_semantic_payload(profile), "token-profile-semantic/1")


def profile(config: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_config(config)
    runtime = _load_runtime_commitment(normalized["runtime"])
    semantic_config_sha = _digest_record(_semantic_config(normalized, runtime), "token-profile-config/1")
    tokenizer, tokenizer_evidence = _load_tokenizer(normalized, runtime)
    chat_template_sha: str | None = None
    if normalized["input"]["mode"] == "messages":
        chat_template = getattr(tokenizer, "chat_template", None)
        if not isinstance(chat_template, str) or not chat_template:
            raise ProfilerError("CHAT_TEMPLATE_UNAVAILABLE")
        chat_template_sha = _sha256_bytes(chat_template.encode("utf-8"))
    component_names = list(MESSAGE_ROLE_BUCKETS) if normalized["input"]["mode"] == "messages" else (["text"] if normalized["input"]["mode"] == "text" else sorted(normalized["input"]["components"]))
    component_histograms = {name: Histogram() for name in component_names}
    total_histogram, over_histogram, slack_histogram = Histogram(), Histogram(), Histogram()
    input_hasher, input_bytes, row_count = hashlib.sha256(), [0], 0
    budget, reserve, over_count = normalized["budgets"]["max_sequence_tokens"], normalized["budgets"]["completion_reserve_tokens"], 0
    for row in _stream_rows(Path(normalized["input"]["jsonl_path"]), normalized["limits"], input_hasher, input_bytes):
        components, total = _row_counts(tokenizer, row, normalized["input"], normalized["limits"])
        for name in component_names:
            component_histograms[name].add(components[name])
        total_histogram.add(total)
        if budget is not None:
            over, slack = max(0, total + reserve - budget), max(0, budget - reserve - total)
            over_histogram.add(over)
            slack_histogram.add(slack)
            over_count += int(over > 0)
        row_count += 1
    if row_count == 0:
        raise ProfilerError("EMPTY_INPUT")
    result: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA,
        "model": {key: normalized["model"][key] for key in ("ref", "model_revision", "tokenizer_revision")},
        "runtime": runtime,
        "tokenizer": {"load_closure": tokenizer_evidence["load_closure"], "chat_template_sha256": chat_template_sha},
        "config_provenance": {"semantic_config_sha256": semantic_config_sha, "component_label_max_chars": normalized["limits"]["max_component_label_chars"]},
        "input_provenance": {"jsonl_sha256": input_hasher.hexdigest(), "bytes": input_bytes[0], "row_count": row_count},
        "components": {name: component_histograms[name].distribution() for name in sorted(component_histograms)},
        "total": total_histogram.distribution(),
        "budget": {"max_sequence_tokens": budget, "completion_reserve_tokens": reserve, "prompt_budget_tokens": None if budget is None else budget - reserve, "evaluated_record_count": row_count, "threshold_semantics": "total + completion_reserve > max_sequence_tokens", "over_limit_count": None if budget is None else over_count, "over_limit_fraction_ppm": None if budget is None else (over_count * 1_000_000) // row_count, "over_limit": None if budget is None else over_histogram.distribution(), "slack": None if budget is None else slack_histogram.distribution()},
        "operational_provenance": {"snapshot_root_identity_sha256": tokenizer_evidence["source_root_identity_sha256"]},
    }
    result["profile_semantic_id"] = _compute_profile_id(result)
    return result


def _output_directory(output_prefix: Path) -> Path:
    return Path(str(output_prefix) + ".token-profile")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        raise OSError(errno.ENOTSUP, "directory durability unsupported")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as error:
            raise ProfilerError("OUTPUT_COLLISION") from error
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise ProfilerError("ATOMIC_NOREPLACE_UNAVAILABLE")
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise ProfilerError("OUTPUT_COLLISION")
            raise OSError(error_number, os.strerror(error_number))
        return
    raise ProfilerError("ATOMIC_NOREPLACE_UNAVAILABLE")


def _csv_bytes(result: dict[str, Any]) -> bytes:
    buffer = io.StringIO(newline="")
    rows = [("component:" + name, stats) for name, stats in result["components"].items()]
    if result["budget"]["over_limit"] is not None:
        rows.extend([("over_limit", result["budget"]["over_limit"]), ("slack", result["budget"]["slack"])])
    rows.append(("total", result["total"]))
    writer = csv.DictWriter(buffer, fieldnames=["series", "n", "min", "p50", "p90", "p95", "p99", "max"], lineterminator="\n")
    writer.writeheader()
    for name, stats in rows:
        writer.writerow({"series": name, **stats})
    return buffer.getvalue().encode("utf-8")


def _publication_digest(payloads: dict[str, bytes]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"token-profile-publication/1\0")
    for name in PROFILE_FILENAMES:
        data = payloads[name]
        hasher.update(name.encode("ascii") + b"\0" + len(data).to_bytes(8, "big") + data)
    return hasher.hexdigest()


def _artifact_payloads(profile_result: dict[str, Any], root_stat: os.stat_result) -> dict[str, bytes]:
    profile_bytes, csv_bytes = (_canonical_json(profile_result) + "\n").encode("utf-8"), _csv_bytes(profile_result)
    manifest = {"schema_version": MANIFEST_SCHEMA, "profile_semantic_id": profile_result["profile_semantic_id"], "publication_root_identity_sha256": _identity_digest(root_stat), "files": {"profile.csv": {"bytes": len(csv_bytes), "sha256": _sha256_bytes(csv_bytes)}, "profile.json": {"bytes": len(profile_bytes), "sha256": _sha256_bytes(profile_bytes)}}}
    return {"manifest.json": (_canonical_json(manifest) + "\n").encode("utf-8"), "profile.json": profile_bytes, "profile.csv": csv_bytes}


def _validate_distribution(value: Any) -> None:
    mapping = _require_mapping(value, "PROFILE_ARTIFACT_INVALID")
    _require_exact_keys(mapping, {"n", "min", "p50", "p90", "p95", "p99", "max"}, "PROFILE_ARTIFACT_INVALID")
    numbers = [mapping[key] for key in ("n", "min", "p50", "p90", "p95", "p99", "max")]
    if any(isinstance(number, bool) or not isinstance(number, int) or number < 0 for number in numbers):
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    if not (mapping["min"] <= mapping["p50"] <= mapping["p90"] <= mapping["p95"] <= mapping["p99"] <= mapping["max"]):
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")


def _validate_profile_shape(profile: dict[str, Any]) -> None:
    _require_exact_keys(
        profile,
        {"schema_version", "profile_semantic_id", "model", "runtime", "tokenizer", "config_provenance", "input_provenance", "components", "total", "budget", "operational_provenance"},
        "PROFILE_ARTIFACT_INVALID",
    )
    if profile["schema_version"] != PROFILE_SCHEMA:
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    _digest(profile["profile_semantic_id"], "PROFILE_ARTIFACT_INVALID")
    model = _require_mapping(profile["model"], "PROFILE_ARTIFACT_INVALID")
    _require_exact_keys(model, {"ref", "model_revision", "tokenizer_revision"}, "PROFILE_ARTIFACT_INVALID")
    if not isinstance(model["ref"], str) or MODEL_REF_RE.fullmatch(model["ref"]) is None:
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    if any(not isinstance(model[key], str) or re.fullmatch(r"[0-9a-f]{40}", model[key]) is None for key in ("model_revision", "tokenizer_revision")):
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    runtime = _require_mapping(profile["runtime"], "PROFILE_ARTIFACT_INVALID")
    _require_exact_keys(runtime, {"provider", "runtime_lock_sha256", "runtime_schema_sha256", "registry_reference", "python_version", "transformers_version", "tokenizer_stack_claim"}, "PROFILE_ARTIFACT_INVALID")
    if runtime["provider"] != "modal" or not all(isinstance(runtime[key], str) and runtime[key] for key in ("registry_reference", "python_version", "transformers_version", "tokenizer_stack_claim")):
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    _digest(runtime["runtime_lock_sha256"], "PROFILE_ARTIFACT_INVALID")
    _digest(runtime["runtime_schema_sha256"], "PROFILE_ARTIFACT_INVALID")
    tokenizer = _require_mapping(profile["tokenizer"], "PROFILE_ARTIFACT_INVALID")
    _require_exact_keys(tokenizer, {"load_closure", "chat_template_sha256"}, "PROFILE_ARTIFACT_INVALID")
    closure = _require_mapping(tokenizer["load_closure"], "PROFILE_ARTIFACT_INVALID")
    _require_exact_keys(closure, {"algorithm", "sha256", "member_count", "total_bytes", "files"}, "PROFILE_ARTIFACT_INVALID")
    if closure["algorithm"] != CLOSURE_ALGORITHM or any(isinstance(closure[key], bool) or not isinstance(closure[key], int) or closure[key] < 0 for key in ("member_count", "total_bytes")):
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    _digest(closure["sha256"], "PROFILE_ARTIFACT_INVALID")
    files = closure["files"]
    if not isinstance(files, list) or len(files) != closure["member_count"]:
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    total_bytes = 0
    prior_path = ""
    for item in files:
        member = _require_mapping(item, "PROFILE_ARTIFACT_INVALID")
        _require_exact_keys(member, {"path", "bytes", "sha256"}, "PROFILE_ARTIFACT_INVALID")
        path = member["path"]
        if path not in TOKENIZER_ALLOWED_FILES or path <= prior_path or isinstance(member["bytes"], bool) or not isinstance(member["bytes"], int) or member["bytes"] < 0:
            raise ProfilerError("PROFILE_ARTIFACT_INVALID")
        _digest(member["sha256"], "PROFILE_ARTIFACT_INVALID")
        prior_path = path
        total_bytes += member["bytes"]
    if total_bytes != closure["total_bytes"] or closure["sha256"] != _digest_record({"algorithm": CLOSURE_ALGORITHM, "files": files}, "tokenizer-closure/1"):
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    if tokenizer["chat_template_sha256"] is not None:
        _digest(tokenizer["chat_template_sha256"], "PROFILE_ARTIFACT_INVALID")
    config_provenance = _require_mapping(profile["config_provenance"], "PROFILE_ARTIFACT_INVALID")
    _require_exact_keys(config_provenance, {"semantic_config_sha256", "component_label_max_chars"}, "PROFILE_ARTIFACT_INVALID")
    _digest(config_provenance["semantic_config_sha256"], "PROFILE_ARTIFACT_INVALID")
    component_label_max_chars = config_provenance["component_label_max_chars"]
    if isinstance(component_label_max_chars, bool) or not isinstance(component_label_max_chars, int) or not 0 < component_label_max_chars <= HARD_LIMITS["max_component_label_chars"]:
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    input_provenance = _require_mapping(profile["input_provenance"], "PROFILE_ARTIFACT_INVALID")
    _require_exact_keys(input_provenance, {"jsonl_sha256", "bytes", "row_count"}, "PROFILE_ARTIFACT_INVALID")
    _digest(input_provenance["jsonl_sha256"], "PROFILE_ARTIFACT_INVALID")
    if any(isinstance(input_provenance[key], bool) or not isinstance(input_provenance[key], int) or input_provenance[key] <= 0 for key in ("bytes", "row_count")):
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    components = _require_mapping(profile["components"], "PROFILE_ARTIFACT_INVALID")
    if not components or any(not isinstance(name, str) or len(name) > component_label_max_chars or COMPONENT_LABEL_RE.fullmatch(name) is None for name in components):
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    for distribution in components.values():
        _validate_distribution(distribution)
    _validate_distribution(profile["total"])
    budget = _require_mapping(profile["budget"], "PROFILE_ARTIFACT_INVALID")
    _require_exact_keys(budget, {"max_sequence_tokens", "completion_reserve_tokens", "prompt_budget_tokens", "evaluated_record_count", "threshold_semantics", "over_limit_count", "over_limit_fraction_ppm", "over_limit", "slack"}, "PROFILE_ARTIFACT_INVALID")
    if budget["threshold_semantics"] != "total + completion_reserve > max_sequence_tokens" or budget["evaluated_record_count"] != input_provenance["row_count"]:
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    for key in ("max_sequence_tokens", "completion_reserve_tokens", "prompt_budget_tokens", "over_limit_count", "over_limit_fraction_ppm"):
        value = budget[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    if (budget["over_limit"] is None) != (budget["slack"] is None):
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    if budget["over_limit"] is not None:
        _validate_distribution(budget["over_limit"])
        _validate_distribution(budget["slack"])
    operational = _require_mapping(profile["operational_provenance"], "PROFILE_ARTIFACT_INVALID")
    _require_exact_keys(operational, {"snapshot_root_identity_sha256"}, "PROFILE_ARTIFACT_INVALID")
    _digest(operational["snapshot_root_identity_sha256"], "PROFILE_ARTIFACT_INVALID")


def verify_profile_directory(profile_dir: Path, expected_profile_id: str | None = None) -> dict[str, str]:
    if expected_profile_id is not None:
        _digest(expected_profile_id, "INVALID_EXPECTED_PROFILE_ID")
    _assert_no_redirects(profile_dir, code="PROFILE_ARTIFACT_INVALID")
    try:
        root_stat = os.lstat(profile_dir)
        if not stat.S_ISDIR(root_stat.st_mode) or _is_reparse(root_stat):
            raise ProfilerError("PROFILE_ARTIFACT_INVALID")
        names = [entry.name for entry in os.scandir(profile_dir)]
    except ProfilerError:
        raise
    except OSError as error:
        raise ProfilerError("PROFILE_ARTIFACT_INVALID") from error
    if len({name.casefold() for name in names}) != len(names) or set(names) != set(PROFILE_FILENAMES):
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    payloads = {name: _stable_read_file(profile_dir / name, ARTIFACT_MEMBER_MAX_BYTES, "PROFILE_ARTIFACT_INVALID")[0] for name in PROFILE_FILENAMES}
    manifest = _require_mapping(_strict_json_loads(payloads["manifest.json"], "PROFILE_ARTIFACT_INVALID"), "PROFILE_ARTIFACT_INVALID")
    profile_result = _require_mapping(_strict_json_loads(payloads["profile.json"], "PROFILE_ARTIFACT_INVALID"), "PROFILE_ARTIFACT_INVALID")
    if payloads["manifest.json"] != (_canonical_json(manifest) + "\n").encode("utf-8") or payloads["profile.json"] != (_canonical_json(profile_result) + "\n").encode("utf-8"):
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    _require_exact_keys(manifest, {"schema_version", "profile_semantic_id", "publication_root_identity_sha256", "files"}, "PROFILE_ARTIFACT_INVALID")
    _validate_profile_shape(profile_result)
    if manifest["schema_version"] != MANIFEST_SCHEMA or manifest["publication_root_identity_sha256"] != _identity_digest(root_stat):
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    profile_id = profile_result.get("profile_semantic_id")
    if not isinstance(profile_id, str) or profile_id != _compute_profile_id(profile_result) or manifest["profile_semantic_id"] != profile_id:
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    if expected_profile_id is not None and profile_id != expected_profile_id:
        raise ProfilerError("PROFILE_ID_MISMATCH")
    expected_files = {name: {"bytes": len(payloads[name]), "sha256": _sha256_bytes(payloads[name])} for name in ("profile.csv", "profile.json")}
    if manifest["files"] != expected_files or payloads["profile.csv"] != _csv_bytes(profile_result):
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    try:
        root_after = os.lstat(profile_dir)
    except OSError as error:
        raise ProfilerError("PROFILE_ARTIFACT_INVALID") from error
    if _stable_identity(root_after) != _stable_identity(root_stat):
        raise ProfilerError("PROFILE_ARTIFACT_INVALID")
    return {"profile_id": profile_id, "publication_artifact_digest": _publication_digest(payloads)}


def publish_profile(result: dict[str, Any], output_prefix: Path) -> dict[str, str]:
    destination, parent = _output_directory(output_prefix), _output_directory(output_prefix).parent
    _assert_no_redirects(parent, code="OUTPUT_PARENT_INVALID")
    try:
        if destination.exists() or destination.is_symlink():
            raise ProfilerError("OUTPUT_COLLISION")
        stage: Path | None = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=parent))
        os.chmod(stage, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    except ProfilerError:
        raise
    except OSError as error:
        raise ProfilerError("OUTPUT_WRITE_FAILED") from error
    committed = False
    try:
        payloads = _artifact_payloads(result, os.lstat(stage))
        for name in PROFILE_FILENAMES:
            descriptor = os.open(stage / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), stat.S_IRUSR | stat.S_IWUSR)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payloads[name])
                handle.flush()
                os.fsync(handle.fileno())
        _sync_stage_directory(stage)
        verify_profile_directory(stage, result["profile_semantic_id"])
        _rename_directory_no_replace(stage, destination)
        committed, stage = True, None
        if _parent_durability_supported():
            try:
                _fsync_directory(parent)
            except OSError as error:
                raise PublicationUncertain(result["profile_semantic_id"], "parent_durability") from error
        try:
            verified = verify_profile_directory(destination, result["profile_semantic_id"])
        except ProfilerError as error:
            raise PublicationUncertain(result["profile_semantic_id"], "final_verification") from error
        return {**verified, "state": "PUBLISHED_DURABLE" if _parent_durability_supported() else "VERIFIED_DURABILITY_UNPROVEN"}
    except (ProfilerError, PublicationUncertain):
        raise
    except OSError as error:
        if committed:
            raise PublicationUncertain(result["profile_semantic_id"], "final_verification") from error
        raise ProfilerError("OUTPUT_WRITE_FAILED") from error
    finally:
        if not committed and stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def reconcile_profile(profile_dir: Path, expected_profile_id: str) -> dict[str, str]:
    _digest(expected_profile_id, "INVALID_EXPECTED_PROFILE_ID")
    durable = False
    if _parent_durability_supported():
        try:
            _fsync_directory(profile_dir.parent)
            durable = True
        except OSError as error:
            raise PublicationUncertain(expected_profile_id, "parent_durability") from error
    verified = verify_profile_directory(profile_dir, expected_profile_id)
    return {**verified, "state": "PUBLISHED_DURABLE" if durable else "VERIFIED_DURABILITY_UNPROVEN"}


def _success_response(result: dict[str, str]) -> tuple[int, dict[str, Any]]:
    if result["state"] == "PUBLISHED_DURABLE":
        return 0, {"status": "ok", "profile_id": result["profile_id"], "durability": "durable"}
    return 3, {"status": "verified", "profile_id": result["profile_id"], "durability": "unproven_platform"}


def main(argv: list[str] | None = None) -> int:
    try:
        parser = SafeArgumentParser(description="Create or verify offline tokenizer-profile evidence.")
        subparsers = parser.add_subparsers(dest="operation", required=True)
        create = subparsers.add_parser("create")
        create.add_argument("--config", required=True, type=Path)
        create.add_argument("--output-prefix", required=True, type=Path)
        verify = subparsers.add_parser("verify")
        verify.add_argument("--profile-dir", required=True, type=Path)
        verify.add_argument("--expected-profile-id", required=True)
        reconcile = subparsers.add_parser("reconcile")
        reconcile.add_argument("--profile-dir", required=True, type=Path)
        reconcile.add_argument("--expected-profile-id", required=True)
        args = parser.parse_args(argv)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if args.operation == "create":
                response_code, response = _success_response(publish_profile(profile(load_config(args.config)), args.output_prefix))
            elif args.operation == "verify":
                verified = verify_profile_directory(args.profile_dir, args.expected_profile_id)
                response_code, response = 0, {"status": "verified", "profile_id": verified["profile_id"], "durability": "not_assessed"}
            else:
                response_code, response = _success_response(reconcile_profile(args.profile_dir, args.expected_profile_id))
        print(_canonical_json(response))
        return response_code
    except PublicationUncertain as error:
        print(_canonical_json({"status": "reconcile_required", "profile_id": error.profile_id, "phase": error.phase}))
        return 3
    except ProfilerError as error:
        print(_canonical_json({"status": "error", "error": {"code": error.code}}))
        return 2
    except Exception:
        print(_canonical_json({"status": "error", "error": {"code": "PROFILE_FAILED"}}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
