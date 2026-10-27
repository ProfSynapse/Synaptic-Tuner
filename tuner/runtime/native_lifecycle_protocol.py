"""Stdlib-only bounded numeric lifecycle frames for trusted native supervision.

This codec neither opens a journal nor grants execution/capability authority.
The host must bind the exact policy and invocation before deriving a one-attempt
key. Only trusted supervision receives that key, through an anonymous channel;
the ML worker must never receive it. Authenticated prefixes establish integrity,
not freshness or completeness. A cold terminal requires a complete final frame.
"""

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
import stat


MAXIMUM_RECORD_BYTES = 4096
_FRAME_DOMAIN = b"synaptic-native-lifecycle-frame/v1\0"
_PHASES = frozenset({"preparation", "training", "validation", "reload", "publication"})
_METRICS = frozenset({"loss", "gradient_norm", "learning_rate", "tokens"})
_REASONS = frozenset({"worker_exited", "cancel_requested", "owner_lost", "deadline", "output_limit"})


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("native lifecycle duplicate field")
        value[key] = item
    return value


def _parse(raw):
    value = json.loads(raw, object_pairs_hook=_object,
                       parse_constant=lambda _: (_ for _ in ()).throw(
                           ValueError("native lifecycle nonfinite value")))
    if _canonical(value) != raw:
        raise ValueError("native lifecycle noncanonical bytes")
    return value


def _digest(value):
    if (type(value) is not str or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("native lifecycle digest is invalid")


def _key(value):
    if type(value) is not bytes or len(value) != 32:
        raise ValueError("native lifecycle requires an exact derived key")


@dataclass(frozen=True)
class NativeLifecyclePolicyV1:
    maximum_progress_records: int
    maximum_journal_bytes: int

    def __post_init__(self):
        if (type(self.maximum_progress_records) is not int
                or not 1 <= self.maximum_progress_records <= 65536
                or type(self.maximum_journal_bytes) is not int
                or not 8192 <= self.maximum_journal_bytes <= 16 * 1024 * 1024):
            raise ValueError("native lifecycle policy bounds are invalid")

    def to_dict(self):
        return {"schema": "synaptic-native-lifecycle-policy/v1",
                "maximum_progress_records": self.maximum_progress_records,
                "maximum_journal_bytes": self.maximum_journal_bytes}

    @property
    def digest(self):
        return hashlib.sha256(_canonical(self.to_dict())).hexdigest()


def validate_event(event):
    """Closed fields only; never accept diagnostic strings or worker text."""
    if type(event) is not dict:
        raise ValueError("native lifecycle event is invalid")
    if event.get("kind") == "progress":
        if (set(event) != {"kind", "phase", "step", "metrics"}
                or type(event["phase"]) is not str or event["phase"] not in _PHASES
                or type(event["step"]) is not int or not 0 <= event["step"] <= 1000000
                or type(event["metrics"]) is not dict or not set(event["metrics"]) <= _METRICS):
            raise ValueError("native lifecycle progress fields are invalid")
        for name, number in event["metrics"].items():
            if type(number) not in (int, float) or not -1e30 <= number <= 1e30 or not math.isfinite(number):
                raise ValueError("native lifecycle metric is invalid")
            if name == "tokens" and (type(number) is not int or not 0 <= number <= 2**63 - 1):
                raise ValueError("native lifecycle token count is invalid")
            if name in {"gradient_norm", "learning_rate"} and number < 0:
                raise ValueError("native lifecycle metric is invalid")
    elif event.get("kind") == "terminal":
        outcome = event.get("outcome")
        if (set(event) != {"kind", "outcome"} or type(outcome) is not dict
                or set(outcome) != {"schema", "reason", "exit_code", "termination_confirmed",
                                    "stdout_bytes", "stderr_bytes"}
                or outcome["schema"] != "synaptic-native-process-outcome/v1"
                or type(outcome["reason"]) is not str or outcome["reason"] not in _REASONS
                or type(outcome["exit_code"]) is not int or not -(2**31) <= outcome["exit_code"] < 2**31
                or outcome["termination_confirmed"] is not True
                or any(type(outcome[name]) is not int or not 0 <= outcome[name] <= (16 << 20) + 65536
                       for name in ("stdout_bytes", "stderr_bytes"))):
            raise ValueError("native lifecycle terminal fields are invalid")
    else:
        raise ValueError("native lifecycle event kind is invalid")


def encode_worker_progress(event):
    validate_event(event)
    raw = _canonical(event) + b"\n"
    if event["kind"] != "progress" or len(raw) > MAXIMUM_RECORD_BYTES:
        raise ValueError("native worker progress is invalid")
    return raw


def parse_worker_progress(raw):
    if type(raw) is not bytes or not raw or len(raw) >= MAXIMUM_RECORD_BYTES:
        raise ValueError("native worker progress bound is invalid")
    event = _parse(raw)
    if encode_worker_progress(event) != raw + b"\n":
        raise ValueError("native worker progress bytes differ")
    return event


class NativeLifecycleWriter:
    """Append/fsync complete frames to a caller-owned create-only descriptor.

    The caller retains the descriptor and its private root/path identity. This
    writer never opens paths or stores the derived key in a file. An interrupted
    write poisons this writer; it cannot resume or append after that partial tail.
    """

    def __init__(self, fd, *, invocation_digest, policy, key):
        _digest(invocation_digest)
        _key(key)
        if type(fd) is not int or type(policy) is not NativeLifecyclePolicyV1:
            raise TypeError("exact native lifecycle writer inputs required")
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1 or info.st_size != 0
                or os.lseek(fd, 0, os.SEEK_CUR) != 0):
            raise ValueError("native lifecycle descriptor is not fresh and private")
        self._fd, self._key = fd, key
        self._identity = info.st_dev, info.st_ino
        self._invocation, self._policy = invocation_digest, policy
        self._sequence, self._size, self._previous = 0, 0, "0" * 64
        self._terminal = self._failed = False

    def _check(self, size):
        info = os.fstat(self._fd)
        if (not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != self._identity
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1 or info.st_size != size
                or os.lseek(self._fd, 0, os.SEEK_CUR) != size):
            raise ValueError("native lifecycle retained descriptor changed")

    def append(self, event):
        if self._terminal or self._failed:
            raise ValueError("native lifecycle writer is closed")
        validate_event(event)
        terminal = event["kind"] == "terminal"
        content = {"schema": "synaptic-native-lifecycle-frame/v1",
                   "invocation_digest": self._invocation, "policy_digest": self._policy.digest,
                   "sequence": self._sequence, "previous_digest": self._previous, "event": event}
        tag = hmac.new(self._key, _FRAME_DOMAIN + _canonical(content), hashlib.sha256).hexdigest()
        raw = _canonical({"content": content, "tag": tag}) + b"\n"
        reserve = 0 if terminal else MAXIMUM_RECORD_BYTES
        if (len(raw) > MAXIMUM_RECORD_BYTES
                or self._size + len(raw) + reserve > self._policy.maximum_journal_bytes
                or not terminal and self._sequence >= self._policy.maximum_progress_records):
            raise ValueError("native lifecycle publication bound exceeded")
        try:
            self._check(self._size)
            offset = 0
            while offset < len(raw):
                written = os.write(self._fd, raw[offset:])
                if written <= 0:
                    raise ValueError("native lifecycle short publication")
                offset += written
            os.fsync(self._fd)
            self._check(self._size + len(raw))
        except BaseException:
            self._failed = True
            raise
        self._size += len(raw)
        self._sequence += 1
        self._previous = hashlib.sha256(raw).hexdigest()
        self._terminal = terminal


@dataclass(frozen=True)
class NativeLifecycleView:
    event_bytes: tuple[bytes, ...]
    committed_bytes: int
    prefix_digest: str
    partial_tail: bool
    terminal_bytes: bytes | None


def verify_lifecycle_journal(raw, *, invocation_digest, policy, key, previous_raw=None):
    """Verify an exact captured snapshot; partial tails never establish terminal.

    Path custody and retention of the last snapshot are host responsibilities.
    Supply that snapshot as previous_raw to reject rollback or prefix replacement;
    an earlier uncommitted partial tail may change.
    This return value is data, not a sealed public capability or execution grant.
    """
    _digest(invocation_digest)
    _key(key)
    if type(policy) is not NativeLifecyclePolicyV1 or type(raw) is not bytes:
        raise TypeError("exact native lifecycle reader inputs required")
    if len(raw) > policy.maximum_journal_bytes:
        raise ValueError("native lifecycle journal bound exceeded")
    parts = raw.split(b"\n")
    tail = parts.pop()
    if len(tail) >= MAXIMUM_RECORD_BYTES or len(parts) > policy.maximum_progress_records + 1:
        raise ValueError("native lifecycle record bound exceeded")
    previous, size, terminal = "0" * 64, 0, None
    events = []
    for sequence, part in enumerate(parts):
        if terminal is not None or not part or len(part) + 1 > MAXIMUM_RECORD_BYTES:
            raise ValueError("native lifecycle frame boundary is invalid")
        frame = _parse(part)
        if type(frame) is not dict or set(frame) != {"content", "tag"}:
            raise ValueError("native lifecycle envelope fields are invalid")
        content = frame["content"]
        _digest(frame["tag"])
        if (type(content) is not dict or set(content) != {"schema", "invocation_digest", "policy_digest",
                                                       "sequence", "previous_digest", "event"}
                or content["schema"] != "synaptic-native-lifecycle-frame/v1"
                or content["invocation_digest"] != invocation_digest or content["policy_digest"] != policy.digest
                or type(content["sequence"]) is not int or content["sequence"] != sequence
                or content["previous_digest"] != previous or type(frame["tag"]) is not str
                or not hmac.compare_digest(frame["tag"], hmac.new(
                    key, _FRAME_DOMAIN + _canonical(content), hashlib.sha256).hexdigest())):
            raise ValueError("native lifecycle frame authentication differs")
        event = content["event"]
        validate_event(event)
        size += len(part) + 1
        if event["kind"] == "terminal":
            terminal = _canonical(event["outcome"])
        else:
            if (sequence >= policy.maximum_progress_records
                    or size + MAXIMUM_RECORD_BYTES > policy.maximum_journal_bytes):
                raise ValueError("native lifecycle progress bound exceeded")
        events.append(_canonical(event))
        previous = hashlib.sha256(part + b"\n").hexdigest()
    if terminal is not None and tail:
        raise ValueError("native lifecycle bytes follow terminal")
    if previous_raw is not None:
        previous_view = verify_lifecycle_journal(
            previous_raw, invocation_digest=invocation_digest, policy=policy, key=key)
        committed = previous_view.committed_bytes
        if len(raw) < committed or raw[:committed] != previous_raw[:committed]:
            raise ValueError("native lifecycle committed prefix changed")
    return NativeLifecycleView(tuple(events), size, hashlib.sha256(raw[:size]).hexdigest(), bool(tail), terminal)
