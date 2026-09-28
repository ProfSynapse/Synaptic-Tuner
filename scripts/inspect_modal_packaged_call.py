"""Read-only, non-authorizing status for one retained Modal training call.

The exact private submit claim, packaged command binding and call catalog must
agree before a Modal client is created. Provider result bytes are never loaded
as Python objects. This diagnostic cannot verify a training run or grant replay.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tuner.execution.foundation_v2.canonical import canonical_bytes, parse_canonical_object, safe_ref
from tuner.execution.foundation_v2.commands import SubmitCommandV2
from tuner.training.modal_host_effects import (
    ModalPackagedMarkerMaterial, _decode_binding, _decode_marker_materials,
    _encode_binding, _encode_marker_materials,
)
from tuner.execution.providers.modal.bounded_volume_read import (
    BoundedModalVolumeReader, BoundedVolumeReadError,
)


_NAMESPACE = "standalone-training"
_BINDINGS = "packaged-bindings-v1"
_CALLS = "packaged-calls-v1"
_MARKERS = "packaged-marker-materials-v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MAX_CLAIM = 16 * 1024
_MAX_BINDING = 16 * 1024 * 1024
_MAX_CALL = 1024
_MAX_MARKERS = 4096
_MAX_FIXED_RESULT = 512
_RESULT_SCHEMA = "synaptic-modal-packaged-worker-result/v1"
_RESULT_SCHEMA_V2 = "synaptic-modal-packaged-worker-result/v2"
_WORKER_FAILURE_STAGES = (
    "ENTRYPOINT_SETUP", "ENTRYPOINT_IMPORTS", "ENTRYPOINT_DISPATCH_AUTH",
    "ENTRYPOINT_PROVIDER_ID", "ENTRYPOINT_VOLUME_ID", "ENTRYPOINT_CALL_ID",
    "ENTRYPOINT_MOUNTS", "ENTRYPOINT_WORKER_SETUP",
    "ENTRYPOINT_MOUNT_CONTROL_DIR", "ENTRYPOINT_MOUNT_CONTROL_LINK",
    "ENTRYPOINT_MOUNT_ARTIFACTS_DIR", "ENTRYPOINT_MOUNT_ARTIFACTS_LINK",
    "ENTRYPOINT_MOUNT_MODEL_CACHE_DIR", "ENTRYPOINT_MOUNT_MODEL_CACHE_LINK",
    "DISPATCH_AUTH", "STAGED_INPUT", "PATH_CLAIM",
    "SFT_ADMISSION", "SFT_ADMISSION_CONTRACTS", "SFT_ADMISSION_RELEASE",
    "SFT_ADMISSION_PATHS", "SFT_ADMISSION_INPUT", "SFT_ADMISSION_ENVIRONMENT",
    "SFT_ADMISSION_INVOCATION", "SFT_ADMISSION_COMMITMENT",
    "SFT_PREPARATION", "SFT_REVALIDATION",
    "SFT_INVOCATION", "SFT_TRAINER", "SFT_EVIDENCE", "SFT_ARTIFACT",
    "SFT_UNKNOWN", "COMPLETION", "ARTIFACT_COMMIT", "CONTROL_COMMIT",
)


class DiagnosticUnavailable(RuntimeError):
    """A closed local failure; underlying exception text must not escape."""


def _private_regular(info: os.stat_result) -> bool:
    return (stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            and info.st_uid == os.geteuid()
            and not stat.S_IMODE(info.st_mode) & 0o077)


def _catalog_bytes(connection, catalog: str, item: str, maximum: int) -> bytes:
    rows = connection.execute(
        "SELECT payload,digest FROM catalogs WHERE namespace_ref=? AND catalog_ref=? "
        "AND item_ref=? AND length(payload)<=?",
        (_NAMESPACE, catalog, item, maximum),
    ).fetchmany(2)
    if len(rows) != 1:
        raise ValueError
    payload, digest = rows[0]
    if (type(payload) is not bytes or not payload or len(payload) > maximum
            or type(digest) is not str
            or hashlib.sha256(payload).hexdigest() != digest):
        raise ValueError
    return payload


def _read_retained(database: Path, claim_ref: str,
                   supplied_call_id: str | None) -> str | tuple[ModalPackagedMarkerMaterial, ...]:
    """Authenticate a submit claim and exactly one diagnostic catalog read-only."""
    descriptor = None
    try:
        if (os.name != "posix" or not database.is_absolute()
                or type(claim_ref) is not str or _HEX.fullmatch(claim_ref) is None
                or (supplied_call_id is not None and (
                    type(supplied_call_id) is not str
                    or not supplied_call_id.startswith("fc-")
                    or len(supplied_call_id) > _MAX_CALL
                    or safe_ref(supplied_call_id, "provider_job_ref") != supplied_call_id))):
            raise ValueError
        parent = database.parent
        parent_info = parent.lstat()
        if (not stat.S_ISDIR(parent_info.st_mode)
                or stat.S_ISLNK(parent_info.st_mode)
                or parent.resolve(strict=True) != parent
                or parent_info.st_uid != os.geteuid()
                or stat.S_IMODE(parent_info.st_mode) & 0o077):
            raise ValueError
        descriptor = os.open(database, os.O_RDONLY | os.O_NOFOLLOW)
        info = os.fstat(descriptor)
        if (not _private_regular(info) or database.is_symlink()
                or (database.lstat().st_dev, database.lstat().st_ino)
                != (info.st_dev, info.st_ino)):
            raise ValueError
        uri = f"file:/proc/self/fd/{descriptor}?mode=ro&immutable=1"
        with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=2.0)) as connection:
            connection.execute("PRAGMA query_only=ON")
            rows = connection.execute(
                "SELECT digest,evidence FROM attempts WHERE namespace_ref=? "
                "AND attempt_ref=? AND length(evidence)<=?",
                (_NAMESPACE, claim_ref, _MAX_CLAIM),
            ).fetchmany(2)
            if len(rows) != 1:
                raise ValueError
            claim_digest, raw_claim = rows[0]
            if (type(claim_digest) is not str or type(raw_claim) is not bytes
                    or not raw_claim or len(raw_claim) > _MAX_CLAIM
                    or hashlib.sha256(raw_claim).hexdigest() != claim_digest):
                raise ValueError
            claim = parse_canonical_object(raw_claim, name="packaged claim")
            if (canonical_bytes(claim) != raw_claim or set(claim) != {
                    "schema_version", "command_digest", "binding_digest",
            } or claim["schema_version"] != "synaptic-modal-packaged-host-attempt/v1"
                    or claim["command_digest"] != claim_ref
                    or type(claim["binding_digest"]) is not str
                    or _HEX.fullmatch(claim["binding_digest"]) is None):
                raise ValueError
            raw_binding = _catalog_bytes(connection, _BINDINGS, claim_ref, _MAX_BINDING)
            binding = _decode_binding(raw_binding)
            if (_encode_binding(binding) != raw_binding
                    or type(binding.command) is not SubmitCommandV2
                    or binding.command_digest != claim_ref
                    or binding.authenticated_binding_digest != claim["binding_digest"]):
                raise ValueError
            if supplied_call_id is None:
                raw_markers = _catalog_bytes(connection, _MARKERS, claim_ref, _MAX_MARKERS)
                materials = _decode_marker_materials(raw_markers)
                facts = binding.provider_facts
                expected = {"control": facts.control_volume_id,
                            "artifacts": facts.artifact_volume_id}
                if facts.model_cache_volume_id is not None:
                    expected["model_cache"] = facts.model_cache_volume_id
                commitments = tuple(item.commitment for item in materials)
                if (_encode_marker_materials(materials) != raw_markers
                        or tuple(item.role for item in commitments) != tuple(expected)
                        or any(item.volume_id != expected[item.role]
                               for item in commitments)
                        or len({item.volume_id for item in commitments}) != len(expected)
                        or len({item.marker_name for item in commitments}) != len(expected)
                        or len({item.value_sha256 for item in commitments}) != len(expected)):
                    raise ValueError
            else:
                raw_call = _catalog_bytes(connection, _CALLS, claim_ref, _MAX_CALL)
                call = parse_canonical_object(raw_call, name="packaged call")
                if (canonical_bytes(call) != raw_call or set(call) != {"provider_job_ref"}
                        or type(call["provider_job_ref"]) is not str
                        or call["provider_job_ref"] != supplied_call_id):
                    raise ValueError
        retained = os.fstat(descriptor)
        current = database.lstat()
        if ((retained.st_dev, retained.st_ino) != (info.st_dev, info.st_ino)
                or (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino)
                or not _private_regular(current)):
            raise ValueError
        return materials if supplied_call_id is None else supplied_call_id
    except Exception:
        raise DiagnosticUnavailable("JOURNAL_INVALID") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def read_retained_call(database: Path, claim_ref: str, supplied_call_id: str) -> str:
    """Authenticate one submit claim, full binding and exact call without writes."""
    result = _read_retained(database, claim_ref, supplied_call_id)
    assert type(result) is str
    return result


def read_retained_markers(database: Path, claim_ref: str) -> tuple[ModalPackagedMarkerMaterial, ...]:
    """Authenticate exact marker material against the submitted Volume facts."""
    result = _read_retained(database, claim_ref, None)
    assert type(result) is tuple
    return result


def _pinned_python() -> bool:
    return (sys.implementation.name == "cpython"
            and sys.version_info[:3] == (3, 11, 14))


def _classify_fixed_failure(output: object, api_pb2: object, serialize: object) -> str:
    """Compare raw bytes to the fixed failure objects; never deserialize them."""
    try:
        result = output.result
        if (not _pinned_python() or output.data_format != api_pb2.DATA_FORMAT_PICKLE
                or type(result.data_blob_id) is not str or result.data_blob_id
                or type(result.data) is not bytes or not result.data
                or len(result.data) > _MAX_FIXED_RESULT or not callable(serialize)):
            return "PROVIDER_SUCCESS_UNKNOWN"
        expected = serialize({
            "schema_version": _RESULT_SCHEMA,
            "effect_id": "unavailable",
            "status_code": "failed",
            "completion_sha256": "0" * 64,
        })
        if (type(expected) is bytes and len(expected) <= _MAX_FIXED_RESULT
                and hmac.compare_digest(result.data, expected)):
            return "WORKER_FAILED"
        for stage in _WORKER_FAILURE_STAGES:
            expected = serialize({
                "schema_version": _RESULT_SCHEMA_V2,
                "effect_id": "unavailable",
                "status_code": "failed",
                "completion_sha256": "0" * 64,
                "failure_stage": stage,
            })
            if (type(expected) is bytes and len(expected) <= _MAX_FIXED_RESULT
                    and hmac.compare_digest(result.data, expected)):
                return f"WORKER_{stage}"
    except Exception:
        pass
    return "PROVIDER_SUCCESS_UNKNOWN"


async def inspect_call(client: object, call_id: str, api_pb2: object,
                       serialize: object) -> str:
    """Poll one raw response; never unpickle or consume provider output."""
    request = api_pb2.FunctionGetOutputsRequest(
        function_call_id=call_id, timeout=0, last_entry_id="0-0",
        clear_on_success=False, requested_at=time.time(),
        start_idx=0, end_idx=0, max_values=1,
    )
    try:
        response = await asyncio.wait_for(
            client.stub.FunctionGetOutputs(request, retry=None, timeout=15), timeout=16,
        )
    except Exception:
        return "POLL_UNAVAILABLE"
    try:
        if (type(response) is not api_pb2.FunctionGetOutputsResponse
                or type(response.num_unfinished_inputs) is not int
                or response.num_unfinished_inputs < 0):
            return "INVALID_RESPONSE"
        if len(response.outputs) == 0:
            return "PENDING" if response.num_unfinished_inputs > 0 else "OUTPUT_EXPIRED"
        if len(response.outputs) != 1 or response.outputs[0].idx != 0:
            return "INVALID_RESPONSE"
        status = response.outputs[0].result.status
        if status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
            return _classify_fixed_failure(response.outputs[0], api_pb2, serialize)
        if status in {
                api_pb2.GenericResult.GENERIC_STATUS_FAILURE,
                api_pb2.GenericResult.GENERIC_STATUS_TERMINATED,
                api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT,
                api_pb2.GenericResult.GENERIC_STATUS_INIT_FAILURE,
                api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE,
                api_pb2.GenericResult.GENERIC_STATUS_IDLE_TIMEOUT,
                api_pb2.GenericResult.GENERIC_STATUS_MEMORY_MANAGER_EVICTION,
        }:
            return "PROVIDER_FAILURE"
        return "INVALID_RESPONSE"
    except Exception:
        return "INVALID_RESPONSE"


def _proven_not_found(error: BoundedVolumeReadError) -> bool:
    if error.args != ("modal_volume_range_unavailable",):
        return False
    try:
        from grpclib.const import Status
        from grpclib.exceptions import GRPCError
        cause = error.__context__
        return type(cause) is GRPCError and cause.status is Status.NOT_FOUND
    except Exception:
        return False


async def inspect_markers(reader: BoundedModalVolumeReader,
                          materials: tuple[ModalPackagedMarkerMaterial, ...]) -> dict[str, str]:
    """Read only the exact claim-bound 32-byte files; report fixed categories."""
    results = {}
    for material in materials:
        commitment = material.commitment
        try:
            await asyncio.wait_for(reader.read_exact(
                volume_id=commitment.volume_id, path=commitment.marker_name,
                expected_size=32, expected_sha256=commitment.value_sha256,
                max_bytes=32,
            ), timeout=50)
            results[commitment.role] = "MATCH"
        except BoundedVolumeReadError as error:
            results[commitment.role] = "NOT_FOUND" if _proven_not_found(error) else "UNAVAILABLE"
        except Exception:
            results[commitment.role] = "UNAVAILABLE"
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--claim-ref", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--call-id")
    selection.add_argument("--inspect-markers", action="store_true")
    parser.add_argument("--modal-profile", required=True)
    args = parser.parse_args(argv)
    try:
        retained = (read_retained_markers(args.journal, args.claim_ref)
                    if args.inspect_markers else
                    read_retained_call(args.journal, args.claim_ref, args.call_id))
        profile = safe_ref(args.modal_profile, "modal_profile")
        with open(os.devnull, "w") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                import modal
                from modal.config import config

                if modal.__version__ != "1.5.4":
                    raise DiagnosticUnavailable("SDK_INVALID")
                token_id = config.get("token_id", profile=profile, use_env=False)
                token_secret = config.get("token_secret", profile=profile, use_env=False)
                if any(type(value) is not str or not value.strip()
                       for value in (token_id, token_secret)):
                    raise DiagnosticUnavailable("CREDENTIAL_UNAVAILABLE")
                from modal._utils.async_utils import synchronizer
                client = modal.Client.from_credentials(token_id, token_secret)
                if args.inspect_markers:
                    reader = BoundedModalVolumeReader(sdk=modal, client=client)
                    category = synchronizer.create_blocking(inspect_markers)(reader, retained)
                else:
                    from modal._serialization import serialize
                    from modal_proto import api_pb2
                    category = synchronizer.create_blocking(inspect_call)(
                        client, retained, api_pb2, serialize,
                    )
    except DiagnosticUnavailable as error:
        category = error.args[0]
    except Exception:
        category = "LOCAL_UNAVAILABLE"
    if args.inspect_markers:
        payload = {"schema_version": "synaptic-modal-packaged-marker-diagnostic/v1",
                   "authority": "DIAGNOSTIC_ONLY", "result": category}
    else:
        payload = {"schema_version": "synaptic-modal-packaged-call-diagnostic/v1",
                   "authority": "DIAGNOSTIC_ONLY", "result": category}
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    if args.inspect_markers:
        return 0 if type(category) is dict and all(
            value == "MATCH" for value in category.values()) else 1
    return 0 if category in {
        "PENDING", "PROVIDER_SUCCESS_UNKNOWN", "WORKER_FAILED", "PROVIDER_FAILURE",
    } or category in {f"WORKER_{stage}" for stage in _WORKER_FAILURE_STAGES} else 1


if __name__ == "__main__":
    sys.exit(main())
