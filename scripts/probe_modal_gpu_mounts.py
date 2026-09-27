"""One-shot private Modal GPU mount-topology diagnostic.

This maintenance probe is deliberately outside TrainingAPI. It creates three
empty v1 Volumes and one private L40S Function using an existing exact Image.
Its result is diagnostic only and cannot authorize training or a mount guard
change. The claim and provider resources are retained on every outcome.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import itertools
import json
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import sys
import tempfile
import time


_NAME = re.compile(r"[a-z][a-z0-9-]{0,62}[a-z0-9]\Z")
_PREFIX = re.compile(r"[a-z][a-z0-9-]{0,17}[a-z0-9]\Z")
_IMAGE = re.compile(r"im-[A-Za-z0-9]{1,64}\Z")
_VOLUME = re.compile(r"vo-[A-Za-z0-9]{1,64}\Z")
_CALL = re.compile(r"fc-[A-Za-z0-9]{1,77}\Z")
_ROLES = ("control", "artifacts", "model_cache")
_MOUNTS = ("/mnt/control", "/mnt/artifacts", "/mnt/model-cache")
_CATEGORIES = ("DIRECTORY", "LINK", "ABSENT", "OTHER", "UNAVAILABLE")
_SCHEMA = "synaptic-modal-gpu-mount-probe/v1"
_TIMEOUT = 120
_STAGE_FAILURES = frozenset({
    "APP_CONSTRUCT_UNAVAILABLE",
    "VOLUME_CREATE_UNAVAILABLE", "VOLUME_HYDRATE_UNAVAILABLE",
    "IMAGE_HANDLE_UNAVAILABLE", "FUNCTION_CONSTRUCT_UNAVAILABLE",
    "APP_DEPLOY_UNAVAILABLE", "FUNCTION_SPAWN_UNAVAILABLE",
    "CALL_RETENTION_UNAVAILABLE", "POLL_UNAVAILABLE",
})
_SDK_ORIGIN_MODULES = {
    "modal.app": "app.py",
    "modal.runner": "runner.py",
    "modal._functions": "_functions.py",
    "modal.image": "image.py",
    "modal._resolver": "_resolver.py",
    "modal._object": "_object.py",
    "modal.mount": "mount.py",
    "modal.client": "client.py",
}
_FUNCTION_CREATE_TOPICS = (
    ("VOLUME", frozenset({"volume", "volumes"})),
    ("MOUNT", frozenset({"mount", "mounts"})),
    ("IMAGE", frozenset({"image", "images"})),
    ("GPU", frozenset({"gpu", "gpus"})),
    ("SECRET", frozenset({"secret", "secrets"})),
    ("NETWORK", frozenset({"network", "networks"})),
    ("MEMORY", frozenset({"memory"})),
    ("CPU", frozenset({"cpu", "cpus"})),
    ("TIMEOUT", frozenset({"timeout"})),
    ("FUNCTION", frozenset({"function", "functions"})),
)
_FUNCTION_CREATE_ACTIONS = (
    ("UNSUPPORTED", frozenset({"unsupported"})),
    ("INVALID", frozenset({"invalid"})),
    ("MISSING", frozenset({"missing", "required"})),
    ("DUPLICATE", frozenset({"duplicate"})),
    ("LIMIT", frozenset({"limit", "exceeded"})),
    ("PERMISSION", frozenset({"permission", "forbidden"})),
    ("CONFLICT", frozenset({"conflict"})),
)


class ProbeUnavailable(RuntimeError):
    """Closed diagnostic failure whose underlying details must stay private."""

    def __init__(self, code: str, *, failure_class: str | None = None,
                 failure_origin: str | None = None,
                 failure_topics: tuple[str, ...] = (),
                 failure_action: str | None = None):
        super().__init__(code)
        self.failure_class = failure_class
        self.failure_origin = failure_origin
        self.failure_topics = failure_topics
        self.failure_action = failure_action


def _failure_class(error: BaseException) -> str:
    kind = type(error)
    admitted = {
        TypeError, ValueError, RuntimeError, TimeoutError, PermissionError,
        AttributeError, ImportError, ModuleNotFoundError, OSError,
    }
    if kind in admitted:
        return kind.__name__
    module = getattr(kind, "__module__", "")
    name = getattr(kind, "__name__", "")
    if module.startswith("modal.") and name in {
        "InvalidError", "ExecutionError", "SerializationError", "NotFoundError",
    }:
        return name
    if module.startswith("grpclib.") and name == "GRPCError":
        return name
    return "OTHER"


def _failure_origin(error: BaseException) -> str | None:
    """Expose one admitted SDK source site, never a traceback path or message."""
    trace = error.__traceback__
    origin = None
    for _ in range(64):
        if trace is None:
            break
        frame = trace.tb_frame
        module = frame.f_globals.get("__name__")
        filename = _SDK_ORIGIN_MODULES.get(module) if type(module) is str else None
        path = frame.f_code.co_filename.replace("\\", "/")
        line = trace.tb_lineno
        if (filename is not None and path.endswith("/modal/" + filename)
                and type(line) is int and 0 < line <= 10000):
            origin = f"{filename}:{line}"
        trace = trace.tb_next
    return origin


def _function_create_topics(error: BaseException) -> tuple[tuple[str, ...], str | None]:
    """Return advisory keyword hints, not causes; never emit provider text."""
    try:
        words = set(re.findall(r"[a-z_]{2,32}", str(error)[:2048].lower()))
    except Exception:
        return (), None
    topics = tuple(label for label, names in _FUNCTION_CREATE_TOPICS if words & names)[:4]
    action = next((label for label, names in _FUNCTION_CREATE_ACTIONS if words & names), None)
    return topics, action


@contextlib.contextmanager
def _provider_stage(code: str):
    assert code in _STAGE_FAILURES
    try:
        yield
    except ProbeUnavailable as error:
        failure_class = ("TimeoutError" if str(error) == "PROVIDER_INDETERMINATE"
                         else "OTHER")
        raise ProbeUnavailable(code, failure_class=failure_class) from None
    except Exception as error:
        failure_class = _failure_class(error)
        failure_origin = _failure_origin(error)
        topics, action = ((), None)
        if (code == "APP_DEPLOY_UNAVAILABLE" and failure_class == "InvalidError"
                and type(error).__module__ == "modal.exception"
                and failure_origin == "_functions.py:1148"):
            topics, action = _function_create_topics(error)
        raise ProbeUnavailable(
            code, failure_class=failure_class, failure_origin=failure_origin,
            failure_topics=topics, failure_action=action,
        ) from None


def _remote_probe() -> dict[str, object]:
    """Inspect only the mount roots through one retained /mnt descriptor."""
    import os
    import stat

    categories = []
    parent = None
    try:
        parent = os.open("/mnt", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for leaf in ("control", "artifacts", "model-cache"):
            try:
                info = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                categories.append("ABSENT")
                continue
            except BaseException:
                categories.append("UNAVAILABLE")
                continue
            if stat.S_ISLNK(info.st_mode):
                categories.append("LINK")
            elif stat.S_ISDIR(info.st_mode):
                descriptor = None
                try:
                    descriptor = os.open(
                        leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=parent,
                    )
                    opened = os.fstat(descriptor)
                    categories.append(
                        "DIRECTORY" if (opened.st_dev, opened.st_ino)
                        == (info.st_dev, info.st_ino) else "UNAVAILABLE"
                    )
                except BaseException:
                    categories.append("UNAVAILABLE")
                finally:
                    if descriptor is not None:
                        os.close(descriptor)
            else:
                categories.append("OTHER")
    except BaseException:
        categories = ["UNAVAILABLE"] * 3
    finally:
        if parent is not None:
            os.close(parent)
    return {"schema_version": _SCHEMA, "roots": tuple(categories)}


def _exact(value: str, pattern: re.Pattern[str]) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ProbeUnavailable("INPUT_INVALID")
    return value


def _require_host() -> None:
    if (os.name != "posix" or sys.implementation.name != "cpython"
            or sys.version_info[:3] != (3, 11, 14)
            or sys.version_info.releaselevel != "final"):
        raise ProbeUnavailable("HOST_INCOMPATIBLE")


def _private_directory(path: Path) -> int:
    if not path.is_absolute():
        raise ProbeUnavailable("CLAIM_PATH_INVALID")
    descriptor = None
    try:
        descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        parts = path.parts[1:]
        if not parts:
            raise ValueError
        for index, part in enumerate(parts):
            before = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            opened = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            after = os.fstat(opened)
            if (not stat.S_ISDIR(before.st_mode)
                    or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                    or before.st_uid not in (0, os.geteuid())
                    or stat.S_IMODE(before.st_mode) & 0o022
                    or (index == len(parts) - 1 and (
                        before.st_uid != os.geteuid()
                        or stat.S_IMODE(before.st_mode) & 0o077
                    ))):
                os.close(opened)
                raise ValueError
            os.close(descriptor)
            descriptor = opened
        return descriptor
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise ProbeUnavailable("CLAIM_PATH_INVALID")


def _exclusive_record(root: int, leaf: str, value: dict[str, object]) -> None:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(raw) > 4096:
        raise ProbeUnavailable("CLAIM_INVALID")
    descriptor = None
    try:
        descriptor = os.open(
            leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600, dir_fd=root,
        )
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short claim write")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.fsync(root)
    except FileExistsError:
        raise ProbeUnavailable("CLAIM_ALREADY_CONSUMED") from None
    except Exception:
        raise ProbeUnavailable("CLAIM_INDETERMINATE") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _bounded_alarm(_signum, _frame):
    raise ProbeUnavailable("PROVIDER_INDETERMINATE")


@contextlib.contextmanager
def _deadline(seconds: int):
    former = signal.signal(signal.SIGALRM, _bounded_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, former)


def _classify_raw(output: object, api_pb2: object, serialize: object) -> str | dict[str, str]:
    """Compare only pinned small inline pickle bytes; never deserialize them."""
    try:
        if output.result.status != api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
            return "PROVIDER_FAILURE"
        if (output.data_format != api_pb2.DATA_FORMAT_PICKLE
                or output.result.data_blob_id or type(output.result.data) is not bytes
                or len(output.result.data) > 512):
            return "OUTPUT_UNCLASSIFIED"
        raw = output.result.data
        for values in itertools.product(_CATEGORIES, repeat=3):
            expected = serialize({"schema_version": _SCHEMA, "roots": values})
            if type(expected) is bytes and hmac.compare_digest(raw, expected):
                return dict(zip(_ROLES, values))
    except Exception:
        pass
    return "OUTPUT_UNCLASSIFIED"


async def _poll_raw(client: object, call_id: str, api_pb2: object,
                    serialize: object) -> str | dict[str, str]:
    import asyncio

    request = api_pb2.FunctionGetOutputsRequest(
        function_call_id=call_id, timeout=0, last_entry_id="0-0",
        clear_on_success=False, requested_at=time.time(),
        start_idx=0, end_idx=0, max_values=1,
    )
    try:
        response = await asyncio.wait_for(
            client.stub.FunctionGetOutputs(request, retry=None, timeout=15), timeout=16,
        )
        if len(response.outputs) == 0:
            return "PENDING" if response.num_unfinished_inputs > 0 else "OUTPUT_EXPIRED"
        if len(response.outputs) != 1 or response.outputs[0].idx != 0:
            return "OUTPUT_UNCLASSIFIED"
        return _classify_raw(response.outputs[0], api_pb2, serialize)
    except Exception:
        return "POLL_UNAVAILABLE"


def _client(sdk: object, profile: str):
    from modal.config import config

    if sdk.__version__ != "1.5.4":
        raise ProbeUnavailable("SDK_INCOMPATIBLE")
    token_id = config.get("token_id", profile=profile, use_env=False)
    token_secret = config.get("token_secret", profile=profile, use_env=False)
    if any(type(value) is not str or not value.strip()
           for value in (token_id, token_secret)):
        raise ProbeUnavailable("CREDENTIAL_UNAVAILABLE")
    return sdk.Client.from_credentials(token_id, token_secret)


async def _read_app_absence(client: object, app_name: str,
                            environment: str, api_pb2: object) -> bool:
    import asyncio
    from modal.exception import NotFoundError

    request = api_pb2.AppGetByDeploymentNameRequest(
        name=app_name, environment_name=environment,
    )
    try:
        response = await asyncio.wait_for(
            client.stub.AppGetByDeploymentName(request, retry=None, timeout=15), timeout=16,
        )
    except NotFoundError:
        return True
    return (
        type(response) is api_pb2.AppGetByDeploymentNameResponse
        and response.app_id == ""
        and response.previous_app_id == ""
        and response.lifecycle == api_pb2.AppLifecycle()
        and response.environment_name in ("", environment)
    )


async def _read_image_identity(client: object, image_id: str,
                               api_pb2: object) -> bool:
    """Read the selected existing Image ID before an app ID exists."""
    import asyncio

    request = api_pb2.ImageFromIdRequest(image_id=image_id)
    response = await asyncio.wait_for(
        client.stub.ImageFromId(request, retry=None, timeout=15), timeout=16,
    )
    return (type(response) is api_pb2.ImageFromIdResponse
            and response.image_id == image_id)


def execute(args: argparse.Namespace, sdk: object, client: object) -> str | dict[str, str]:
    """One app deployment and one Function spawn after an exclusive claim."""
    from modal._utils.async_utils import synchronizer
    from modal_proto import api_pb2

    try:
        absent = synchronizer.create_blocking(_read_app_absence)(
            client, args.app, args.environment, api_pb2,
        )
    except Exception:
        raise ProbeUnavailable("APP_ABSENCE_UNAVAILABLE") from None
    if not absent:
        raise ProbeUnavailable("APP_NOT_FRESH")
    try:
        image_matches = synchronizer.create_blocking(_read_image_identity)(
            client, args.image_id, api_pb2,
        )
    except Exception:
        raise ProbeUnavailable("IMAGE_READ_UNAVAILABLE") from None
    if not image_matches:
        raise ProbeUnavailable("IMAGE_IDENTITY_INVALID")
    with _provider_stage("APP_CONSTRUCT_UNAVAILABLE"):
        app = sdk.App(args.app, include_source=False)
    volumes = {}
    for role, name in zip(_ROLES, args.volume_names):
        with _provider_stage("VOLUME_CREATE_UNAVAILABLE"):
            with _deadline(60):
                sdk.Volume.objects.create(
                    name, version=1, allow_existing=False,
                    environment_name=args.environment, client=client,
                )
        with _provider_stage("VOLUME_HYDRATE_UNAVAILABLE"):
            with _deadline(60):
                volume = sdk.Volume.from_name(
                    name, environment_name=args.environment,
                    create_if_missing=False, version=1, client=client,
                )
                volume.hydrate(client)
        identity = getattr(volume, "object_id", None)
        if (getattr(volume, "is_hydrated", False) is not True
                or type(identity) is not str or _VOLUME.fullmatch(identity) is None
                or identity in [item.object_id for item in volumes.values()]):
            raise ProbeUnavailable("VOLUME_IDENTITY_INVALID")
        volumes[role] = volume
    with _provider_stage("IMAGE_HANDLE_UNAVAILABLE"):
        image = sdk.Image.from_id(args.image_id, client=client)
    if getattr(image, "object_id", None) != args.image_id:
        raise ProbeUnavailable("IMAGE_IDENTITY_INVALID")
    with _provider_stage("FUNCTION_CONSTRUCT_UNAVAILABLE"):
        function = app.function(
            name="mount_probe", image=image, cpu=1, memory=4096, gpu="L40S",
            timeout=_TIMEOUT, retries=0,
            volumes=dict(zip(_MOUNTS, (volumes[role] for role in _ROLES))),
            secrets=[], block_network=True, restrict_modal_access=True,
            single_use_containers=True, serialized=False, include_source=True,
        )(_remote_probe)
    with tempfile.TemporaryDirectory(prefix="synaptic-modal-mount-probe-") as directory:
        original = os.getcwd()
        try:
            os.chdir(directory)
            with _provider_stage("APP_DEPLOY_UNAVAILABLE"):
                with _deadline(300):
                    app.deploy(environment_name=args.environment, client=client)
        finally:
            os.chdir(original)
    if (getattr(image, "is_hydrated", False) is not True
            or getattr(image, "object_id", None) != args.image_id):
        raise ProbeUnavailable("POSTDEPLOY_IMAGE_INVALID")
    with _provider_stage("FUNCTION_SPAWN_UNAVAILABLE"):
        with _deadline(60):
            call = function.spawn()
    call_id = getattr(call, "object_id", None)
    if type(call_id) is not str or _CALL.fullmatch(call_id) is None:
        raise ProbeUnavailable("CALL_ID_INDETERMINATE")
    with _provider_stage("CALL_RETENTION_UNAVAILABLE"):
        _exclusive_record(args.claim_fd, "call.json", {
            "schema_version": _SCHEMA, "call_id": call_id,
        })
    from modal._serialization import serialize

    until = time.monotonic() + 180
    while time.monotonic() < until:
        with _provider_stage("POLL_UNAVAILABLE"):
            result = synchronizer.create_blocking(_poll_raw)(
                client, call_id, api_pb2, serialize,
            )
        if result != "PENDING":
            return result
        time.sleep(min(2, max(0, until - time.monotonic())))
    return "PENDING"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-dir", type=Path, required=True)
    parser.add_argument("--name-prefix", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--modal-profile", required=True)
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args(argv)
    result: str | dict[str, str] = "LOCAL_UNAVAILABLE"
    failure_class = None
    failure_origin = None
    failure_topics = ()
    failure_action = None
    claim_fd = None
    try:
        _require_host()
        claim_fd = _private_directory(args.claim_dir)
        args.claim_fd = claim_fd
        prefix = _exact(args.name_prefix, _PREFIX)
        args.environment = _exact(args.environment, _NAME)
        args.modal_profile = _exact(args.modal_profile, _NAME)
        args.image_id = _exact(args.image_id, _IMAGE)
        nonce = secrets.token_hex(16)
        args.app = prefix + "-probe-" + nonce
        args.volume_names = tuple(prefix + "-" + role.replace("_", "-")
                                  + "-" + nonce for role in _ROLES)
        if any(_NAME.fullmatch(name) is None
               for name in (args.app, *args.volume_names)):
            raise ProbeUnavailable("INPUT_INVALID")
        claim = {
            "schema_version": _SCHEMA,
            "app": args.app,
            "environment": args.environment,
            "image_id": args.image_id,
            "volume_names": args.volume_names,
        }
        claim["selection_sha256"] = hashlib.sha256(
            json.dumps(claim, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        _exclusive_record(claim_fd, "claim.json", claim)
        with open(os.devnull, "w") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                import modal
                client = _client(modal, args.modal_profile)
                result = execute(args, modal, client)
    except ProbeUnavailable as error:
        failure_class = error.failure_class
        failure_origin = error.failure_origin
        failure_topics = error.failure_topics
        failure_action = error.failure_action
        result = str(error) if str(error) in {
            "HOST_INCOMPATIBLE", "CLAIM_PATH_INVALID", "CLAIM_INVALID",
            "CLAIM_ALREADY_CONSUMED", "CLAIM_INDETERMINATE", "INPUT_INVALID",
            "SDK_INCOMPATIBLE", "CREDENTIAL_UNAVAILABLE",
            "APP_ABSENCE_UNAVAILABLE", "APP_NOT_FRESH",
            "VOLUME_IDENTITY_INVALID", "IMAGE_READ_UNAVAILABLE", "IMAGE_IDENTITY_INVALID",
            "CALL_ID_INDETERMINATE", "PROVIDER_INDETERMINATE",
            "POSTDEPLOY_IMAGE_INVALID", *_STAGE_FAILURES,
        } else "LOCAL_UNAVAILABLE"
    except Exception:
        result = "PROVIDER_INDETERMINATE"
    finally:
        if claim_fd is not None:
            os.close(claim_fd)
    output = {
        "schema_version": _SCHEMA, "authority": "DIAGNOSTIC_ONLY",
        "result": result,
    }
    if failure_class is not None and result in _STAGE_FAILURES:
        output["failure_class"] = failure_class
    if failure_origin is not None and result in _STAGE_FAILURES:
        output["failure_origin"] = failure_origin
    if failure_topics and result == "APP_DEPLOY_UNAVAILABLE":
        output["message_topic_hints"] = failure_topics
    if failure_action is not None and result == "APP_DEPLOY_UNAVAILABLE":
        output["message_action_hint"] = failure_action
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0 if isinstance(result, dict) else 1


if __name__ == "__main__":
    sys.exit(main())
