"""Read-only inspection of one marker bound by a private Modal probe claim.

The retained owner-private claim and marker receipt bind the supplied Volume
ID, path and SHA-256 before any provider client is created. The receipt is
not cryptographically signed against same-UID tampering. This is diagnostic
only: a successful read grants no training, retry, cleanup or replay authority.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tuner.execution.providers.modal.bounded_volume_read import (
    BoundedModalVolumeReader, BoundedVolumeReadError,
)

_SCHEMA = "synaptic-modal-gpu-mount-probe/v1"
_OUTPUT_SCHEMA = "synaptic-modal-volume-marker-inspection/v1"
_ROLES = ("control", "artifacts", "model_cache")
_MARKER = re.compile(r"probe-[0-9a-f]{32}\.bin\Z")
_VOLUME = re.compile(r"vo-[A-Za-z0-9]{1,64}\Z")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE = re.compile(r"[a-z][a-z0-9-]{0,62}[a-z0-9]\Z")


class InspectionUnavailable(RuntimeError):
    """Fixed non-secret failure."""


def _pairs_unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _private_directory(path: Path) -> int:
    if os.name != "posix" or not path.is_absolute():
        raise InspectionUnavailable("CLAIM_INVALID")
    descriptor = None
    try:
        descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        parts = path.parts[1:]
        if not parts:
            raise ValueError
        for index, part in enumerate(parts):
            before = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            opened = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                               dir_fd=descriptor)
            after = os.fstat(opened)
            if (not stat.S_ISDIR(before.st_mode)
                    or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                    or before.st_uid not in (0, os.geteuid())
                    or stat.S_IMODE(before.st_mode) & 0o022
                    or (index == len(parts) - 1 and (
                        before.st_uid != os.geteuid()
                        or stat.S_IMODE(before.st_mode) & 0o077))):
                os.close(opened)
                raise ValueError
            os.close(descriptor)
            descriptor = opened
        return descriptor
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise InspectionUnavailable("CLAIM_INVALID") from None


def _record(root: int, leaf: str) -> dict:
    descriptor = None
    try:
        descriptor = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root)
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o077
                or info.st_size < 2 or info.st_size > 4096):
            raise ValueError
        raw = os.read(descriptor, 4097)
        if len(raw) != info.st_size or not raw.endswith(b"\n"):
            raise ValueError
        value = json.loads(raw, object_pairs_hook=_pairs_unique)
        if (type(value) is not dict
                or (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode() != raw):
            raise ValueError
        after = os.fstat(descriptor)
        current = os.stat(leaf, dir_fd=root, follow_symlinks=False)
        if ((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
             info.st_ctime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                 after.st_ctime_ns)
                or (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino)
                or not stat.S_ISREG(current.st_mode) or current.st_nlink != 1
                or current.st_uid != os.geteuid()
                or stat.S_IMODE(current.st_mode) & 0o077):
            raise ValueError
        return value
    except Exception:
        raise InspectionUnavailable("CLAIM_INVALID") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def authenticate_marker(claim_dir: Path, volume_id: str, path: str,
                        expected_sha256: str) -> None:
    """Authenticate one supplied tuple against the exact retained probe records."""
    root = _private_directory(claim_dir)
    try:
        claim = _record(root, "claim.json")
        receipt = _record(root, "marker-receipt.json")
        selection = claim.pop("selection_sha256", None)
        selection_raw = json.dumps(claim, sort_keys=True, separators=(",", ":")).encode()
        if (type(selection) is not str or _HEX.fullmatch(selection) is None
                or hashlib.sha256(selection_raw).hexdigest() != selection
                or set(claim) != {"schema_version", "app", "environment", "image_id",
                                  "mount_parent", "mount_paths", "inspect_links",
                                  "verify_mapping", "marker_names", "marker_sha256",
                                  "volume_names"}
                or claim["schema_version"] != _SCHEMA
                or claim["verify_mapping"] is not True
                or claim["inspect_links"] is not False
                or set(receipt) != {"schema_version", "selection_sha256",
                                    "marker_names", "marker_sha256", "volume_ids",
                                    "marker_upload"}
                or receipt["schema_version"] != _SCHEMA
                or receipt["selection_sha256"] != selection
                or receipt["marker_upload"] != "accepted_create_only_no_host_readback"
                or receipt["marker_names"] != claim["marker_names"]
                or receipt["marker_sha256"] != claim["marker_sha256"]):
            raise ValueError
        names, digests, ids = (receipt[key] for key in
                               ("marker_names", "marker_sha256", "volume_ids"))
        if (any(type(items) is not list or len(items) != len(_ROLES)
                for items in (names, digests, ids))
                or len(set(names)) != len(_ROLES)
                or len(set(ids)) != len(_ROLES)
                or any(type(item) is not str or _MARKER.fullmatch(item) is None
                       for item in names)
                or any(type(item) is not str or _HEX.fullmatch(item) is None
                       for item in digests)
                or any(type(item) is not str or _VOLUME.fullmatch(item) is None
                       for item in ids)
                or type(path) is not str or type(volume_id) is not str
                or type(expected_sha256) is not str
                or (volume_id, path, expected_sha256) not in zip(ids, names, digests)):
            raise ValueError
    except Exception:
        raise InspectionUnavailable("CLAIM_INVALID") from None
    finally:
        os.close(root)


async def inspect(reader: BoundedModalVolumeReader, *, volume_id: str,
                  path: str, digest: str) -> str:
    try:
        await asyncio.wait_for(
            reader.read_exact(volume_id=volume_id, path=path,
                              expected_size=32, expected_sha256=digest,
                              max_bytes=32),
            timeout=50,
        )
        return "MARKER_MATCH"
    except BoundedVolumeReadError:
        return "PROVIDER_READ_UNAVAILABLE"
    except Exception:
        return "PROVIDER_READ_UNAVAILABLE"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-dir", required=True, type=Path)
    parser.add_argument("--volume-id", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--modal-profile", required=True)
    args = parser.parse_args(argv)
    try:
        if _PROFILE.fullmatch(args.modal_profile) is None:
            raise InspectionUnavailable("INPUT_INVALID")
        authenticate_marker(args.claim_dir, args.volume_id, args.path, args.sha256)
        with open(os.devnull, "w") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                import modal
                from modal.config import config
                from modal._utils.async_utils import synchronizer

                if modal.__version__ != "1.5.4":
                    raise InspectionUnavailable("SDK_INVALID")
                token_id = config.get("token_id", profile=args.modal_profile, use_env=False)
                token_secret = config.get("token_secret", profile=args.modal_profile,
                                          use_env=False)
                if any(type(value) is not str or not value.strip()
                       for value in (token_id, token_secret)):
                    raise InspectionUnavailable("CREDENTIAL_UNAVAILABLE")
                client = modal.Client.from_credentials(token_id, token_secret)
                reader = BoundedModalVolumeReader(sdk=modal, client=client)
                result = synchronizer.create_blocking(inspect)(
                    reader, volume_id=args.volume_id, path=args.path,
                    digest=args.sha256,
                )
    except InspectionUnavailable as error:
        result = error.args[0]
    except Exception:
        result = "LOCAL_UNAVAILABLE"
    print(json.dumps({"schema_version": _OUTPUT_SCHEMA, "authority": "DIAGNOSTIC_ONLY",
                      "result": result}, sort_keys=True, separators=(",", ":")))
    return 0 if result == "MARKER_MATCH" else 1


if __name__ == "__main__":
    sys.exit(main())
