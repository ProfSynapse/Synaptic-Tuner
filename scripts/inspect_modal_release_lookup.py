"""Private, read-only diagnosis of one retained Modal deployment lookup.

This reports current provider shape, not deployment authority or retry permission.
It does not open the host journal through its write-capable constructor.
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
import sqlite3
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tuner.execution.foundation_v2.canonical import (
    canonical_bytes, parse_canonical_object, safe_ref,
)

_NAMESPACE = "standalone-training"
_CLAIM_PREFIX = "deploy-"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MAX_CLAIM = 16 * 1024


class DiagnosticUnavailable(RuntimeError):
    """A closed diagnostic failure; exception details must not be printed."""


def _private_regular(info: os.stat_result) -> bool:
    return (stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            and info.st_uid == os.geteuid()
            and not stat.S_IMODE(info.st_mode) & 0o077)


def read_deploy_claim(database: Path, claim_ref: str) -> str:
    """Read one exact journal row, without creating files or SQLite schema."""
    descriptor = None
    try:
        if (os.name != "posix" or not database.is_absolute()
                or type(claim_ref) is not str
                or not claim_ref.startswith(_CLAIM_PREFIX)
                or _HEX.fullmatch(claim_ref[len(_CLAIM_PREFIX):]) is None):
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
        # The descriptor pins the checked inode; immutable+ro never creates a
        # journal, lock, schema, or sidecar. This journal uses DELETE mode.
        uri = f"file:/proc/self/fd/{descriptor}?mode=ro&immutable=1"
        with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=2.0)) as connection:
            connection.execute("PRAGMA query_only=ON")
            rows = connection.execute(
                "SELECT digest,evidence FROM attempts "
                "WHERE namespace_ref=? AND attempt_ref=? AND length(evidence)<=?",
                (_NAMESPACE, claim_ref, _MAX_CLAIM),
            ).fetchmany(2)
        if len(rows) != 1:
            raise ValueError
        row = rows[0]
        if type(row[0]) is not str or type(row[1]) is not bytes:
            raise ValueError
        raw = row[1]
        if (not raw or len(raw) > _MAX_CLAIM
                or hashlib.sha256(raw).hexdigest() != row[0]):
            raise ValueError
        claim = parse_canonical_object(raw, name="deploy claim")
        if canonical_bytes(claim) != raw or set(claim) != {
            "schema_version", "release_digest", "capture_digest",
            "quote_digest", "deployment_name",
        }:
            raise ValueError
        if (claim["schema_version"] != "synaptic-modal-host-deploy-claim/v1"
                or claim["release_digest"] != claim_ref[len(_CLAIM_PREFIX):]
                or any(type(claim[key]) is not str or _HEX.fullmatch(claim[key]) is None
                       for key in ("release_digest", "capture_digest", "quote_digest"))):
            raise ValueError
        name = safe_ref(claim["deployment_name"], "deployment_name")
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", name):
            raise ValueError
        if os.fstat(descriptor).st_ino != info.st_ino:
            raise ValueError
        return name
    except Exception:
        raise DiagnosticUnavailable("journal_invalid") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


async def inspect_lookup(client: object, name: str, environment: str,
                         api_pb2: object) -> dict[str, object]:
    """Perform exactly one bounded read and classify only a closed shape."""
    request = api_pb2.AppGetByDeploymentNameRequest(
        name=name, environment_name=environment,
    )
    try:
        response = await asyncio.wait_for(
            client.stub.AppGetByDeploymentName(request), timeout=15,
        )
    except asyncio.TimeoutError:
        return {"result": "TIMEOUT"}
    except Exception:
        return {"result": "TRANSPORT_ERROR"}
    try:
        echoed = response.environment_name
        if echoed not in ("", environment):
            return {"result": "ENVIRONMENT_MISMATCH"}
        current, previous = response.app_id, response.previous_app_id
        state, version = response.lifecycle.app_state, response.lifecycle.version
        if type(current) is not str or type(previous) is not str or type(version) is not int:
            raise ValueError
        if current == "" and previous == "" and response.lifecycle == api_pb2.AppLifecycle():
            return {"result": "ABSENT", "environment_echo": "EMPTY" if not echoed else "MATCH"}
        if not echoed:
            return {"result": "ENVIRONMENT_UNCONFIRMED"}
        if (state == api_pb2.APP_STATE_DEPLOYED and current and version >= 1):
            category = "DEPLOYED"
        elif (state == api_pb2.APP_STATE_STOPPED and not current
              and previous and version >= 1):
            category = "STOPPED"
        else:
            raise ValueError
        return {
            "result": category,
            "environment_echo": "EMPTY" if not echoed else "MATCH",
            "has_current_id": bool(current),
            "has_previous_id": bool(previous),
        }
    except Exception:
        return {"result": "INVALID_RESPONSE"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--claim-ref", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--modal-profile", required=True)
    parser.add_argument("--production-reader", action="store_true",
                        help="Use the exact runtime reader instead of raw response classification")
    args = parser.parse_args(argv)
    result: dict[str, object]
    try:
        name = read_deploy_claim(args.journal, args.claim_ref)
        environment = safe_ref(args.environment, "environment")
        profile = safe_ref(args.modal_profile, "modal_profile")
        with open(os.devnull, "w") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                import modal
                from modal.config import config

                if modal.__version__ != "1.5.4":
                    raise DiagnosticUnavailable("sdk_invalid")
                token_id = config.get("token_id", profile=profile, use_env=False)
                token_secret = config.get("token_secret", profile=profile, use_env=False)
                if any(type(value) is not str or not value.strip()
                       for value in (token_id, token_secret)):
                    raise DiagnosticUnavailable("credential_unavailable")
                from modal._utils.async_utils import synchronizer
                from modal_proto import api_pb2

                client = modal.Client.from_credentials(token_id, token_secret)
                if args.production_reader:
                    from tuner.execution.providers.modal.runtime_release_deployment import (
                        ExplicitModal154ReleaseDeploymentReader,
                    )

                    try:
                        observation = ExplicitModal154ReleaseDeploymentReader(
                            sdk=modal,
                        ).observe(
                            client=client, app_name=name,
                            environment_name=environment,
                        )
                        result = {"result": "READER_ABSENT" if observation is None
                                  else "READER_PRESENT"}
                    except Exception:
                        result = {"result": "READER_UNAVAILABLE"}
                else:
                    result = synchronizer.create_blocking(inspect_lookup)(
                        client, name, environment, api_pb2,
                    )
    except DiagnosticUnavailable as error:
        result = {"result": error.args[0].upper()}
    except Exception:
        result = {"result": "LOCAL_ERROR"}
    result = {
        "schema_version": "synaptic-modal-release-lookup-diagnostic/v1",
        "authority": "DIAGNOSTIC_ONLY",
        "environment_selection": "OPERATOR_SUPPLIED_UNAUTHENTICATED",
        **result,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["result"] in {
        "ABSENT", "DEPLOYED", "STOPPED", "READER_ABSENT", "READER_PRESENT",
    } else 1


if __name__ == "__main__":
    sys.exit(main())
