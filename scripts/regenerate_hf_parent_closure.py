"""Read-only check or intentional hash refresh of the fixed HF parent closure."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "tuner/execution/providers/hf_jobs_provider_v1/hf-parent-closure.json"
MEMBERS = ('synaptic_tuner/__init__.py', 'synaptic_tuner/_version.py', 'tuner/__init__.py', 'tuner/execution/__init__.py', 'tuner/execution/contracts.py', 'tuner/execution/lifecycle.py', 'tuner/execution/providers/__init__.py', 'tuner/execution/providers/contracts.py', 'tuner/execution/providers/hf_jobs_provider_v1/__init__.py', 'tuner/execution/providers/hf_jobs_provider_v1/bootstrap.py', 'tuner/execution/providers/hf_jobs_provider_v1/source_worker.py', 'tuner/execution/providers/modal/__init__.py', 'tuner/execution/providers/modal/model_snapshot.py', 'tuner/execution/providers/modal/mounted_io.py', 'tuner/execution/registry.py', 'tuner/execution/service.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    path = ROOT / MANIFEST
    original = path.read_bytes()
    document = json.loads(original)
    if (type(document) is not dict
            or set(document) != {"schema_version", "members", "member_count", "payload_bytes"}
            or document["schema_version"] != "synaptic-hf-parent-closure/v1"
            or type(document["member_count"]) is not int
            or document["member_count"] != len(MEMBERS)
            or type(document["payload_bytes"]) is not int
            or type(document["members"]) is not list
            or len(document["members"]) != len(MEMBERS)
            or any(type(row) is not dict or set(row) != {"path", "size_bytes", "sha256"}
                   or type(row["size_bytes"]) is not int or row["size_bytes"] < 0
                   or type(row["sha256"]) is not str
                   or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
                   for row in document["members"])
            or tuple(row["path"] for row in document["members"]) != MEMBERS
            or document["payload_bytes"] != sum(row["size_bytes"] for row in document["members"])):
        raise SystemExit("HF parent inventory mismatch; explicit inventory review required")
    members = []
    for name in MEMBERS:
        member = ROOT / name
        if member.is_symlink() or not member.is_file():
            raise SystemExit("HF parent member unavailable")
        raw = member.read_bytes()
        members.append({"path": name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    expected = json.dumps({"schema_version": document["schema_version"], "members": members,
        "member_count": len(members), "payload_bytes": sum(row["size_bytes"] for row in members)},
        sort_keys=True, separators=(",", ":")).encode()
    if args.write:
        path.write_bytes(expected)
    elif original != expected:
        raise SystemExit("STALE HF parent hashes; review changes before --write")
    print("CURRENT")


if __name__ == "__main__":
    main()
