"""Dependency-free CPU fixture: writes only below the explicit artifact root."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def main(arguments: list[str]) -> int:
    if len(arguments) != 3:
        return 2
    source = Path(arguments[1])
    artifact_root = Path(arguments[2])
    raw = (source / "input.json").read_bytes()
    value = json.loads(raw)
    if value.get("schema_version") != "synaptic-docker-cpu-fixture-input/v1":
        return 3
    artifact_root.mkdir(parents=True, exist_ok=True)
    output = json.dumps(
        {"input_sha256": hashlib.sha256(raw).hexdigest(), "message": value["message"]},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    (artifact_root / "result.json").write_bytes(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
