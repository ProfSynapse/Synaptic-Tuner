"""Pin the reviewed additive SDK subset without replacing the vLLM image."""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
ADDITIONS = ROOT / "requirements/modal-inference-additions.lock"
LAUNCHER = ROOT / "requirements/modal-launcher-v1.lock"
MEASURED_INVENTORY = "docs/review/evidence/modal-inference-inventory-eefc6a6.json"
EXPECTED = {
    "grpclib": "0.4.9",
    "h2": "4.4.1",
    "hpack": "4.2.0",
    "hyperframe": "6.1.0",
    "modal": "1.5.4",
    "synchronicity": "0.12.5",
    "toml": "0.10.2",
    "types-certifi": "2021.10.8.3",
    "types-toml": "0.10.8.20260518",
}
ALREADY_SATISFIED = {
    "aiohttp",
    "cbor2",
    "certifi",
    "click",
    "protobuf",
    "rich",
    "typing-extensions",
    "watchfiles",
}
REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9_-]+)==(?P<version>[^ ]+) "
    r"--hash=sha256:(?P<hash>[0-9a-f]{64})$"
)


def _requirements(path: Path) -> dict[str, tuple[str, str, str]]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        matched = REQUIREMENT.fullmatch(line)
        assert matched is not None, f"noncanonical locked requirement: {line}"
        name = matched["name"].lower().replace("_", "-")
        assert name not in values
        values[name] = (matched["version"], matched["hash"], line)
    return values


def test_additions_are_the_exact_reviewed_nine_package_inventory():
    additions = _requirements(ADDITIONS)
    assert {name: value[0] for name, value in additions.items()} == EXPECTED
    assert len(additions) == 9
    assert list(additions) == sorted(additions)
    assert additions.keys().isdisjoint(ALREADY_SATISFIED)


def test_every_addition_preserves_the_launcher_lock_version_and_hash():
    additions = _requirements(ADDITIONS)
    launcher = _requirements(LAUNCHER)
    assert {name: launcher[name] for name in additions} == additions


def test_lock_declares_additive_measured_image_scope_not_complete_runtime():
    text = ADDITIONS.read_text(encoding="utf-8")
    assert MEASURED_INVENTORY in text
    assert "--no-deps --require-hashes" in text
    assert "not a complete inference runtime lock" in text
    assert "must not replace its ML stack" in text
