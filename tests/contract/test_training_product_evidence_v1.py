from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "training_product" / "modal_live_v1"
EXPECTED_ROLES = {
    "final_model",
    "tokenizer",
    "training_lineage",
    "training_metrics",
    "workload_record",
}


def _json(relative: str) -> dict[str, object]:
    return json.loads((FIXTURE / relative).read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_modal_live_evidence_index_is_closed_and_identity_locked():
    index = _json("evidence-index.json")
    assert index["schema_version"] == "synaptic-training-product-evidence-index/v1"
    assert index["closed"] is True
    identities = index["identities"]
    assert identities == {
        "engine_commit": "31d2683448919e1e694f36392fa4e40741226ae9",
        "host_commit": "a4926a274b847e1a746ad4de53563b5976fc1574",
        "project_ref": "epistemic-humility-research",
        "provider": "modal",
        "provider_job_ref": "fc-01M0Z8K9MCPN3P368V3CK94TV2",
        "run_id": "modal-sft-20260826T144636Z-7aec224e893d",
    }
    observed = {
        path.relative_to(FIXTURE).as_posix()
        for path in FIXTURE.rglob("*")
        if path.is_file()
    }
    assert set(index["fixture_members"]) == observed


def test_fixture_makes_missing_provider_terminal_bytes_machine_checkable():
    completeness = _json("evidence-index.json")["completeness"]
    for name in (
        "authenticated_provider_completion_manifest_bytes",
        "authenticated_provider_terminal_record_bytes",
    ):
        assert completeness[name] == {
            "captured": False,
            "reason_code": "not_present_in_portable_local_host_evidence",
            "state": "absent",
        }
    assert completeness["artifact_payload_bytes"] == {
        "captured": False,
        "state": "absent_descriptors_only",
    }
    assert completeness["synthetic_substitutes"]["evidence"] is False


def test_sanitized_lifecycle_projection_preserves_recovery_chronology():
    lifecycle = _json("lifecycle-projection.json")
    assert lifecycle["revision"] == len(lifecycle["events"]) == 12
    assert lifecycle["phase"] == "succeeded"
    assert lifecycle["verification"] == "verified"
    assert [event["code"] for event in lifecycle["events"]][-3:] == [
        "verification_invalid",
        "verification_reopened",
        "verification_verified",
    ]
    assert lifecycle["effects"] == [
        {
            "attempt_count": 1,
            "kind": "submit",
            "provider_job_ref": "fc-01M0Z8K9MCPN3P368V3CK94TV2",
            "state": "found",
        }
    ]


def test_publication_projection_and_substitutes_are_exact_but_distinct():
    index = _json("evidence-index.json")
    publication = _json("publication-projection.json")
    projected = {item["kind"]: item for item in publication["artifacts"]}
    assert set(projected) == EXPECTED_ROLES
    assert {item["kind"] for item in index["observed_artifacts"]} == EXPECTED_ROLES
    for item in index["observed_artifacts"]:
        descriptor = projected[item["kind"]]
        assert descriptor["sha256"] == item["observed_sha256"]
        assert descriptor["size"] == item["observed_size"]
        substitute = FIXTURE / item["substitute_path"]
        assert substitute.stat().st_size == item["substitute_size"] < 1024
        assert _digest(substitute) == item["substitute_sha256"]
        assert item["substitute_sha256"] != item["observed_sha256"]
        body = json.loads(substitute.read_text(encoding="utf-8"))
        assert body["artifact_kind"] == item["kind"]
        assert body["observed_payload_included"] is False
        assert body["status"] == "synthetic_placeholder"


def test_projected_evidence_hashes_are_locked():
    for item in _json("evidence-index.json")["projections"]:
        path = FIXTURE / item["path"]
        assert path.stat().st_size == item["size"]
        assert _digest(path) == item["sha256"]


def test_fixture_contains_no_secret_shaped_values_or_private_paths():
    rendered = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(FIXTURE.rglob("*"))
        if path.is_file()
    )
    forbidden = (
        r"(?i)bearer\s+[a-z0-9._-]+",
        r"(?i)(?:hf|runpod|modal)_[a-z0-9]{16,}",
        r"-----BEGIN [A-Z ]+PRIVATE KEY-----",
        r"(?i)[a-z]:\\",
        r"/(?:home|users)/[^/]+/",
        r"(?i)(?:file|local)://",
    )
    assert all(re.search(pattern, rendered) is None for pattern in forbidden)


def test_authoritative_architecture_doc_uses_closed_evidence_statuses():
    document = (ROOT / "docs" / "architecture" / "submodule-first-training-v1.md").read_text(
        encoding="utf-8"
    )
    assert "`LIVE_PROVEN`" in document
    assert "`IMPLEMENTED_FAKE_TESTED`" in document
    assert "`CONTRACT_ONLY`" in document
    assert "`NOT_IMPLEMENTED`" in document
    assert "No paid cloud rerun is part of this gate." in document
    roadmap = (
        ROOT / "docs" / "plans" / "submodule-first-training-product-roadmap-plan.md"
    ).read_text(encoding="utf-8")
    assert "> Status: APPROVED" in roadmap
    assert "PENDING APPROVAL" not in roadmap
