from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/initialize_modal_inference_lock.py"
SPEC = importlib.util.spec_from_file_location("initialize_modal_inference_lock", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)
EVIDENCE = "docs/review/evidence/modal-inference-engine-42da029.json"
BASE = "docker.io/vllm/vllm-openai@sha256:116aa00ee0b68855616a56e1d7e1ae937e591a8bd6969ee45cbcedb246ddf355"


def test_hash_bound_inference_locks_declare_lf_git_exports():
    lines = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    for relative in (tool._ADDITIVE_RELATIVE, tool.DEPENDENCY_RELATIVE):
        assert f"{relative} text eol=lf" in lines


def _copy(root: Path, relative: str) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (ROOT / relative).read_bytes()
    # Git's Windows worktree conversion is irrelevant to the accepted Linux
    # image bytes. Only this fixture restores the reviewed LF lock payload.
    if relative == tool._ADDITIVE_RELATIVE:
        payload = payload.replace(b"\r\n", b"\n")
    target.write_bytes(payload)


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "repository"
    for relative in (*tool.SOURCE_MEMBERS, EVIDENCE, tool._ADDITIVE_RELATIVE):
        _copy(root, relative)
    evidence = (root / EVIDENCE).read_bytes()
    return root, {
        "accepted_evidence": EVIDENCE,
        "accepted_evidence_sha256": hashlib.sha256(evidence).hexdigest(),
        "base_registry_reference": BASE,
        "additive_lock_sha256": hashlib.sha256(
            (root / tool._ADDITIVE_RELATIVE).read_bytes()
        ).hexdigest(),
    }


def test_proposal_is_deterministic_and_read_only(tmp_path):
    root, values = _fixture(tmp_path)
    first = tool.proposal(root, **values)
    second = tool.proposal(root, **values)

    assert first == second
    assert set(first) == {
        tool.DEPENDENCY_RELATIVE,
        tool.CLOSURE_RELATIVE,
        tool.RUNTIME_RELATIVE,
    }
    assert all(not (root / relative).exists() for relative in first)
    closure = json.loads(first[tool.CLOSURE_RELATIVE])
    runtime = json.loads(first[tool.RUNTIME_RELATIVE])
    dependency = json.loads(first[tool.DEPENDENCY_RELATIVE])
    assert closure["member_count"] == len(tool.SOURCE_MEMBERS) == 118
    assert [item["path"] for item in closure["members"]] == list(tool.SOURCE_MEMBERS)
    assert len(runtime["source_inventory"]) == 120
    assert dependency["distributions"] == runtime["distributions"]
    assert dependency["base_registry_reference"] == runtime["base_registry_reference"]
    assert dependency["provenance"]["status"] == "ACCEPTED_AS_STARTING_PINS_ONLY"


def test_write_exclusively_creates_resources_accepted_by_maintainer(tmp_path):
    root, values = _fixture(tmp_path)
    proposed = tool.initialize(root, **values)

    for relative, payload in proposed.items():
        assert (root / relative).read_bytes() == payload
    before = {relative: (root / relative).read_bytes() for relative in proposed}
    assert tool._maintenance.refresh(root)[0] == before[tool.RUNTIME_RELATIVE]
    assert tool._maintenance.refresh(root)[1] == before[tool.CLOSURE_RELATIVE]
    with pytest.raises(tool.InitializationFault, match="TARGET_EXISTS"):
        tool.initialize(root, **values)
    with pytest.raises(tool.InitializationFault, match="TARGET_EXISTS"):
        tool.proposal(root, **values)
    assert {relative: (root / relative).read_bytes() for relative in proposed} == before


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ("evidence_digest", "ACCEPTED_EVIDENCE_MISMATCH"),
        ("base", "BASE_IMAGE_MISMATCH"),
        ("additions", "ADDITIVE_LOCK_MISMATCH"),
        ("candidate_status", "EVIDENCE_STATUS_INVALID"),
        ("distribution", "EVIDENCE_CANDIDATE_INVALID"),
    ],
)
def test_candidate_and_additive_inputs_are_cross_bound(tmp_path, change, code):
    root, values = _fixture(tmp_path)
    if change == "evidence_digest":
        values["accepted_evidence_sha256"] = "0" * 64
    elif change == "base":
        values["base_registry_reference"] = "docker.io/vllm/other@sha256:" + "1" * 64
    elif change == "additions":
        path = root / tool._ADDITIVE_RELATIVE
        path.write_bytes(path.read_bytes() + b"# changed\n")
    else:
        path = root / EVIDENCE
        value = json.loads(path.read_bytes())
        if change == "candidate_status":
            value["candidate"]["status"] = "QUALIFIED"
        else:
            value["candidate"]["distributions"]["Modal"] = value["candidate"][
                "distributions"
            ].pop("modal")
        path.write_bytes(tool._canonical(value))
        values["accepted_evidence_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    with pytest.raises(tool.InitializationFault, match=code):
        tool.proposal(root, **values)
    assert not any(
        (root / relative).exists()
        for relative in (
            tool.DEPENDENCY_RELATIVE,
            tool.CLOSURE_RELATIVE,
            tool.RUNTIME_RELATIVE,
        )
    )


@pytest.mark.parametrize("kind", ["file", "symlink", "broken_symlink"])
def test_any_existing_target_fails_before_writes(tmp_path, kind):
    root, values = _fixture(tmp_path)
    target = root / tool.CLOSURE_RELATIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "file":
        target.write_bytes(b"occupied")
    else:
        target.symlink_to(root / ("present" if kind == "symlink" else "absent"))
        if kind == "symlink":
            (root / "present").write_bytes(b"occupied")

    with pytest.raises(tool.InitializationFault, match="TARGET_EXISTS"):
        tool.initialize(root, **values)
    assert not (root / tool.DEPENDENCY_RELATIVE).exists()
    assert not (root / tool.RUNTIME_RELATIVE).exists()


def test_source_change_during_recheck_fails_before_writes(tmp_path, monkeypatch):
    root, values = _fixture(tmp_path)
    original = tool.proposal
    calls = 0

    def changing(repo, **arguments):
        nonlocal calls
        result = original(repo, **arguments)
        calls += 1
        if calls == 1:
            member = repo / tool.SOURCE_MEMBERS[0]
            member.write_bytes(member.read_bytes() + b"# changed\n")
        return result

    monkeypatch.setattr(tool, "proposal", changing)
    with pytest.raises(tool.InitializationFault, match="INPUT_CHANGED_BEFORE_WRITE"):
        tool.initialize(root, **values)
    assert not any(
        (root / relative).exists()
        for relative in (
            tool.DEPENDENCY_RELATIVE,
            tool.CLOSURE_RELATIVE,
            tool.RUNTIME_RELATIVE,
        )
    )


def test_cli_default_only_reports_proposal(tmp_path, capsys):
    root, values = _fixture(tmp_path)
    argv = [
        "--accepted-evidence",
        values["accepted_evidence"],
        "--accepted-evidence-sha256",
        values["accepted_evidence_sha256"],
        "--base-registry-reference",
        values["base_registry_reference"],
        "--additive-lock-sha256",
        values["additive_lock_sha256"],
    ]
    assert tool.main(argv, root=root) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PROPOSED"
    assert report["member_count"] == 118
    assert all(not (root / relative).exists() for relative in report["resource_sha256"])


def test_initializer_does_not_modify_hash_only_maintainer_contract():
    assert tool.SOURCE_MEMBERS is tool._maintenance.SOURCE_MEMBERS
    assert len(tool.SOURCE_MEMBERS) == 118
    assert not hasattr(tool._maintenance, "initialize")


@pytest.mark.parametrize(
    "mutation",
    ["python_preparation", "sandbox", "digest_type", "source_control"],
)
def test_strict_envelope_provenance_is_required(tmp_path, mutation):
    root, values = _fixture(tmp_path)
    path = root / EVIDENCE
    document = json.loads(path.read_bytes())
    if mutation == "python_preparation":
        document["python_preparation"]["executable"] = "/usr/bin/python"
    elif mutation == "sandbox":
        document["sandbox_id"] = "not-a-sandbox"
    elif mutation == "digest_type":
        document["engine_wheel_sha256"] = 111
    else:
        document["candidate"]["operator_selection"]["source_commit"] = "a\n" + "0" * 38
    path.write_bytes(tool._canonical(document))
    values["accepted_evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(tool.InitializationFault):
        tool.proposal(root, **values)


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_name"])
def test_additive_lock_requires_exact_nine_unique_names(tmp_path, mutation):
    root, values = _fixture(tmp_path)
    path = root / tool._ADDITIVE_RELATIVE
    lines = path.read_text().splitlines()
    indexes = [
        index for index, line in enumerate(lines) if line and not line.startswith("#")
    ]
    if mutation == "duplicate":
        lines[indexes[-1]] = lines[indexes[0]]
    else:
        lines[indexes[-1]] = lines[indexes[-1]].replace("types-toml", "requests", 1)
    path.write_bytes(("\n".join(lines) + "\n").encode())
    values["additive_lock_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = json.loads((root / EVIDENCE).read_bytes())
    evidence["modal_additions_sha256"] = values["additive_lock_sha256"]
    (root / EVIDENCE).write_bytes(tool._canonical(evidence))
    values["accepted_evidence_sha256"] = hashlib.sha256(
        (root / EVIDENCE).read_bytes()
    ).hexdigest()

    with pytest.raises(tool.InitializationFault, match="ADDITIVE_LOCK_INVALID"):
        tool.proposal(root, **values)


def test_generated_manifest_uses_production_parser(tmp_path, monkeypatch):
    root, values = _fixture(tmp_path)
    from tuner.execution.providers.modal import inference_runtime

    called = False
    original = inference_runtime._manifest

    def observed():
        nonlocal called
        called = True
        return original()

    monkeypatch.setattr(inference_runtime, "_manifest", observed)
    tool.proposal(root, **values)
    assert called
