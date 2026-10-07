from __future__ import annotations

import copy
import base64
import hashlib
import json
import os
import shutil
import stat
import subprocess
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import quote

import pytest
from jsonschema import Draft202012Validator

from tuner.cloud.hf_provisioning import (
    EVIDENCE_SCHEMA_VERSION,
    HFConsumableSourceTransport,
    canonical_json_bytes,
    consume_hf_source_transport,
    load_canonical_json,
    load_hf_source_transport,
    prepare_hf_source_transport,
    validate_hf_evidence_binding,
    validate_hf_bootstrap_volume_config,
    validate_hf_provisioning_evidence,
    validate_hf_source_transport_descriptor,
)
from tuner.cloud.checkout import CheckoutPolicy, SSHCheckoutPolicy
from tuner.cloud.hf_jobs import CloudJobSpec, HFJobExecutor
from tuner.cloud.hf_volume_transport import HFVerifiedVolume, HFVerifiedVolumeSpec
from tuner.core.exceptions import CloudProviderError
from tuner.project import ProjectContext
from tuner.project.source_bundle import GitSource, RepositoryLocation, SourceLock
from tuner.handlers.stages._util import preflight_hf_source_lock, prepare_hf_source


REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixture(tmp_path: Path, *, run_id: str = "run-test") -> tuple[ProjectContext, SourceLock]:
    engine = tmp_path / "engine"
    modules = engine / "tuner" / "cloud"
    modules.mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / "tuner/cloud/bootstrap_core.py", modules / "bootstrap_core.py")
    shutil.copyfile(REPO_ROOT / "tuner/cloud/bootstrap_capsule.py", modules / "bootstrap_capsule.py")
    (engine / "config.yaml").write_text("model: test\n", encoding="utf-8")
    _git(engine, "init", "-b", "main")
    _git(engine, "config", "user.name", "Synaptic Test")
    _git(engine, "config", "user.email", "test@example.invalid")
    _git(engine, "add", ".")
    _git(engine, "commit", "-m", "fixture")
    commit = _git(engine, "rev-parse", "HEAD")
    source = GitSource(
        location=RepositoryLocation.parse("https://git.example.test/team/engine.git"),
        commit=commit,
        branch="main",
        pushed=True,
    )
    lock = SourceLock(
        run_id=run_id,
        mode="standalone",
        project_source=source,
        engine_source=source,
        project={
            "manifest_uri": "engine://config.yaml",
            "manifest_sha256": hashlib.sha256(b"model: test\n").hexdigest(),
            "engine_requires": "*",
        },
        configuration={
            "resolved_uri": "engine://config.yaml",
            "resolved_sha256": hashlib.sha256(b"model: test\n").hexdigest(),
            "documents": [
                {
                    "uri": "engine://config.yaml",
                    "sha256": hashlib.sha256(b"model: test\n").hexdigest(),
                }
            ],
        },
    )
    return ProjectContext.standalone(engine_root=engine), lock


def _prepare(
    tmp_path: Path,
    *,
    name: str = "transport",
    run_id: str = "run-test",
    checkout_policy: CheckoutPolicy | None = None,
):
    context, lock = _fixture(tmp_path, run_id=run_id)
    source_lock_uri = f"tracking://experiments/{run_id}/source-lock.json"
    descriptor_uri = f"tracking://experiments/{run_id}/cloud/hf/source-transport/descriptor.json"
    prepared = prepare_hf_source_transport(
        context,
        source_lock=lock,
        source_lock_uri=source_lock_uri,
        descriptor_uri=descriptor_uri,
        transport_root=(tmp_path / name).resolve(),
        volume_source="owner/bootstrap-bucket",
        path_prefix="prepared/sources",
        checkout_policy=checkout_policy,
    )
    return context, prepared, source_lock_uri, descriptor_uri


def _evidence(prepared) -> dict[str, object]:
    descriptor = prepared.descriptor
    volume = descriptor["volume"]
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "descriptor": {
            "uri": prepared.descriptor_uri,
            "sha256": prepared.descriptor_sha256,
        },
        "run_id": descriptor["run_id"],
        "provider": "hf_jobs",
        "profile": "C",
        "volume": {
            "source": volume["source"],
            "path": volume["path"],
            "type": "bucket",
            "read_only": True,
        },
        "bundle_sha256": descriptor["bundle"]["content_sha256"],
        "capsule_manifest_sha256": descriptor["capsule"]["manifest"]["sha256"],
        "source_lock_sha256": descriptor["source_lock"]["sha256"],
        "checkout_policy_sha256": descriptor["checkout_policy"]["sha256"],
        "status": "provisioned",
        "authority": "protected_workflow",
        "actor": "workflow-42",
        "asserted_at": "2026-08-19T12:00:00Z",
        "provider_receipt_id": "receipt-42",
    }


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _make_writable(path: Path) -> None:
    os.chmod(path, path.lstat().st_mode | stat.S_IWRITE)


def test_closed_schemas_are_valid_and_reject_extensions(tmp_path: Path) -> None:
    context, prepared, source_lock_uri, descriptor_uri = _prepare(tmp_path)
    for name in (
        "synaptic-hf-source-transport-v1.schema.json",
        "synaptic-hf-provisioning-evidence-v1.schema.json",
    ):
        schema = json.loads((REPO_ROOT / "schemas" / name).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)

    descriptor = copy.deepcopy(prepared.descriptor)
    descriptor["volume"]["extension"] = True
    with pytest.raises(CloudProviderError, match="invalid"):
        validate_hf_source_transport_descriptor(descriptor)

    evidence = _evidence(prepared)
    evidence["provider_response"] = {"secret": "forbidden"}
    with pytest.raises(CloudProviderError, match="invalid"):
        validate_hf_provisioning_evidence(evidence)


def test_preparation_is_deterministic_atomic_and_idempotent(tmp_path: Path) -> None:
    context, prepared, source_lock_uri, descriptor_uri = _prepare(tmp_path)
    first = _tree(prepared.root)
    again = prepare_hf_source_transport(
        context,
        source_lock=prepared.source_lock,
        source_lock_uri=source_lock_uri,
        descriptor_uri=descriptor_uri,
        transport_root=prepared.root,
        volume_source="owner/bootstrap-bucket",
        path_prefix="prepared/sources",
    )
    assert _tree(again.root) == first
    assert again.descriptor_sha256 == prepared.descriptor_sha256
    assert again.descriptor["source_lock"]["uri"] == source_lock_uri
    assert again.descriptor["volume"]["path"] == (
        f"prepared/sources/run-test/{again.descriptor['bundle']['content_sha256']}"
    )
    assert not list(prepared.root.parent.glob(f".{prepared.root.name}.prepare-*"))

    with pytest.raises(CloudProviderError, match="conflicts"):
        prepare_hf_source_transport(
            context,
            source_lock=prepared.source_lock,
            source_lock_uri="tracking://standalone/other/source-lock.json",
            descriptor_uri=descriptor_uri,
            transport_root=prepared.root,
            volume_source="owner/bootstrap-bucket",
            path_prefix="prepared/sources",
        )


def test_concurrent_identical_preparation_publishes_one_complete_tree(tmp_path: Path) -> None:
    context, lock = _fixture(tmp_path)
    root = (tmp_path / "transport").resolve()
    arguments = {
        "source_lock": lock,
        "source_lock_uri": "tracking://experiments/run-test/source-lock.json",
        "descriptor_uri": "tracking://experiments/run-test/cloud/hf/source-transport/descriptor.json",
        "transport_root": root,
        "volume_source": "owner/bootstrap-bucket",
        "path_prefix": "prepared/sources",
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: prepare_hf_source_transport(context, **arguments), range(2)))
    assert results[0].descriptor_sha256 == results[1].descriptor_sha256
    assert load_canonical_json(root / "descriptor.json") == results[0].descriptor
    assert not list(root.parent.glob(f".{root.name}.prepare-*"))


@pytest.mark.parametrize(
    "extra",
    [
        {"local_root": "C:/mutable"},
        {"path": "static/path"},
        {"mount_path": "/mutable"},
    ],
)
def test_preparation_api_has_no_mutable_config_escape_hatches(tmp_path: Path, extra: dict[str, str]) -> None:
    context, lock = _fixture(tmp_path)
    arguments = {
        "source_lock": lock,
        "source_lock_uri": "tracking://standalone/run-test/source-lock.json",
        "descriptor_uri": "tracking://standalone/run-test/source-transport/descriptor.json",
        "transport_root": (tmp_path / "transport").resolve(),
        "volume_source": "owner/bootstrap-bucket",
        "path_prefix": "prepared/sources",
        **extra,
    }
    with pytest.raises(TypeError):
        prepare_hf_source_transport(context, **arguments)


def test_cloud_config_exposes_only_nested_source_and_path_prefix() -> None:
    import yaml

    config = yaml.safe_load(
        (REPO_ROOT / "Trainers/cloud/cloud_config.yaml").read_text(encoding="utf-8")
    )
    volume = config["cloud"]["hf_jobs"]["bootstrap_volume"]
    assert set(volume) == {"source", "path_prefix"}
    assert validate_hf_bootstrap_volume_config(volume) == (
        "professorsynapse/toolset-training-bootstrap",
        "synaptic/source-transport",
    )
    for forbidden in ("local_root", "path", "mount_path"):
        with pytest.raises(CloudProviderError, match="exactly"):
            validate_hf_bootstrap_volume_config({**volume, forbidden: "forbidden"})


def test_named_source_preflight_builds_and_validates_before_prepared(tmp_path: Path) -> None:
    context, lock = _fixture(tmp_path)
    with patch("tuner.handlers.stages._util.build_source_lock", return_value=lock) as build:
        with patch("tuner.handlers.stages._util.validate_source_lock_for_cloud") as validate:
            result = preflight_hf_source_lock(context, run_id=lock.run_id)
    assert result is lock
    build.assert_called_once()
    validate.assert_called_once_with(lock)


def test_prepared_requires_explicit_preflighted_source_lock(tmp_path: Path) -> None:
    context, _lock = _fixture(tmp_path)
    with pytest.raises(TypeError, match="source_lock"):
        prepare_hf_source(
            context,
            run_id="run-test",
            config_path=context.engine_root / "config.yaml",
            volume_settings={"source": "owner/bootstrap-bucket", "path_prefix": "prepared/sources"},
            transport_root=(tmp_path / "transport").resolve(),
        )


def test_consumption_rebuilds_capsule_and_policy_then_returns_exact_volume(tmp_path: Path) -> None:
    context, prepared, source_lock_uri, descriptor_uri = _prepare(tmp_path)
    result = consume_hf_source_transport(
        context,
        transport_root=prepared.root,
        descriptor_uri=descriptor_uri,
        source_lock_uri=source_lock_uri,
        evidence=_evidence(prepared),
    )
    assert isinstance(result, HFConsumableSourceTransport)
    assert result.volume_spec.source == "owner/bootstrap-bucket"
    assert result.volume_spec.path == prepared.descriptor["volume"]["path"]
    assert result.volume_spec.local_root == prepared.bundle_root.resolve()
    assert result.evidence_sha256 == hashlib.sha256(canonical_json_bytes(result.evidence)).hexdigest()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "other-run"),
        ("bundle_sha256", "0" * 64),
        ("capsule_manifest_sha256", "0" * 64),
        ("source_lock_sha256", "0" * 64),
        ("checkout_policy_sha256", "0" * 64),
    ],
)
def test_wrong_run_replay_and_digest_tampering_fail_closed(
    tmp_path: Path, field: str, value: str,
) -> None:
    context, prepared, source_lock_uri, descriptor_uri = _prepare(tmp_path)
    evidence = _evidence(prepared)
    evidence[field] = value
    with pytest.raises(CloudProviderError, match="does not match descriptor binding"):
        consume_hf_source_transport(
            context,
            transport_root=prepared.root,
            descriptor_uri=descriptor_uri,
            source_lock_uri=source_lock_uri,
            evidence=evidence,
        )


def test_writable_or_changed_volume_evidence_fails_closed(tmp_path: Path) -> None:
    _context, prepared, _source_lock_uri, _descriptor_uri = _prepare(tmp_path)
    evidence = _evidence(prepared)
    evidence["volume"]["read_only"] = False
    with pytest.raises(CloudProviderError, match="invalid"):
        validate_hf_evidence_binding(
            prepared.descriptor,
            evidence,
            descriptor_uri=prepared.descriptor_uri,
        )
    evidence = _evidence(prepared)
    evidence["volume"]["path"] = "prepared/sources/other/" + "0" * 64
    with pytest.raises(CloudProviderError, match="does not match"):
        validate_hf_evidence_binding(
            prepared.descriptor,
            evidence,
            descriptor_uri=prepared.descriptor_uri,
        )


def test_consumption_rejects_mutated_capsule_and_policy(tmp_path: Path) -> None:
    context, prepared, source_lock_uri, descriptor_uri = _prepare(tmp_path)
    capsule_member = prepared.bundle_root / "capsule/tuner/cloud/bootstrap_core.py"
    _make_writable(capsule_member)
    capsule_member.write_bytes(capsule_member.read_bytes() + b"\n# changed\n")
    with pytest.raises(CloudProviderError, match="conflicts|content digest"):
        load_hf_source_transport(
            context,
            transport_root=prepared.root,
            descriptor_uri=descriptor_uri,
            source_lock_uri=source_lock_uri,
        )

    other = tmp_path / "other"
    context, prepared, source_lock_uri, descriptor_uri = _prepare(other)
    policy = prepared.bundle_root / "checkout-policy.json"
    _make_writable(policy)
    policy.write_bytes(canonical_json_bytes({"allowed_hosts": ["evil.example"]}))
    with pytest.raises(CloudProviderError, match="checkout policy"):
        load_hf_source_transport(
            context,
            transport_root=prepared.root,
            descriptor_uri=descriptor_uri,
            source_lock_uri=source_lock_uri,
        )


def test_load_rejects_duplicate_noncanonical_and_extension_bundle_members(tmp_path: Path) -> None:
    _context, prepared, _source_lock_uri, _descriptor_uri = _prepare(tmp_path)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"schema_version":"x","schema_version":"y"}\n')
    with pytest.raises(CloudProviderError, match="duplicate"):
        load_canonical_json(duplicate)
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(b'{ "schema_version": "x" }\n')
    with pytest.raises(CloudProviderError, match="canonical"):
        load_canonical_json(noncanonical)

    extension = prepared.bundle_root / "provider-response.json"
    extension.write_text("{}", encoding="utf-8")
    with pytest.raises(CloudProviderError, match="extensions"):
        load_hf_source_transport(
            _context,
            transport_root=prepared.root,
            descriptor_uri=_descriptor_uri,
            source_lock_uri=_source_lock_uri,
        )


def test_invalid_receipt_shapes_and_secret_like_extensions_are_rejected(tmp_path: Path) -> None:
    _context, prepared, _source_lock_uri, _descriptor_uri = _prepare(tmp_path)
    for key, value in (
        ("authority", "launcher"),
        ("provider_receipt_id", "https://signed.example/?token=secret"),
        ("provider_receipt_id", "hf_actualSecretValue"),
        ("asserted_at", "yesterday"),
    ):
        evidence = _evidence(prepared)
        evidence[key] = value
        with pytest.raises(CloudProviderError, match="invalid|known-secret pattern"):
            validate_hf_provisioning_evidence(evidence)


@pytest.mark.parametrize(
    "value",
    [
        "xoxb-1234567890-secretvalue",
        "AKIAIOSFODNN7EXAMPLE",
        "token=credentialvalue",
        base64.b64encode(b"xoxb-1234567890-secretvalue").decode("ascii").rstrip("="),
        "aws_secret_access_key=bounded-secret-value",
        "AWS-SECRET-ACCESS-KEY:bounded-secret-value",
        "aws_session_token=bounded-session-value",
        quote("aws_secret_access_key=bounded-secret-value", safe=""),
        base64.b64encode(b"aws_session_token=bounded-session-value").decode("ascii"),
    ],
)
def test_known_secret_patterns_and_encoded_forms_are_rejected(
    tmp_path: Path,
    value: str,
) -> None:
    _context, prepared, _source_lock_uri, _descriptor_uri = _prepare(tmp_path)
    evidence = _evidence(prepared)
    evidence["provider_receipt_id"] = value
    with pytest.raises(CloudProviderError, match="invalid|known-secret pattern"):
        validate_hf_provisioning_evidence(evidence)


def test_tighter_https_policy_survives_prepare_load_and_consume(tmp_path: Path) -> None:
    policy = CheckoutPolicy(
        allowed_hosts=frozenset({"git.example.test"}),
        allowed_schemes=frozenset({"https"}),
    )
    context, prepared, source_lock_uri, descriptor_uri = _prepare(
        tmp_path,
        checkout_policy=policy,
    )
    loaded = load_hf_source_transport(
        context,
        transport_root=prepared.root,
        descriptor_uri=descriptor_uri,
        source_lock_uri=source_lock_uri,
    )
    assert loaded.checkout_policy == policy
    consumed = consume_hf_source_transport(
        context,
        transport_root=prepared.root,
        descriptor_uri=descriptor_uri,
        source_lock_uri=source_lock_uri,
        evidence=_evidence(prepared),
    )
    assert consumed.prepared.checkout_policy == policy


@pytest.mark.parametrize(
    ("location", "policy", "message"),
    [
        (
            "https://other.example.test/team/engine.git",
            CheckoutPolicy(
                allowed_hosts=frozenset({"git.example.test"}),
                allowed_schemes=frozenset({"https"}),
            ),
            "actual project repository host",
        ),
        (
            "ssh://git@git.example.test/team/engine.git",
            CheckoutPolicy(
                allowed_hosts=frozenset({"git.example.test"}),
                allowed_schemes=frozenset({"https"}),
            ),
            "actual project repository scheme",
        ),
        (
            "ssh://git@git.example.test/team/engine.git",
            CheckoutPolicy(
                allowed_hosts=frozenset({"git.example.test"}),
                allowed_schemes=frozenset({"ssh"}),
            ),
            "controlled SSH",
        ),
    ],
)
def test_prepare_rejects_policy_incompatible_with_actual_repository_location(
    tmp_path: Path,
    location: str,
    policy: CheckoutPolicy,
    message: str,
) -> None:
    context, lock = _fixture(tmp_path)
    source = replace(lock.project_source, location=RepositoryLocation.parse(location))
    if source.location.host != lock.engine_source.location.host:
        engine_source = replace(
            lock.engine_source,
            submodule_path="vendor/engine",
            gitlink_commit=lock.engine_source.commit,
        )
        incompatible = replace(
            lock,
            mode="dual_clone",
            project_source=source,
            engine_source=engine_source,
        )
    else:
        incompatible = replace(lock, project_source=source, engine_source=source)
    with pytest.raises(CloudProviderError, match=message):
        prepare_hf_source_transport(
            context,
            source_lock=incompatible,
            source_lock_uri="tracking://experiments/run-test/source-lock.json",
            descriptor_uri="tracking://experiments/run-test/cloud/hf/source-transport/descriptor.json",
            transport_root=(tmp_path / "incompatible").resolve(),
            volume_source="owner/bootstrap-bucket",
            path_prefix="prepared/sources",
            checkout_policy=policy,
        )


def test_prepare_validates_actual_engine_repository_independently(tmp_path: Path) -> None:
    context, lock = _fixture(tmp_path)
    project_source = replace(
        lock.project_source,
        submodule_path=None,
        gitlink_commit=None,
    )
    engine_source = replace(
        lock.engine_source,
        location=RepositoryLocation.parse("https://engine.example.test/team/engine.git"),
        submodule_path="vendor/engine",
        gitlink_commit=lock.engine_source.commit,
    )
    dual = replace(
        lock,
        mode="dual_clone",
        project_source=project_source,
        engine_source=engine_source,
    )
    persisted = CheckoutPolicy(
        allowed_hosts=frozenset({"git.example.test"}),
        allowed_schemes=frozenset({"https"}),
    )
    current = CheckoutPolicy(
        allowed_hosts=frozenset({"git.example.test", "engine.example.test"}),
        allowed_schemes=frozenset({"https"}),
    )
    with patch(
        "tuner.cloud.hf_provisioning.checkout_policy_from_context",
        return_value=current,
    ), pytest.raises(CloudProviderError, match="actual engine repository host"):
        prepare_hf_source_transport(
            context,
            source_lock=dual,
            source_lock_uri="tracking://experiments/run-test/source-lock.json",
            descriptor_uri="tracking://experiments/run-test/cloud/hf/source-transport/descriptor.json",
            transport_root=(tmp_path / "engine-incompatible").resolve(),
            volume_source="owner/bootstrap-bucket",
            path_prefix="prepared/sources",
            checkout_policy=persisted,
        )


@pytest.mark.parametrize("operation", ["load", "consume"])
def test_load_and_consume_revalidate_actual_repository_against_persisted_policy(
    tmp_path: Path,
    operation: str,
) -> None:
    context, lock = _fixture(tmp_path)
    engine_source = replace(
        lock.engine_source,
        submodule_path="vendor/engine",
        gitlink_commit=lock.engine_source.commit,
    )
    incompatible = replace(
        lock,
        project_source=replace(
            lock.project_source,
            location=RepositoryLocation.parse("https://other.example.test/team/project.git"),
        ),
        engine_source=engine_source,
        mode="dual_clone",
    )
    policy = CheckoutPolicy(
        allowed_hosts=frozenset({"git.example.test"}),
        allowed_schemes=frozenset({"https"}),
    )
    with patch("tuner.cloud.hf_provisioning._validate_policy_against_context"):
        prepared = prepare_hf_source_transport(
            context,
            source_lock=incompatible,
            source_lock_uri="tracking://experiments/run-test/source-lock.json",
            descriptor_uri="tracking://experiments/run-test/cloud/hf/source-transport/descriptor.json",
            transport_root=(tmp_path / "persisted-incompatible").resolve(),
            volume_source="owner/bootstrap-bucket",
            path_prefix="prepared/sources",
            checkout_policy=policy,
        )
    arguments = dict(
        context=context,
        transport_root=prepared.root,
        descriptor_uri=prepared.descriptor_uri,
        source_lock_uri=str(prepared.descriptor["source_lock"]["uri"]),
    )
    with pytest.raises(CloudProviderError, match="actual project repository host"):
        if operation == "load":
            load_hf_source_transport(**arguments)
        else:
            consume_hf_source_transport(**arguments, evidence=_evidence(prepared))


def test_controlled_ssh_policy_survives_prepare_load_and_consume(tmp_path: Path) -> None:
    executable = (tmp_path / "controlled-ssh.exe").resolve()
    known_hosts = (tmp_path / "known_hosts").resolve()
    executable.write_bytes(b"test executable")
    known_hosts.write_text("git.example.test ssh-ed25519 AAAATEST\n", encoding="utf-8")
    policy = CheckoutPolicy(
        allowed_hosts=frozenset({"git.example.test"}),
        allowed_schemes=frozenset({"https", "ssh"}),
        ssh=SSHCheckoutPolicy(
            ssh_executable=executable,
            agent_socket="controlled-agent-id",
            known_hosts=known_hosts,
        ),
    )
    context, prepared, source_lock_uri, descriptor_uri = _prepare(
        tmp_path,
        checkout_policy=policy,
    )
    consumed = consume_hf_source_transport(
        context,
        transport_root=prepared.root,
        descriptor_uri=descriptor_uri,
        source_lock_uri=source_lock_uri,
        evidence=_evidence(prepared),
    )
    assert consumed.prepared.checkout_policy == policy


def test_policy_broader_than_current_context_fails_closed(tmp_path: Path) -> None:
    context, lock = _fixture(tmp_path)
    broader = CheckoutPolicy(
        allowed_hosts=frozenset({"git.example.test", "evil.example"}),
        allowed_schemes=frozenset({"https"}),
    )
    with pytest.raises(CloudProviderError, match="outside the current context"):
        prepare_hf_source_transport(
            context,
            source_lock=lock,
            source_lock_uri="tracking://experiments/run-test/source-lock.json",
            descriptor_uri="tracking://experiments/run-test/cloud/hf/source-transport/descriptor.json",
            transport_root=(tmp_path / "transport").resolve(),
            volume_source="owner/bootstrap-bucket",
            path_prefix="prepared/sources",
            checkout_policy=broader,
        )


class _ProviderVolume:
    def __init__(self, spec) -> None:
        self.read_only = True
        self._wire = {
            "type": "bucket",
            "source": spec.source,
            "path": spec.path,
            "mountPath": spec.mount_path,
            "readOnly": True,
        }

    def to_dict(self):
        return dict(self._wire)


def test_submission_revalidates_consumed_binding_then_fails_at_approval_gate(tmp_path: Path) -> None:
    context, prepared, source_lock_uri, descriptor_uri = _prepare(tmp_path)
    consumed = consume_hf_source_transport(
        context,
        transport_root=prepared.root,
        descriptor_uri=descriptor_uri,
        source_lock_uri=source_lock_uri,
        evidence=_evidence(prepared),
    )
    volume = HFVerifiedVolume(
        spec=consumed.volume_spec,
        provider_volume=_ProviderVolume(consumed.volume_spec),
        descriptor_sha256=prepared.descriptor_sha256,
        provisioning_evidence_sha256=consumed.evidence_sha256,
        descriptor_uri=descriptor_uri,
        source_lock_uri=source_lock_uri,
        transport_root=prepared.root,
        provisioning_evidence=consumed.evidence,
        verification_context=context,
    )
    calls: list[dict[str, object]] = []
    executor = HFJobExecutor(SimpleNamespace(run_job=lambda **kwargs: calls.append(kwargs)))
    with pytest.raises(CloudProviderError, match="exact-run approval"):
        executor.submit(
            CloudJobSpec(
                provider="hf_jobs",
                image="image@sha256:test",
                command=["true"],
                flavor="cpu-basic",
                volumes=(volume,),
            )
        )
    assert calls == []


def test_submission_rejects_forged_consumed_binding_before_approval_gate(tmp_path: Path) -> None:
    context, prepared, source_lock_uri, descriptor_uri = _prepare(tmp_path)
    consumed = consume_hf_source_transport(
        context,
        transport_root=prepared.root,
        descriptor_uri=descriptor_uri,
        source_lock_uri=source_lock_uri,
        evidence=_evidence(prepared),
    )
    volume = HFVerifiedVolume(
        spec=consumed.volume_spec,
        provider_volume=_ProviderVolume(consumed.volume_spec),
        descriptor_sha256="D" * 64,
        provisioning_evidence_sha256=consumed.evidence_sha256,
        descriptor_uri=descriptor_uri,
        source_lock_uri=source_lock_uri,
        transport_root=prepared.root,
        provisioning_evidence=consumed.evidence,
        verification_context=context,
    )
    with pytest.raises(CloudProviderError, match="canonical CONSUMABLE"):
        HFJobExecutor(SimpleNamespace(run_job=lambda **_kwargs: None)).submit(
            CloudJobSpec(
                provider="hf_jobs",
                image="image@sha256:test",
                command=["true"],
                flavor="cpu-basic",
                volumes=(volume,),
            )
        )


def test_submission_rejects_changed_volume_tuple_and_writable_wire(tmp_path: Path) -> None:
    context, prepared, source_lock_uri, descriptor_uri = _prepare(tmp_path)
    consumed = consume_hf_source_transport(
        context,
        transport_root=prepared.root,
        descriptor_uri=descriptor_uri,
        source_lock_uri=source_lock_uri,
        evidence=_evidence(prepared),
    )

    def bound(spec, provider_volume):
        return HFVerifiedVolume(
            spec=spec,
            provider_volume=provider_volume,
            descriptor_sha256=prepared.descriptor_sha256,
            provisioning_evidence_sha256=consumed.evidence_sha256,
            descriptor_uri=descriptor_uri,
            source_lock_uri=source_lock_uri,
            transport_root=prepared.root,
            provisioning_evidence=consumed.evidence,
            verification_context=context,
        )

    calls: list[dict[str, object]] = []
    executor = HFJobExecutor(SimpleNamespace(run_job=lambda **kwargs: calls.append(kwargs)))
    changed_spec = HFVerifiedVolumeSpec(
        source=consumed.volume_spec.source,
        path=consumed.volume_spec.path + "-changed",
        mount_path=consumed.volume_spec.mount_path,
        capsule_path=consumed.volume_spec.capsule_path,
        capsule_manifest_sha256=consumed.volume_spec.capsule_manifest_sha256,
        source_lock_path=consumed.volume_spec.source_lock_path,
        source_lock_sha256=consumed.volume_spec.source_lock_sha256,
        checkout_policy_path=consumed.volume_spec.checkout_policy_path,
        checkout_policy_sha256=consumed.volume_spec.checkout_policy_sha256,
        local_root=consumed.volume_spec.local_root,
    )
    with pytest.raises(CloudProviderError, match="volume tuple changed"):
        executor.submit(
            CloudJobSpec(
                provider="hf_jobs",
                image="image@sha256:test",
                command=["true"],
                flavor="cpu-basic",
                volumes=(bound(changed_spec, _ProviderVolume(changed_spec)),),
            )
        )

    writable = _ProviderVolume(consumed.volume_spec)
    writable.read_only = False
    writable._wire["readOnly"] = False
    with pytest.raises(CloudProviderError, match="serialization semantics|read-only"):
        executor.submit(
            CloudJobSpec(
                provider="hf_jobs",
                image="image@sha256:test",
                command=["true"],
                flavor="cpu-basic",
                volumes=(bound(consumed.volume_spec, writable),),
            )
        )
    assert calls == []
