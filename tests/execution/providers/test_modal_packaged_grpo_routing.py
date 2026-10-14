"""Actual owned GRPO child/native chain with explicit CPU/provider substitutes."""

from dataclasses import replace
import hashlib

import pytest

from tuner.execution.foundation_v2.commands import (
    build_submit_command,
    CanonicalProviderPayloadV1,
)
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.references import StagePredecessorV2
from tuner.execution.providers.modal.packaged_binding import ModalPackagedCommandBinding
from tuner.execution.providers.modal.packaged_dispatch import (
    _packaged_workload,
    build_modal_packaged_dispatch,
    parse_modal_packaged_dispatch,
)
from tuner.execution.providers.modal.packaged_staging import ModalPackagedStageReceipt
from tuner.execution.providers.modal.contracts import (
    operation_path,
    provider_entry_identity,
)
from tuner.execution.providers.modal.packaged_worker import (
    InstalledPackagedSFTTrainerExecutor,
    ModalPackagedWorker,
    ModalPackagedWorkerRoots,
)
from tuner.runtime.releases import PackagedExecutionBindingV1
from tests.execution.providers.test_modal_packaged_binding import _binding
from tests.execution.providers.test_modal_packaged_dispatch import Auth
from tests.execution.providers.test_modal_packaged_worker import Signer
from tests.runtime.test_packaged_grpo_native import material
from tests.runtime.test_packaged_grpo_child import _ports, _prepare


def _case(tmp_path):
    custody_root = tmp_path / "material"
    custody_root.mkdir()
    admitted, compiled = material(custody_root)
    raw = admitted.paths.prepared_input.read_bytes()
    old = _binding(raw)
    release = admitted.release
    facts = replace(
        old.provider_facts,
        image_digest=release.image_digest,
        package_digest=release.package_digest,
        installed_distributions_digest=release.installed_distributions_digest,
        worker_closure_digest=release.worker_closure_digest,
    )
    provider = facts.build_provider_binding(release)
    value = admitted.execution.to_dict()
    execution = PackagedExecutionBindingV1.build(
        run_ref=value["run_ref"],
        runtime_release=release,
        provider_runtime_binding=provider,
        prepared_input_ref=admitted.execution.prepared_input_ref,
        prepared_input_revision=admitted.execution.prepared_input_revision,
        prepared_input_content_digest=admitted.execution.prepared_input_content_digest,
        prepared_input_size_bytes=admitted.execution.prepared_input_size_bytes,
        prepared_input_format=admitted.execution.prepared_input_format,
        workload_digest=compiled.fingerprint,
        configuration_digest=admitted.execution.configuration_digest,
        artifact_policy_digest=admitted.execution.artifact_policy_digest,
    )
    prep = old.command.preparation.to_dict()
    prep["run_id"] = execution.run_ref
    prep["source_digest"] = execution.binding_digest
    prep["workload_digest"] = execution.workload_digest
    prep.pop("schema_version")
    preparation = CanonicalPreparationV2.build(
        **{
            **prep,
            "provider": old.command.preparation.provider,
            "scope": old.command.preparation.scope,
        }
    )
    predecessor = StagePredecessorV2(
        preparation.provider.provider_id,
        preparation.provider.profile_ref,
        preparation.scope.account_ref,
        preparation.scope.namespace_ref,
        preparation.project_ref,
        preparation.run_id,
        preparation.plan_fingerprint,
        preparation.preparation_digest,
        preparation.workload_digest,
        old.command.operation.effect.effect_id,
        "8" * 64,
        "9" * 64,
    )
    submit = build_submit_command(
        preparation,
        "grpo-submit",
        CanonicalProviderPayloadV1.build(
            "modal", "submit-payload/v2", execution.workload_digest
        ),
        old.command.executor,
        predecessor,
    )
    binding = ModalPackagedCommandBinding(
        submit.canonical_bytes,
        release.canonical_bytes(),
        provider.canonical_bytes(),
        facts.canonical_bytes,
        execution.canonical_bytes(),
    )
    path = operation_path(
        predecessor.stage_effect_id,
        "input",
        "prepared",
        execution.prepared_input_content_digest,
        "payload.bin",
    )
    receipt = ModalPackagedStageReceipt(
        predecessor.stage_effect_id,
        execution.binding_digest,
        facts.artifact_volume_id,
        path,
        len(raw),
        hashlib.sha256(raw).hexdigest(),
        provider_entry_identity(facts.artifact_volume_id, path, len(raw)),
    )
    policy = admitted.artifact_policy
    admitted.close()
    roots = ModalPackagedWorkerRoots(
        *(tmp_path / name for name in ("control", "artifacts", "cache"))
    )
    for root in (roots.control, roots.artifacts, roots.cache):
        root.mkdir()
    staged = roots.artifacts / path
    staged.parent.mkdir(parents=True)
    staged.write_bytes(raw)
    auth = Auth()
    dispatch = build_modal_packaged_dispatch(
        binding,
        receipt,
        compiled.canonical_bytes,
        policy,
        auth,
        key_ref="dispatch-key",
        environment=(),
    )
    return binding, compiled, policy, roots, dispatch, auth


def test_actual_child_native_verification_precedes_signed_completion(
    tmp_path, monkeypatch
):
    binding, compiled, policy, roots, dispatch, auth = _case(tmp_path)
    commands = _ports(monkeypatch)
    from tuner.training.modal_host_authority import HMACAuthenticator

    signer = HMACAuthenticator(
        {"dispatch-key": b"q" * 32},
        allowed_purposes=frozenset({"modal-packaged-completion/v1"}),
    )
    worker = ModalPackagedWorker(
        expected_facts=binding.provider_facts,
        dispatch_verifier=auth,
        trainer_executor=InstalledPackagedSFTTrainerExecutor(model_preparer=_prepare),
        evidence_signer=signer,
        roots=roots,
    )
    result = worker(
        dispatch,
        provider_job_ref="fc-grpo",
        commit_control=lambda: None,
        commit_artifacts=lambda: None,
    )
    assert result["status_code"] == "completed", result
    assert len(commands) == 1
    _readback(binding, roots, signer, result)
    assert _packaged_workload(compiled.canonical_bytes) == compiled
    assert (
        parse_modal_packaged_dispatch(dispatch, auth).workload_bytes
        == compiled.canonical_bytes
    )


@pytest.mark.parametrize("hook", ["runner", "evaluation"])
def test_sft_only_hooks_rejected_before_grpo_admission(tmp_path, monkeypatch, hook):
    from tuner.runtime import packaged_grpo_execution as runtime

    _, compiled, _, _, _, _ = _case(tmp_path)
    monkeypatch.setattr(
        runtime, "admit_packaged_grpo", lambda **kwargs: pytest.fail("admitted hook")
    )
    executor = InstalledPackagedSFTTrainerExecutor(
        model_preparer=lambda *args: None,
        runner=(lambda *args: None) if hook == "runner" else None,
    )
    with pytest.raises(ValueError):
        executor.execute(
            runtime_release=None,
            provider_binding=None,
            execution_binding=None,
            workload_bytes=compiled.canonical_bytes,
            artifact_policy=None,
            paths=None,
            environment=(),
            on_training_complete=(lambda *args: None) if hook == "evaluation" else None,
        )


def test_grpo_evaluation_rejects_before_provider_read(tmp_path):
    from tuner.execution.providers.modal.packaged_reader import ModalPackagedReader

    _, compiled, _, _, _, _ = _case(tmp_path)
    reader = object.__new__(ModalPackagedReader)
    with pytest.raises(ValueError, match="GRPO evaluation is unsupported"):
        reader.read_evaluation(
            None, provider_job_ref="fc-grpo", workload_bytes=compiled.canonical_bytes
        )


@pytest.mark.parametrize("mutation", ["lineage", "metrics", "exit"])
def test_native_failure_never_signs_or_commits_completion(
    tmp_path, monkeypatch, mutation
):
    from tests.runtime.test_packaged_grpo_child import _BOOTSTRAP

    binding, _, _, roots, dispatch, auth = _case(tmp_path)
    change = "result=child.main()\n"
    if mutation == "exit":
        change += "result=123\n"
    else:
        name = "training_lineage" if mutation == "lineage" else "training_metrics"
        key = "actual_steps" if mutation == "lineage" else "global_step"
        change += (
            "payload_for_mutation=json.loads(Path(grp_args[1]).read_bytes())\n"
            f"changed_path=Path(payload_for_mutation['paths']['state'])/'grpo-output'/'{name}.json'\n"
            "changed_document=json.loads(changed_path.read_bytes())\n"
            f"changed_document['{key}']=999\n"
            "changed_path.write_text(json.dumps(changed_document))\n"
        )
    commands = _ports(monkeypatch, _BOOTSTRAP.replace("result=child.main()\n", change))
    signer = Signer()
    commits = []
    worker = ModalPackagedWorker(
        expected_facts=binding.provider_facts,
        dispatch_verifier=auth,
        trainer_executor=InstalledPackagedSFTTrainerExecutor(model_preparer=_prepare),
        evidence_signer=signer,
        roots=roots,
    )
    result = worker(
        dispatch,
        provider_job_ref="fc-grpo",
        commit_control=lambda: commits.append("control"),
        commit_artifacts=lambda: commits.append("artifacts"),
    )
    assert result["status_code"] == "failed"
    assert result["failure_stage"] in {"GRPO_TRAINER", "GRPO_ARTIFACT"}
    assert len(commands) == 1 and not signer.calls and not commits


def test_changed_staged_input_rejects_before_child(tmp_path, monkeypatch):
    binding, _, _, roots, dispatch, auth = _case(tmp_path)
    parsed = parse_modal_packaged_dispatch(dispatch, auth)
    (roots.artifacts / parsed.stage_receipt.path).write_bytes(b"changed")
    commands = _ports(monkeypatch)
    signer = Signer()
    worker = ModalPackagedWorker(
        expected_facts=binding.provider_facts,
        dispatch_verifier=auth,
        trainer_executor=InstalledPackagedSFTTrainerExecutor(model_preparer=_prepare),
        evidence_signer=signer,
        roots=roots,
    )
    result = worker(
        dispatch,
        provider_job_ref="fc-grpo",
        commit_control=lambda: pytest.fail("commit"),
        commit_artifacts=lambda: pytest.fail("commit"),
    )
    assert result["failure_stage"] == "STAGED_INPUT"
    assert not commands and not signer.calls


@pytest.mark.parametrize(
    "field", ["schema_version", "method", "limits", "core_workload"]
)
def test_canonical_workload_rewrite_rejects_before_dispatch(tmp_path, field):
    from tuner.training.recipes import canonical_json_bytes

    _, compiled, _, _, _, _ = _case(tmp_path)
    value = compiled.document
    value[field] = {
        "schema_version": "synaptic-packaged-sft-workload/v1",
        "method": "sft",
        "limits": {"child_max_seconds": True},
        "core_workload": {},
    }[field]
    with pytest.raises(ValueError):
        _packaged_workload(canonical_json_bytes(value))


def _readback(binding, roots, signer, result):
    from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
    from tuner.execution.providers.modal.packaged_deployment import (
        ModalPackagedDeploymentObserver,
    )
    from tuner.execution.providers.modal.packaged_reader import ModalPackagedReader
    from tests.execution.providers.test_modal_sdk154_adapter import FakeVolume, SDK
    from tests.execution.providers.test_modal_packaged_deployment import Reader

    facts = binding.provider_facts
    client = object()
    control, artifacts = (
        FakeVolume(facts.control_volume_id),
        FakeVolume(facts.artifact_volume_id),
    )
    for volume, root in ((control, roots.control), (artifacts, roots.artifacts)):
        volume.files = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }
    FakeVolume.registry = {"grpo-control": control, "grpo-artifacts": artifacts}
    facade = ExplicitModal154ReadFacade(
        facts.client_binding,
        sdk=SDK,
        client=client,
        scope_observer=lambda supplied: (
            (
                facts.account_ref,
                facts.workspace_ref,
                facts.environment_ref,
                facts.client_ref,
            )
            if supplied is client
            else ()
        ),
        deployment_observer=lambda **kwargs: None,
        volume_names={
            facts.control_volume_id: "grpo-control",
            facts.artifact_volume_id: "grpo-artifacts",
        },
    )
    deployment = ModalPackagedDeploymentObserver(
        sdk=SDK,
        client=client,
        client_binding=facts.client_binding,
        reader=Reader(facts),
    )
    reader = ModalPackagedReader(
        facade=facade,
        deployment_observer=deployment,
        verifier=signer,
        key_ref="dispatch-key",
    )
    observed = reader.observe_completion(binding, provider_job_ref="fc-grpo")
    assert len(observed.members) == 5
    for member in observed.members:
        content = b"".join(
            reader.iter_artifact(
                binding, observed, role=member.role, maximum_bytes=member.size
            )
        )
        assert len(content) == member.size
        assert hashlib.sha256(content).hexdigest() == member.sha256
    assert observed.completion_digest == result["completion_sha256"]
    fresh = ModalPackagedReader(
        facade=facade,
        deployment_observer=deployment,
        verifier=signer,
        key_ref="dispatch-key",
    )
    assert fresh.observe_completion(binding, provider_job_ref="fc-grpo") == observed
    evidence = (
        operation_path(observed.effect_id, "evidence") + "/packaged-completion.mac"
    )
    control.files[evidence] = b"wrong-tag"
    with pytest.raises(ValueError, match="authentication failed"):
        fresh.observe_completion(binding, provider_job_ref="fc-grpo")


@pytest.mark.parametrize("name", ["PATH", "LD_LIBRARY_PATH"])
def test_unsupported_trainer_override_rejects_without_child(
    tmp_path, monkeypatch, name
):
    binding, _, _, roots, raw, auth = _case(tmp_path)
    parsed = parse_modal_packaged_dispatch(raw, auth)
    raw = build_modal_packaged_dispatch(
        binding,
        parsed.stage_receipt,
        parsed.workload_bytes,
        parsed.artifact_policy,
        auth,
        key_ref="dispatch-key",
        environment=((name, "/arbitrary"),),
    )
    commands = _ports(monkeypatch)
    signer = Signer()
    worker = ModalPackagedWorker(
        expected_facts=binding.provider_facts,
        dispatch_verifier=auth,
        trainer_executor=InstalledPackagedSFTTrainerExecutor(model_preparer=_prepare),
        evidence_signer=signer,
        roots=roots,
    )
    result = worker(
        raw,
        provider_job_ref="fc-grpo",
        commit_control=lambda: pytest.fail("commit"),
        commit_artifacts=lambda: pytest.fail("commit"),
    )
    assert result["failure_stage"] == "GRPO_ADMISSION"
    assert not commands and not signer.calls
