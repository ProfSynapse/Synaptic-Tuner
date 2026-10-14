"""Actual CPU child/receipt joined under explicit image/provider observations."""

import json

from tuner.execution.providers.modal import (
    runtime_release_qualification as qualification,
)
from tuner.execution.providers.modal.runtime_release_deployment import (
    ModalRuntimeReleaseDeploymentObservationV1,
)
from tests.execution.providers.test_modal_runtime_release_qualification import (
    _real_facts,
    Observer,
)
from tests.execution.providers.test_modal_packaged_binding import _release_and_execution
from tests.runtime.test_packaged_cpu_methods import _image, _release, _child


def test_grpo_cpu_parent_selects_method_and_publishes_actual_signed_receipt(
    tmp_path, monkeypatch
):
    image_root = tmp_path / "image"
    image_root.mkdir()
    config_path, config = _image(image_root, monkeypatch)
    release = _release("grpo", config)
    binding = _release_and_execution()[1].client_binding
    facts = _real_facts(release, binding)
    from tuner.runtime.releases import ProviderRuntimeBindingV1

    provider = ProviderRuntimeBindingV1.build(
        provider_ref="modal",
        runtime_release=release,
        provider_facts_schema=facts.schema_version,
        provider_facts_digest=facts.facts_digest,
    )
    fixture = qualification.ModalRuntimeReleaseFixtureReceiptV1.create(
        effect_id="grpo-cpu", artifact_volume_id="vo-artifact"
    )
    dispatch = qualification.ModalRuntimeReleaseQualificationDispatchV1(
        "grpo-cpu",
        release,
        provider,
        facts,
        fixture,
        qualification.ModalRuntimeReleaseQualificationPolicyV1(),
        qualification.QUALIFICATION_HMAC_KEY_REF,
    )
    auth = qualification.ModalRuntimeQualificationHmacAuthenticator(b"k" * 32)
    payload = qualification.build_modal_runtime_release_qualification_dispatch(
        dispatch, auth
    )
    roots = qualification.ModalRuntimeReleaseQualificationRoots(
        tmp_path / "control", tmp_path / "artifacts"
    )
    roots.control.mkdir()
    roots.artifacts.mkdir()
    staged = roots.artifacts / fixture.path
    staged.parent.mkdir(parents=True)
    staged.write_bytes(qualification.LOCAL_CPU_DATA)
    current = ModalRuntimeReleaseDeploymentObservationV1(
        facts.app_name,
        facts.client_binding.environment_ref,
        True,
        facts.app_id,
        "",
        facts.generation,
        tuple(sorted((i.spec.name, i.function_id) for i in facts.functions)),
    )
    calls = _child(monkeypatch, config_path)
    inspection = []

    def inspect(value, **kwargs):
        assert value == release and kwargs == {"method": "grpo"}
        inspection.append(value.manifest_digest)

    monkeypatch.setattr(
        "tuner.runtime.packaged_sft_execution._inspect_parent_release", inspect
    )
    worker = qualification.ModalRuntimeReleaseQualificationWorker(
        expected_facts=facts,
        verifier=auth,
        signer=auth,
        observer=Observer(current),
        call_id_provider=lambda: "fc-grpo-cpu",
        roots=roots,
    )
    commits = []
    result = worker(
        payload,
        commit_artifacts=lambda: commits.append("artifacts"),
        commit_control=lambda: commits.append("control"),
    )
    assert result["status_code"] == "completed", result
    assert len(calls) == 1 and inspection == [release.manifest_digest]
    assert commits == ["artifacts", "control"]
    receipt_path = next(roots.control.rglob("receipt.json"))
    receipt = qualification.ModalRuntimeReleaseQualificationReceiptV1.parse(
        receipt_path.read_bytes()
    )
    tag = receipt_path.with_name("receipt.mac").read_bytes()
    assert (
        auth.verify(
            "modal-runtime-release-qualification-receipt/v1",
            receipt.canonical_bytes,
            tag,
            qualification.QUALIFICATION_HMAC_KEY_REF,
        )
        is True
    )
    output = json.loads((roots.artifacts / receipt.output_path).read_bytes())
    assert output["child"]["runtime_release_digest"] == release.manifest_digest
    assert output["child"]["trainer_source_compiled"] is True
    assert output["child"]["training_executed"] is False
    assert output["gpu_qualified"] is False
