"""Fresh full-host cancellation against retained stores and a fixed fake call.

No real Modal client, credentials or service is used. Test-only capabilities
exercise existing machinery; production descriptors remain disabled.
"""

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.training.test_modal_cancel_machinery import _scripted
from tuner.training import modal_standalone_runner as runner

_SOURCE = Path(runner.__file__).resolve().parents[2]
_TESTS = Path(__file__).resolve().parents[2]
_BOOTSTRAP = """
import sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from tests.training.test_modal_cancel_process_restart import _cold
_cold(Path(sys.argv[3]), sys.argv[4])
"""


def _cold(control_path, operation):
    from contextlib import ExitStack
    from synaptic_tuner.api.v1 import TrainingRunRef, SecretRef
    from synaptic_tuner.api.v1.training_facade import TrainingRequest
    from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
    from synaptic_tuner.api.v1.reference.training import (
        ReferenceAuthorizationError,
        _commitment_document,
    )
    from tuner.training import modal_host_composition as composer
    from tuner.training.modal_retained_reader import _typed_inputs, _Keys
    from tuner.training.modal_run_retention import (
        load_retained_modal_run,
        WORKFLOW_FILE,
    )
    from tuner.training.modal_host_storage import ModalHostStorageV1
    from tuner.training.modal_host_effects import ModalPackagedHostEffectsV1
    from tuner.training.modal_host_authority import HMACAuthenticator
    from tuner.project.context import ProjectContext
    from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
    from tuner.execution.providers.modal.packaged_deployment import (
        ModalPackagedDeploymentObserver,
    )
    from tuner.execution.providers.modal.packaged_reader import ModalPackagedReader
    from tuner.execution.providers.modal.packaged_staging import (
        ModalPackagedInputStager,
    )
    from tuner.execution.providers.modal.packaged_composition import (
        compose_modal_packaged_adapter,
        ModalPackagedCatalogPorts,
        ModalPackagedSourcePorts,
        ModalPackagedAuthorityPorts,
    )

    control = json.loads(control_path.read_bytes())
    run = TrainingRunRef.from_dict(control["run"])
    root = Path(control["root"])
    retained = load_retained_modal_run(root, expected_run=run)
    inputs, release, _, facts, scope, components, workload, provider, plan = (
        _typed_inputs(retained)
    )
    assert Path(runner.__file__).resolve().is_relative_to(_SOURCE)

    class Clock:
        def now_epoch(self):
            return control["epoch"] + 901

        def now(self):
            return (
                datetime.fromtimestamp(self.now_epoch(), timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

        now_iso = now

    clock = Clock()
    client = object()

    class Call:
        object_id = "fc-owned"
        is_hydrated = True

        def __init__(self, supplied):
            self.client = supplied

        def hydrate(self, supplied):
            assert self.client is supplied
            return self

        def cancel(self, *, terminate_containers):
            assert terminate_containers is True and operation == "cancel"
            with open(control_path.parent / "cancel-events", "a") as output:
                output.write("fc-owned\n")
            if control["lost"]:
                raise RuntimeError("controlled lost reply")

    class Calls:
        @staticmethod
        def from_id(reference, *, client):
            assert reference == "fc-owned" and operation == "cancel"
            return Call(client)

    sdk = SimpleNamespace(__version__="1.5.4", FunctionCall=Calls)
    signer_ref = "modal-runtime-release-qualification-hmac-v1"
    signer = HMACAuthenticator(
        {signer_ref: retained.qualification_key},
        allowed_purposes=frozenset(
            {
                "modal-packaged-dispatch/v1",
                "modal-packaged-dispatch/v2",
                "modal-packaged-completion/v1",
                "modal-packaged-evaluation/v1",
                "modal-packaged-progress/v1",
            }
        ),
    )

    class DeniedSource:
        def open_lease(self):
            raise AssertionError("cancel reopened training source")

    original_descriptor = composer._modal_packaged_descriptor

    def descriptor(**kwargs):
        value = original_descriptor(**kwargs)
        return replace(
            value, capabilities=replace(value.capabilities, cancel=True, reconcile=True)
        )

    original_compose = composer.compose_reference_host

    def compose(**kwargs):
        return original_compose(**kwargs, allow_current_cancel_authorization=True)

    with pytest.MonkeyPatch.context() as patch, ExitStack() as stack:
        patch.setattr(composer, "_modal_packaged_descriptor", descriptor)
        patch.setattr(composer, "compose_reference_host", compose)
        storage = stack.enter_context(
            ModalHostStorageV1(
                Path(inputs["attempt_journal"]), run.project_ref, create=False
            )
        )
        records = stack.enter_context(
            SQLiteDurableStoreV1(root / WORKFLOW_FILE, create=False)
        )
        effects = ModalPackagedHostEffectsV1(
            storage=storage,
            runtime_release=release,
            provider_binding=provider,
            provider_facts=facts,
            execution_binding=components.execution_source,
            retained_source=DeniedSource(),
            workload_bytes=workload.canonical_bytes,
            artifact_policy=components.artifact_policy,
            signer=signer,
            key_ref=signer_ref,
            maximum_cost_minor_units=inputs["maximum_cost_minor_units"],
        )
        observation = SimpleNamespace(observe=lambda **kwargs: facts)
        deployment = ModalPackagedDeploymentObserver(
            sdk=sdk, client=client, client_binding=scope, reader=observation
        )
        facade = ExplicitModal154ReadFacade(
            scope,
            sdk=sdk,
            client=client,
            scope_observer=lambda supplied: (
                scope.account_ref,
                scope.workspace_ref,
                scope.environment_ref,
                scope.client_ref,
            ),
            deployment_observer=observation.observe,
            volume_names=dict(inputs["volume_names"]),
        )
        reader = ModalPackagedReader(
            facade=facade,
            deployment_observer=deployment,
            verifier=signer,
            key_ref=signer_ref,
        )
        adapter = compose_modal_packaged_adapter(
            profile_ref=control["profile_ref"],
            account_ref=facts.account_ref,
            namespace_ref=control["namespace"],
            sdk=sdk,
            client=client,
            facade=facade,
            deployment_observer=deployment,
            stager=ModalPackagedInputStager(facade),
            catalogs=ModalPackagedCatalogPorts(
                effects.bindings, effects.stage_receipts, effects.calls
            ),
            sources=ModalPackagedSourcePorts(
                effects.stages, effects.dispatches, effects.marker_materials
            ),
            authorities=ModalPackagedAuthorityPorts(effects, signer),
            clock=clock,
            trainer=SimpleNamespace(
                execute=lambda **kwargs: pytest.fail("cancel trained")
            ),
            reader=reader,
        )
        host = composer.compose_modal_packaged_reference_host(
            adapter=adapter,
            effects=effects,
            prepared_request=TrainingRequest(**inputs["prepared_request"]),
            run=run,
            components=components,
            context=ProjectContext.standalone(engine_root=Path(control["engine_root"])),
            input_preparation=None,
            clock=clock,
            secrets=_Keys(retained.signing_keys),
            authority_secret=SecretRef("env", "SYNAPTIC_TRAIN_AUTHORITY_KEY"),
            reader_secret=SecretRef("env", "SYNAPTIC_TRAIN_READER_KEY"),
            profile_digest=inputs["profile_digest"],
            quote_digest=inputs["quote_digest"],
            maximum_cost_minor_units=inputs["maximum_cost_minor_units"],
            records=records,
            streams=records,
        )
        resolved = host.api.training.resolve(
            TrainingRequest(**inputs["prepared_request"])
        )
        assert resolved.workload_digest == plan.basis.workload_digest
        assert (
            host.api.training.plan(
                resolved,
                host.composition.stores.planning_store.get_context(
                    plan.provider_plan.context_digest
                ).provider,
            )
            == plan
        )
        old = host.composition.stores.workflow_store.get(run)
        commitment = host.composition.coordinator._authorization._commitment(
            old.preflight_digest
        )
        before = _commitment_document(commitment)
        if operation == "cancel":
            for intent in (old.stage, old.submit):
                with pytest.raises(
                    ReferenceAuthorizationError, match="authorization_expired"
                ):
                    host.composition.coordinator._authorization.issue_effect_grant(
                        intent.canonical_command_bytes,
                        preflight_digest=old.preflight_digest,
                        now_epoch=clock.now_epoch(),
                    )
        else:
            patch.setattr(
                effects,
                "authorize",
                lambda *args, **kwargs: pytest.fail("repeat renewed authority"),
            )
            patch.setattr(
                effects,
                "bind",
                lambda *args, **kwargs: pytest.fail("repeat rebound command"),
            )
        result = host.api.runs.cancel(run, "operator stop")
        if control["lost"]:
            assert result.state.value == "reconcile_required"
            if operation == "cancel":
                assert host.api.runs.reconcile(run).state.value == "reconcile_required"
        else:
            assert result.state.value == "cancel_requested"
        current = host.composition.stores.workflow_store.get(run)
        assert current.submit == old.submit and current.stage == old.stage
        assert (
            _commitment_document(
                host.composition.coordinator._authorization._commitment(
                    old.preflight_digest
                )
            )
            == before
        )
        claim = storage.attempts.resolve(current.cancel.command_digest)
        assert claim is not None
        print(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "state": result.state.value,
                    "claim": claim.hex() if isinstance(claim, bytes) else claim,
                }
            )
        )


@pytest.mark.parametrize("lost", [False, True])
def test_fresh_process_full_host_cancel_and_no_reissue(tmp_path, monkeypatch, lost):
    control_path = tmp_path / "control.json"

    def action(host, run, clock, calls, events):
        root = tmp_path / "state/synaptic-training" / run.run_id
        control = {
            "run": run.to_dict(),
            "root": str(root),
            "lost": lost,
            "epoch": clock.now_epoch(),
            "namespace": host.adapter.effect_executor.namespace_ref,
            "profile_ref": host.adapter.effect_executor.profile_ref,
            "engine_root": str(tmp_path),
        }
        control_path.write_text(json.dumps(control))
        control_path.chmod(0o600)
        assert calls == [] and events == ["bootstrap", "cpu", "stage", "spawn"]

    _scripted(tmp_path, monkeypatch, action)
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
    }
    results = []
    for operation in ("cancel", "repeat"):
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _BOOTSTRAP,
                str(_TESTS),
                str(_SOURCE),
                str(control_path),
                operation,
            ],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr.decode()
        results.append(json.loads(completed.stdout))
    assert len({result["pid"] for result in results} | {os.getpid()}) == 3
    assert results[0]["state"] == results[1]["state"]
    assert results[0]["claim"] == results[1]["claim"]
    assert (tmp_path / "cancel-events").read_text() == "fc-owned\n"
