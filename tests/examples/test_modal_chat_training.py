from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.modal_chat.authority import HMACAuthenticator
from examples.modal_chat.deployment import ModalChatOwnedDeployment, ModalChatScope
from examples.modal_chat.remote import ModalChatGitRemote
from examples.modal_chat.replay import ModalChatEvidenceReplay
from examples.modal_chat.storage import ModalChatStorage
from examples.modal_chat.training import (
    _AllocatedIdentity,
    _training_quote,
    ModalChatRunIdentity,
    ModalChatTrainingCompositionError,
    compose_modal_training_host,
)
from synaptic_tuner.api.v1.results import TrainingRunRef
from tests.examples.test_modal_chat_deployment import _SDK, _profile
from tests.examples.test_modal_chat_resolution import _document, _source
from tuner.execution.providers.modal.composition import ModalVerificationPolicyV1
from tuner.execution.providers.modal.coordinator_preflight import QUOTE_PURPOSE
from tuner.project.context import ProjectContext
from tuner.project.git_verification import GitCliLocalSourceInspector


class Clock:
    def now_iso(self):
        return "2026-09-14T12:00:00Z"


class Billing:
    def rates(self):
        return {
            "gpu_hour_cost_a10g": Decimal("1.10000"),
            "cpu_hour_cost": Decimal("0.04730"),
            "mem_gib_hour_cost": Decimal("0.00800"),
        }


class Workspace:
    name = "workspace-a"
    is_hydrated = False
    billing = Billing()

    def hydrate(self, client):
        self.is_hydrated = True


class SDK:
    class Workspace:
        @staticmethod
        def from_context(*, client):
            return Workspace()


class Scope:
    sdk, client = SDK(), object()
    binding = SimpleNamespace(workspace_ref="workspace-a")

    def observe(self, client):
        assert client is self.client
        return ("workspace-a",) * 4


def test_training_quote_uses_decimal_workspace_rates_and_actual_binding() -> None:
    auth = HMACAuthenticator(
        {"quote-key": b"q" * 32}, allowed_purposes=frozenset({QUOTE_PURPOSE})
    )
    quote, trust, calculation = _training_quote(
        scope=Scope(),
        binding={
            "provider_id": "modal",
            "profile_ref": "profile-a",
            "account_ref": "workspace-a",
            "namespace_ref": "namespace-a",
            "resource_digest": "a" * 64,
            "timeout_seconds": 600,
        },
        maximum_cost_minor_units=20,
        issued_at="2026-09-14T12:00:00Z",
        expires_at="2026-09-14T12:05:00Z",
        issuer_ref="operator",
        audience_ref="project-run",
        challenge_nonce="quote-nonce",
        key_ref="quote-key",
        authenticator=auth,
        clock=Clock(),
    )
    assert quote.body.resource_digest == "a" * 64
    assert quote.body.namespace_ref == "namespace-a"
    assert quote.body.maximum_cost_minor_units == 20
    assert trust.key_ref == "quote-key"
    assert b'"gpu_hour_cost_a10g":"1.10000"' in calculation
    assert b'"gpu_only_timeout_estimate_minor_units":19' in calculation
    assert (
        b'"excluded_billing_dimensions":["build","cpu","memory","storage"'
        in calculation
    )
    assert b'"resource_digest":"' + (b"a" * 64) + b'"' in calculation
    assert auth.verify(QUOTE_PURPOSE, quote.body_bytes, quote.tag, "quote-key")


def test_training_quote_rejects_operator_authorization_below_nominal() -> None:
    auth = HMACAuthenticator(
        {"quote-key": b"q" * 32}, allowed_purposes=frozenset({QUOTE_PURPOSE})
    )
    with pytest.raises(ValueError):
        _training_quote(
            scope=Scope(),
            binding={
                "provider_id": "modal",
                "profile_ref": "profile-a",
                "account_ref": "workspace-a",
                "namespace_ref": "namespace-a",
                "resource_digest": "a" * 64,
                "timeout_seconds": 3600,
            },
            maximum_cost_minor_units=1,
            issued_at="2026-09-14T12:00:00Z",
            expires_at="2026-09-14T12:05:00Z",
            issuer_ref="operator",
            audience_ref="project-run",
            challenge_nonce="quote-nonce",
            key_ref="quote-key",
            authenticator=auth,
            clock=Clock(),
        )


def test_identity_is_allocated_once_and_rejects_changed_request() -> None:
    class Identities:
        calls = 0

        def allocate(self, *, project_ref, request_digest):
            self.calls += 1
            return "request-a", TrainingRunRef("run-a", project_ref)

    source = Identities()
    retained = _AllocatedIdentity(source, "project-a", "a" * 64)
    assert (
        retained.allocate(project_ref="project-a", request_digest="a" * 64)[0]
        == "request-a"
    )
    assert source.calls == 1
    with pytest.raises(ModalChatTrainingCompositionError, match="identity_changed"):
        retained.allocate(project_ref="project-a", request_digest="b" * 64)


def test_public_identity_returns_only_declared_project_run() -> None:
    identity = ModalChatRunIdentity(
        "request-a", TrainingRunRef("run-a", "project-a"), "a" * 64
    )
    assert identity.allocate(project_ref="project-a", request_digest="a" * 64) == (
        "request-a",
        TrainingRunRef("run-a", "project-a"),
    )
    with pytest.raises(ModalChatTrainingCompositionError, match="identity_changed"):
        identity.allocate(project_ref="project-b", request_digest="a" * 64)


def test_full_training_composer_starts_real_public_graph_once(
    monkeypatch, tmp_path: Path
) -> None:
    os.chmod(tmp_path, 0o700)
    sdk, client = _SDK(), object()

    class E2EClock:
        def now(self):
            return self.now_iso()

        def now_iso(self):
            return "2026-08-25T12:03:00Z"

        def now_epoch(self):
            return 1787659380

    class Rates:
        def rates(self):
            return Billing().rates()

    original_workspace = sdk.Workspace.from_context

    def workspace(*, client):
        value = original_workspace(client=client)
        value.billing = Rates()
        return value

    monkeypatch.setattr(sdk.Workspace, "from_context", workspace)
    files = {}

    class Batch:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put_file(self, stream, path):
            files[path] = stream.read()

    class Volume:
        is_hydrated = False

        def __init__(self, name):
            self.object_id = "vo-control" if name == "control" else "vo-artifact"

        def hydrate(self, client):
            self.is_hydrated = True

        def read_file(self, path):
            key = (self.object_id, path)
            return iter((files[key],)) if key in files else iter(())

        def iterdir(self, prefix, recursive):
            return iter(
                SimpleNamespace(path=path, size=len(data), mtime=1, type=1)
                for (volume_id, path), data in files.items()
                if volume_id == self.object_id and path.startswith(prefix)
            )

        def batch_upload(self, *, force):
            assert force is False
            volume_id = self.object_id

            class VolumeBatch(Batch):
                def put_file(self, stream, path):
                    files[(volume_id, path)] = stream.read()

            return VolumeBatch()

    monkeypatch.setattr(
        sdk.Volume,
        "from_name",
        staticmethod(lambda name, **kwargs: Volume(name)),
    )
    monkeypatch.setattr(sdk.Volume, "read_file", lambda *args: (), raising=False)
    monkeypatch.setattr(sdk.Volume, "iterdir", lambda *args: (), raising=False)

    class Secret:
        object_id = "st-1"
        is_hydrated = False

        def __init__(self, name):
            self.name = name

        def hydrate(self, client):
            self.is_hydrated = True

    monkeypatch.setattr(
        sdk.Secret,
        "from_name",
        staticmethod(lambda name, **kwargs: Secret(name)),
    )
    profile = _profile()
    storage = ModalChatStorage(tmp_path / "consumer.sqlite3", "namespace-a")
    scope = ModalChatScope(
        sdk=sdk,
        client=client,
        environment_name="environment-a",
        client_ref="host-a",
    )
    owner = ModalChatOwnedDeployment(
        scope=scope,
        storage=storage,
        profile=profile,
        runtime_environment={"ENGINE_MODE": "coordinator"},
        timeout_seconds=600,
        evidence_environment_key="EVIDENCE_KEY",
        evidence_key_ref="evidence-key",
        model_token_key="MODEL_TOKEN",
    )
    from examples.modal_chat import deployment as deployment_module
    from examples.modal_chat.deployment_readback import (
        CurrentModalDeployment,
        CurrentModalFunction,
    )

    def current_deployment(**kwargs):
        assert kwargs["client"] is client and kwargs["sdk"] is sdk
        assert kwargs["environment_name"] == "environment-a"
        assert kwargs["app_name"] == profile.app_name
        assert kwargs["function_name"] == profile.function_name
        if sdk.current_function is None:
            return None
        return CurrentModalDeployment(
            sdk.app_id,
            1,
            True,
            ((profile.function_name, sdk.current_function.object_id),),
            (),
            (
                CurrentModalFunction(
                    sdk.current_function.object_id,
                    profile.function_name,
                    sdk.app_id,
                    "",
                    sdk.definition_id,
                ),
            ),
        )

    monkeypatch.setattr(
        deployment_module, "read_current_deployment", current_deployment
    )
    owner.deploy_once(attempt_ref="deploy-a")
    spawn_calls = []

    class Call:
        object_id = "fc-training-1"

    def spawn(function, payload):
        spawn_calls.append(bytes(payload))
        return Call()

    monkeypatch.setattr(type(sdk.current_function), "spawn", spawn, raising=False)
    source_document = _source().to_dict()
    source_document["sources"]["project"]["branch"] = "main"
    source_document["sources"]["engine"]["branch"] = "main"
    from tuner.project.source_bundle import SourceLock

    source = SourceLock.from_dict(source_document)
    monkeypatch.setattr(
        GitCliLocalSourceInspector,
        "inspect",
        lambda self, *, context: source,
    )

    def remote(self, *, canonical_url, exact_ref):
        commit = (
            source.project_source.commit
            if canonical_url == source.project_source.location.canonical_url
            else source.engine_source.commit
        )
        return f"{commit}\t{exact_ref}\n".encode()

    monkeypatch.setattr(ModalChatGitRemote, "read_ref", remote)
    project = tmp_path / "project"
    engine = project / "vendor" / "engine"
    (project / "data").mkdir(parents=True)
    engine.mkdir(parents=True)
    (project / "data" / "train.jsonl").write_bytes(b'{"text":"row"}\n')
    context = ProjectContext.host(engine_root=engine, project_root=project)
    request_json = json.dumps(_document(), sort_keys=True, separators=(",", ":"))
    purposes = frozenset(
        {
            "source-lock-evidence/v1",
            "modal-deployment-evidence/v1",
            "modal-stage-claim/v2",
            "modal-launch-claim/v1",
            "provider-run-observation/v1",
            "provider-log-page/v1",
            QUOTE_PURPOSE,
        }
    )
    auth = HMACAuthenticator({"evidence-key": b"e" * 32}, allowed_purposes=purposes)
    clock = E2EClock()
    try:
        graph = compose_modal_training_host(
            scope=scope,
            deployment=owner,
            profile=profile,
            storage=storage,
            context=context,
            request_json=request_json,
            project_ref="project-a",
            request_identity=ModalChatRunIdentity(
                "request-a",
                TrainingRunRef("run-a", "project-a"),
                hashlib.sha256(request_json.encode()).hexdigest(),
            ),
            dataset_project_path=Path("data/train.jsonl"),
            allowed_git_refs=frozenset(
                {
                    (
                        source.project_source.location.canonical_url,
                        "refs/heads/" + source.project_source.branch,
                    ),
                    (
                        source.engine_source.location.canonical_url,
                        "refs/heads/" + source.engine_source.branch,
                    ),
                }
            ),
            load_in_4bit=False,
            runtime_environment={"ENGINE_MODE": "coordinator"},
            timeout_seconds=600,
            evidence_authenticator=auth,
            evidence_key_ref="evidence-key",
            source_policy=ModalVerificationPolicyV1(
                "project-run",
                "source-issuer",
                "deployment-issuer",
                "evidence-key",
                "evidence-key",
                lambda purpose: purpose + "-nonce",
                lambda purpose: purpose + "-evidence",
            ),
            replay=ModalChatEvidenceReplay(storage),
            clock=clock,
            created_at=clock.now_iso(),
            quote_issuer_ref="quote-issuer",
            quote_audience_ref="project-run",
            quote_challenge_nonce="quote-nonce",
            quote_key_ref="evidence-key",
            maximum_cost_minor_units=20,
            log_terminal_policy=json.dumps(
                {
                    "schema_version": "synaptic-modal-log-terminal-policy/v2",
                    "generation": 1,
                    "max_log_chunks": 10,
                    "max_chunk_bytes": 65536,
                    "max_terminal_bytes": 65536,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            control_volume_id="vo-control",
            artifact_volume_id="vo-artifact",
            stage_key_ref="evidence-key",
            artifact_key=b"a" * 32,
            grant_key=b"g" * 32,
            receipt_key=b"r" * 32,
            invalid_evidence_key=b"i" * 32,
            assessment_key=b"s" * 32,
            cursor_key=b"c" * 32,
            observed_at=clock.now_iso(),
        )
        request = graph.host.api.training.load(request_json)
        resolved = graph.host.api.training.resolve(request)
        plan = graph.host.api.training.plan(
            resolved, graph.preparation._snapshot()[0].provider
        )
        preflight = graph.host.api.training.preflight(plan)
        assert preflight.ready is True
        started = graph.host.api.training.start(plan, preflight)
        repeated = graph.host.api.training.start(plan, preflight)
        assert repeated == started
        assert len(spawn_calls) == 1
        assert (graph.material.run_id, graph.material.planning_request.project_ref) == (
            "run-a",
            "project-a",
        )
        assert sdk.deploy_calls
    finally:
        storage.close()
