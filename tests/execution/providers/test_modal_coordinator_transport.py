"""Provider-free tests for the Foundation-native Modal host transport."""

from dataclasses import dataclass, replace

import pytest

from tests.execution.providers.modal_coordinator_fixtures import real_launch_bundle_case
from tuner.execution.foundation_v2.commands import build_cancel_command, parse_exact_command
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.references import CancellationRefV1, ProviderRunRefV1
from tuner.execution.providers.modal.binding import ModalClientBinding
from tuner.execution.providers.modal.coordinator_binding import ModalCommandBinding, _NoPreflightClock
from tuner.execution.providers.modal.coordinator_adapter import ModalPreparationAdapter
from tuner.execution.providers.modal.coordinator_staging import modal_stage_provider_ref
from tuner.execution.providers.modal.coordinator_transport import ModalFoundationHostTransport
from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade


@dataclass
class Entry:
    path: str
    size: int
    mtime: int = 1
    type: int = 1


class Upload:
    def __init__(self, volume): self.volume, self.pending = volume, []
    def __enter__(self): return self
    def put_file(self, source, path): self.pending.append((path, source.read()))
    def __exit__(self, kind, value, trace):
        if value is None:
            for path, data in self.pending:
                if path in self.volume.files: raise RuntimeError("collision")
                self.volume.files[path] = data


class Volume:
    registry = {}
    calls = []
    def __init__(self, object_id): self.object_id, self.files, self.is_hydrated = object_id, {}, False
    @classmethod
    def from_name(cls, name, **kwargs): cls.calls.append((name, kwargs)); return cls.registry[name]
    def hydrate(self, client): self.is_hydrated = True
    def batch_upload(self, force=False): assert force is False; return Upload(self)
    def read_file(self, path): yield self.files[path]
    def iterdir(self, prefix, recursive=True):
        for path, data in sorted(self.files.items()):
            if path.startswith(prefix): yield Entry(path, len(data))


class Call:
    calls = []
    fail_cancel = False
    def __init__(self, object_id="fc-1"): self.object_id = object_id
    def cancel(self, **kwargs):
        type(self).calls.append(kwargs)
        if type(self).fail_cancel: raise TimeoutError


class FunctionCall:
    resolutions = []
    returned_id = None
    @classmethod
    def from_id(cls, value, client=None):
        cls.resolutions.append((value, client)); return Call(cls.returned_id or value)


class Function:
    resolutions = []
    spawns = []
    fail_spawn = False
    returned_id = "fc-1"
    def __init__(self): self.is_hydrated = False
    @classmethod
    def from_name(cls, app, name, **kwargs):
        cls.resolutions.append((app, name, kwargs)); return cls()
    def hydrate(self, client): self.is_hydrated = True
    def spawn(self, value):
        type(self).spawns.append(value)
        if type(self).fail_spawn: raise TimeoutError
        return Call(type(self).returned_id)


class SDK:
    __version__ = "1.5.4"
    Volume = Volume
    Function = Function
    FunctionCall = FunctionCall
    exception = type("Exceptions", (), {"NotFoundError": KeyError})


class Source:
    def __init__(self, value): self.value = value
    def resolve(self, digest): return self.value


def setup(monkeypatch):
    case = real_launch_bundle_case(monkeypatch)
    binding = case["envelope"].submit_binding
    selection = binding.deployment.selection
    profile = __import__(
        "tuner.execution.foundation_v2.canonical", fromlist=["parse_canonical_object"],
    ).parse_canonical_object(binding.preparation_snapshot, name="snapshot")["configuration"]["profile"]
    Volume.registry = {
        profile["volumes"]["control_ref"]: Volume("control-id"),
        profile["volumes"]["artifact_ref"]: Volume("artifact-id"),
    }
    Volume.calls = []; Function.resolutions = []; Function.spawns = []
    Function.fail_spawn = False; Function.returned_id = "fc-1"
    FunctionCall.resolutions = []; FunctionCall.returned_id = None
    Call.calls = []; Call.fail_cancel = False
    client = object()
    facade = ExplicitModal154ReadFacade(
        binding.client_binding, sdk=SDK, client=client,
        scope_observer=lambda supplied: (
            selection.account_ref, selection.workspace_ref,
            selection.environment_ref, selection.client_ref,
        ) if supplied is client else (),
        deployment_observer=lambda **kwargs: selection,
        volume_names={
            "control-id": profile["volumes"]["control_ref"],
            "artifact-id": profile["volumes"]["artifact_ref"],
        },
    )
    transport = ModalFoundationHostTransport(
        facade=facade, deployment=binding.deployment,
        stage_source=Source(case["material"]), launch_source=Source(case["envelope"]),
        binding_authority=case["authority"],
        foundation_authenticator=case["harness"].authenticator,
        assessment_authenticator=case["harness"].foundation,
        stage_verifier=case["authenticator"], launch_verifier=case["authenticator"],
        recipes=case["recipes"],
    )
    return case, transport, facade


def cancel_binding(case, nonce, target, reason):
    submit_binding = case["envelope"].submit_binding
    submit = parse_exact_command(submit_binding.command_bytes)
    adapter = ModalPreparationAdapter.restore(
        submit_binding.preparation_snapshot, clock=_NoPreflightClock(),
    )
    cancel = build_cancel_command(
        submit.preparation, nonce,
        adapter.payload(submit.preparation, EffectKind.CANCEL), submit.executor,
        CancellationRefV1(ProviderRunRefV1(target), reason),
    )
    binding = ModalCommandBinding(
        cancel.canonical_bytes, submit_binding.preparation_snapshot,
        submit_binding.deployment_bytes,
    )
    case["authority"].values.add(binding.canonical_bytes)
    return binding, cancel


def test_stage_writes_exact_three_files_and_returns_claim_identity(monkeypatch):
    case, transport, _ = setup(monkeypatch)
    material = case["material"]
    command = parse_exact_command(material.binding.command_bytes)
    result = transport.execute_once(material.binding, command)
    assert result == __import__(
        "tuner.execution.providers.modal.coordinator_effects", fromlist=["ModalEffectOutcome"],
    ).ModalEffectOutcome(ObservationDisposition.FOUND, modal_stage_provider_ref(material))
    assert sum(len(volume.files) for volume in Volume.registry.values()) == 3


def test_submit_prepares_real_bundle_then_spawns_once(monkeypatch):
    case, transport, _ = setup(monkeypatch)
    binding = case["envelope"].submit_binding
    result = transport.execute_once(binding, parse_exact_command(binding.command_bytes))
    assert result.provider_ref == "fc-1"
    assert len(Function.spawns) == 1 and Function.spawns[0].startswith(b'{"expectation"')


def test_spawn_timeout_is_indeterminate_and_never_retried(monkeypatch):
    case, transport, _ = setup(monkeypatch); Function.fail_spawn = True
    binding = case["envelope"].submit_binding
    result = transport.execute_once(binding, parse_exact_command(binding.command_bytes))
    assert result.disposition is ObservationDisposition.INDETERMINATE
    assert len(Function.spawns) == 1


def test_malformed_spawn_identity_is_indeterminate_after_one_attempt(monkeypatch):
    case, transport, _ = setup(monkeypatch); Function.returned_id = "not valid/id"
    binding = case["envelope"].submit_binding
    result = transport.execute_once(binding, parse_exact_command(binding.command_bytes))
    assert result.disposition is ObservationDisposition.INDETERMINATE
    assert len(Function.spawns) == 1


def test_retained_binding_mismatch_stops_before_sdk(monkeypatch):
    case, transport, _ = setup(monkeypatch)
    binding = case["envelope"].submit_binding
    transport._launch_source.value = case["envelope"].__class__(
        case["material"].binding, case["material"], case["envelope"].stage_record,
        case["envelope"].stage_assessment, case["envelope"].claim,
        case["envelope"].claim_tag,
    )
    with pytest.raises(ValueError, match="envelope mismatch"):
        transport.execute_once(binding, parse_exact_command(binding.command_bytes))
    assert Function.resolutions == [] and Function.spawns == [] and Volume.calls == []


def test_invalid_stage_authentication_stops_before_sdk(monkeypatch):
    case, transport, _ = setup(monkeypatch)
    material = case["material"]
    transport._stage_source.value = replace(material, claim_tag=b"wrong")
    with pytest.raises(ValueError, match="authentication failed"):
        transport.execute_once(
            material.binding, parse_exact_command(material.binding.command_bytes),
        )
    assert Function.resolutions == [] and Volume.calls == []


def test_stage_verifier_exception_stops_before_sdk(monkeypatch):
    case, transport, _ = setup(monkeypatch)
    transport._stage_verifier = type(
        "Unavailable", (), {"verify": lambda *args: (_ for _ in ()).throw(RuntimeError())},
    )()
    material = case["material"]
    with pytest.raises(ValueError, match="authentication unavailable"):
        transport.execute_once(
            material.binding, parse_exact_command(material.binding.command_bytes),
        )
    assert Function.resolutions == [] and Volume.calls == []


def test_changed_verified_deployment_stops_before_sdk(monkeypatch):
    case, transport, _ = setup(monkeypatch)
    object.__setattr__(transport._deployment, "evidence_ref", "other-evidence")
    binding = case["envelope"].submit_binding
    with pytest.raises(ValueError, match="binding mismatch"):
        transport.execute_once(binding, parse_exact_command(binding.command_bytes))
    assert Function.resolutions == [] and Function.spawns == [] and Volume.calls == []


def test_cancel_uses_exact_authenticated_target_once_with_termination(monkeypatch):
    case, transport, facade = setup(monkeypatch)
    binding, cancel = cancel_binding(case, "cancel-nonce", "fc-exact", "a" * 64)
    result = transport.execute_once(binding, cancel)
    assert result.provider_ref == "fc-exact"
    assert FunctionCall.resolutions == [("fc-exact", facade.client)]
    assert Call.calls == [{"terminate_containers": True}]


def test_lookup_is_conservatively_indeterminate_without_sdk_access(monkeypatch):
    case, transport, _ = setup(monkeypatch)
    binding = case["envelope"].submit_binding
    command = parse_exact_command(binding.command_bytes)
    before = (len(Function.resolutions), len(Function.spawns), len(Volume.calls))
    assert transport.lookup_once(binding, command).disposition is ObservationDisposition.INDETERMINATE
    assert (len(Function.resolutions), len(Function.spawns), len(Volume.calls)) == before


def test_cancel_timeout_is_indeterminate_and_never_retried(monkeypatch):
    case, transport, _ = setup(monkeypatch); Call.fail_cancel = True
    binding, cancel = cancel_binding(
        case, "cancel-timeout", "fc-ambiguous", "b" * 64,
    )
    result = transport.execute_once(binding, cancel)
    assert result.disposition is ObservationDisposition.INDETERMINATE
    assert len(Call.calls) == 1


def test_cancel_rejects_wrong_sdk_handle_identity_without_cancel(monkeypatch):
    case, transport, _ = setup(monkeypatch); FunctionCall.returned_id = "fc-other"
    binding, cancel = cancel_binding(
        case, "cancel-wrong-handle", "fc-exact", "c" * 64,
    )
    result = transport.execute_once(binding, cancel)
    assert result.disposition is ObservationDisposition.INDETERMINATE
    assert Call.calls == []
