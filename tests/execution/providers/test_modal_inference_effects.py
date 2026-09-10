"""Provider-free conformance for authenticated Modal chat effects."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tuner.execution.foundation_v2.canonical import FoundationError
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.executors import (
    ExecutionResolutionRequestV2,
    ReconciliationResolutionRequestV2,
)
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.reconciliation import ReconciliationTargetV1
from tuner.execution.providers.modal.coordinator_effects import ModalEffectOutcome
from tuner.execution.providers.modal.inference_commands import (
    ModalInferenceCommandBinding,
)
from tuner.execution.providers.modal.inference_effects import (
    ModalChatEffectExecutor,
    ModalChatExecutorResolver,
    ModalChatReconciliationAdapter,
    ModalChatReconciliationResolver,
)
from tests.execution.providers.test_modal_inference_commands import (
    Authority,
    Catalog,
    _bindings,
)


class Transport:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def execute_once(self, binding, command):
        self.calls.append(("execute", binding, command))
        return self.outcome

    def lookup_once(self, binding, command):
        self.calls.append(("lookup", binding, command))
        return self.outcome


def _parts(monkeypatch, index=0, outcome=None):
    binding = _bindings(monkeypatch)[index]
    command = binding.command_digest
    parsed = parse_exact_command(binding.command_bytes)
    transport = Transport(
        outcome or ModalEffectOutcome(ObservationDisposition.INDETERMINATE)
    )
    catalog = Catalog()
    catalog.values[command] = binding
    authority = Authority([binding.canonical_bytes])
    kwargs = dict(
        profile_ref=parsed.preparation.provider.profile_ref,
        account_ref=parsed.preparation.scope.account_ref,
        namespace_ref=parsed.preparation.scope.namespace_ref,
        implementation_version="v1",
        catalog=catalog,
        authority=authority,
        transport=transport,
    )
    request = ExecutionResolutionRequestV2(
        parsed.digest,
        parsed.executor.digest,
        parsed.preparation.provider.provider_id,
        parsed.preparation.provider.profile_ref,
        parsed.preparation.scope.account_ref,
        parsed.preparation.scope.namespace_ref,
        parsed.operation.effect.kind.value,
        parsed.payload.payload_kind,
        parsed.payload.input_digest,
    )
    return binding, parsed, transport, kwargs, request


@pytest.mark.parametrize("index,provider_ref", ((0, "stage-ref"), (1, "job-ref")))
def test_stage_and_submit_execute_once_return_exact_typed_observation(
    monkeypatch, index, provider_ref
):
    outcome = ModalEffectOutcome(ObservationDisposition.FOUND, provider_ref)
    binding, command, transport, kwargs, request = _parts(monkeypatch, index, outcome)
    observation = ModalChatEffectExecutor(**kwargs).execute_once(
        command.payload, request
    )
    assert observation.command_digest == command.digest
    assert observation.disposition is ObservationDisposition.FOUND
    if index == 0:
        assert observation.stage_ref.stage_ref == provider_ref
    else:
        assert observation.provider_run.provider_job_ref == provider_ref
    assert len(transport.calls) == 1
    assert transport.calls[0][1] is not binding
    assert transport.calls[0][1].canonical_bytes == binding.canonical_bytes
    assert transport.calls[0][2].canonical_bytes == command.canonical_bytes


@pytest.mark.parametrize(
    "field", tuple(ExecutionResolutionRequestV2.__dataclass_fields__)
)
def test_every_execution_request_field_is_bound_before_transport(monkeypatch, field):
    _, command, transport, kwargs, request = _parts(monkeypatch)
    value = "0" * 64 if field.endswith("digest") else "foreign"
    with pytest.raises(FoundationError):
        ModalChatEffectExecutor(**kwargs).execute_once(
            command.payload, replace(request, **{field: value})
        )
    assert transport.calls == []


@pytest.mark.parametrize("mode", ("false", "error"))
def test_authentication_denial_is_closed_and_precedes_transport(monkeypatch, mode):
    _, command, transport, kwargs, request = _parts(monkeypatch)

    class BadAuthority:
        def authenticate(self, binding):
            if mode == "error":
                raise RuntimeError("credential-shaped-private")
            return False

    kwargs["authority"] = BadAuthority()
    with pytest.raises(FoundationError) as caught:
        ModalChatEffectExecutor(**kwargs).execute_once(command.payload, request)
    assert caught.value.__cause__ is None
    assert transport.calls == []


@pytest.mark.parametrize("control", (KeyboardInterrupt, SystemExit))
@pytest.mark.parametrize("callback", ("catalog", "authority", "transport"))
def test_control_flow_from_collaborators_is_preserved(monkeypatch, callback, control):
    _, command, transport, kwargs, request = _parts(monkeypatch)
    if callback == "catalog":

        class ControlCatalog(Catalog):
            def resolve(self, key):
                raise control()

        kwargs["catalog"] = ControlCatalog()
    elif callback == "authority":

        class ControlAuthority(Authority):
            def authenticate(self, binding):
                raise control()

        kwargs["authority"] = ControlAuthority([])
    else:

        class ControlTransport(Transport):
            def execute_once(self, binding, command):
                raise control()

        kwargs["transport"] = ControlTransport(transport.outcome)
    with pytest.raises(control):
        ModalChatEffectExecutor(**kwargs).execute_once(command.payload, request)


def test_indeterminate_is_preserved_without_retry(monkeypatch):
    _, command, transport, kwargs, request = _parts(monkeypatch)
    result = ModalChatEffectExecutor(**kwargs).execute_once(command.payload, request)
    assert result.disposition is ObservationDisposition.INDETERMINATE
    assert len(transport.calls) == 1


def test_reconciliation_lookup_binds_target_and_preserves_unknown(monkeypatch):
    binding, command, transport, kwargs, _ = _parts(monkeypatch)
    adapter = ModalChatReconciliationAdapter(**kwargs)
    resolution = ReconciliationResolutionRequestV2(
        command.digest,
        adapter.descriptor.digest,
        "modal",
        kwargs["profile_ref"],
        kwargs["account_ref"],
        kwargs["namespace_ref"],
    )
    target = ReconciliationTargetV1(
        binding.command_bytes,
        command.digest,
        command.operation.effect.effect_id,
        "owner-a",
        1,
        7,
        1,
        resolution.digest,
    )
    result = adapter.lookup(target, command.preparation)
    assert result.disposition is ObservationDisposition.INDETERMINATE
    assert result.result_epoch == 7 and result.resolution_digest == resolution.digest
    assert len(transport.calls) == 1


@pytest.mark.parametrize("field", ("command_bytes", "command_digest", "effect_id"))
def test_every_reconciliation_target_field_is_bound_before_transport(
    monkeypatch, field
):
    binding, command, transport, kwargs, _ = _parts(monkeypatch)
    adapter = ModalChatReconciliationAdapter(**kwargs)
    resolution = ReconciliationResolutionRequestV2(
        command.digest,
        adapter.descriptor.digest,
        "modal",
        kwargs["profile_ref"],
        kwargs["account_ref"],
        kwargs["namespace_ref"],
    )
    target = ReconciliationTargetV1(
        binding.command_bytes,
        command.digest,
        command.operation.effect.effect_id,
        "owner-a",
        1,
        7,
        1,
        resolution.digest,
    )
    if field == "command_bytes":
        changed = replace(target, command_bytes=target.command_bytes + b" ")
    elif field in ("generation", "ownership_epoch", "claimed_at_epoch"):
        changed = replace(target, **{field: getattr(target, field) + 1})
    else:
        changed = replace(
            target, **{field: "0" * 64 if field.endswith("digest") else "other"}
        )
    with pytest.raises(FoundationError):
        adapter.lookup(changed, command.preparation)
    assert transport.calls == []


def test_reconciliation_context_is_preserved_exactly(monkeypatch):
    binding, command, transport, kwargs, _ = _parts(monkeypatch)
    adapter = ModalChatReconciliationAdapter(**kwargs)
    resolution = ReconciliationResolutionRequestV2(
        command.digest,
        adapter.descriptor.digest,
        "modal",
        kwargs["profile_ref"],
        kwargs["account_ref"],
        kwargs["namespace_ref"],
    )
    target = ReconciliationTargetV1(
        binding.command_bytes,
        command.digest,
        command.operation.effect.effect_id,
        "owner-a",
        2,
        7,
        5,
        resolution.digest,
    )
    result = adapter.lookup(target, command.preparation)
    assert result.result_epoch == target.ownership_epoch
    assert result.resolution_digest == target.resolution_digest
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "target", ("request", "payload", "binding", "command", "executor")
)
def test_transport_callback_mutation_is_detected(monkeypatch, target):
    binding, command, _, kwargs, request = _parts(monkeypatch)
    payload = command.payload

    class MutatingTransport(Transport):
        def execute_once(self, owned_binding, owned_command):
            self.calls.append(("execute", owned_binding, owned_command))
            if target == "request":
                object.__setattr__(request, "provider_id", "other")
            elif target == "payload":
                object.__setattr__(payload, "_raw", b"changed")
            elif target == "binding":
                object.__setattr__(owned_binding, "_command_bytes", b"changed")
            elif target == "command":
                object.__setattr__(owned_command, "_raw", b"changed")
            else:
                kwargs["executor"].profile_ref = "other"
            return self.outcome

    transport = MutatingTransport(
        ModalEffectOutcome(ObservationDisposition.INDETERMINATE)
    )
    kwargs["transport"] = transport
    executor = ModalChatEffectExecutor(**kwargs)
    kwargs["executor"] = executor
    with pytest.raises(FoundationError):
        executor.execute_once(payload, request)
    assert len(transport.calls) == 1


def test_resolvers_mint_only_exact_bound_executor_and_adapter(monkeypatch):
    _, _, _, kwargs, request = _parts(monkeypatch)
    executor = ModalChatEffectExecutor(**kwargs)
    resolved = ModalChatExecutorResolver(executor).resolve(request)
    assert resolved.executor is executor
    adapter = ModalChatReconciliationAdapter(**kwargs)
    reconciliation = ReconciliationResolutionRequestV2(
        request.command_digest,
        adapter.descriptor.digest,
        request.provider_id,
        request.profile_ref,
        request.account_ref,
        request.namespace_ref,
    )
    assert (
        ModalChatReconciliationResolver(adapter).resolve(reconciliation).adapter
        is adapter
    )


def test_wrong_exact_input_types_do_not_evaluate_duck_properties(monkeypatch):
    reads = []

    class Duck:
        @property
        def command_digest(self):
            reads.append("read")
            raise AssertionError

    _, _, transport, kwargs, _ = _parts(monkeypatch)
    with pytest.raises((TypeError, FoundationError)):
        ModalChatEffectExecutor(**kwargs).execute_once(Duck(), Duck())
    assert reads == [] and transport.calls == []


def test_lookup_duck_preparation_equality_is_not_evaluated(monkeypatch):
    reads = []

    class Duck:
        def __eq__(self, other):
            reads.append("eq")
            raise AssertionError

    binding, command, transport, kwargs, _ = _parts(monkeypatch)
    adapter = ModalChatReconciliationAdapter(**kwargs)
    resolution = ReconciliationResolutionRequestV2(
        command.digest,
        adapter.descriptor.digest,
        "modal",
        kwargs["profile_ref"],
        kwargs["account_ref"],
        kwargs["namespace_ref"],
    )
    target = ReconciliationTargetV1(
        binding.command_bytes,
        command.digest,
        command.operation.effect.effect_id,
        "owner-a",
        1,
        1,
        1,
        resolution.digest,
    )
    with pytest.raises(FoundationError):
        adapter.lookup(target, Duck())
    assert reads == [] and transport.calls == []


def test_catalog_callback_cannot_change_wrong_payload_into_expected(monkeypatch):
    stage, submit, _ = _bindings(monkeypatch)
    wrong_payload = parse_exact_command(submit.command_bytes).payload
    expected_payload = parse_exact_command(stage.command_bytes).payload
    transport = Transport(ModalEffectOutcome(ObservationDisposition.INDETERMINATE))

    class MutatingCatalog(Catalog):
        def resolve(self, key):
            object.__setattr__(wrong_payload, "_raw", expected_payload.canonical_bytes)
            return super().resolve(key)

    catalog = MutatingCatalog()
    catalog.values[stage.command_digest] = stage
    command = parse_exact_command(stage.command_bytes)
    kwargs = dict(
        profile_ref=command.preparation.provider.profile_ref,
        account_ref=command.preparation.scope.account_ref,
        namespace_ref=command.preparation.scope.namespace_ref,
        implementation_version="v1",
        catalog=catalog,
        authority=Authority([stage.canonical_bytes]),
        transport=transport,
    )
    request = ExecutionResolutionRequestV2(
        command.digest,
        command.executor.digest,
        "modal",
        kwargs["profile_ref"],
        kwargs["account_ref"],
        kwargs["namespace_ref"],
        "stage",
        "stage-payload/v2",
        command.payload.input_digest,
    )
    with pytest.raises(FoundationError):
        ModalChatEffectExecutor(**kwargs).execute_once(wrong_payload, request)
    assert transport.calls == []


def test_invalid_transport_result_is_closed_without_retry(monkeypatch):
    _, command, transport, kwargs, request = _parts(monkeypatch)
    transport.outcome = object()
    with pytest.raises(FoundationError):
        ModalChatEffectExecutor(**kwargs).execute_once(command.payload, request)
    assert len(transport.calls) == 1


@pytest.mark.parametrize("missing", ("catalog", "authority", "transport"))
def test_constructor_requires_static_collaborator_methods(monkeypatch, missing):
    _, _, _, kwargs, _ = _parts(monkeypatch)
    kwargs[missing] = object()
    with pytest.raises(TypeError):
        ModalChatEffectExecutor(**kwargs)


def test_lookup_rejects_executor_version_mismatch_before_transport(monkeypatch):
    binding, command, transport, kwargs, _ = _parts(monkeypatch)
    kwargs["implementation_version"] = "v2"
    adapter = ModalChatReconciliationAdapter(**kwargs)
    resolution = ReconciliationResolutionRequestV2(
        command.digest,
        adapter.descriptor.digest,
        "modal",
        kwargs["profile_ref"],
        kwargs["account_ref"],
        kwargs["namespace_ref"],
    )
    target = ReconciliationTargetV1(
        binding.command_bytes,
        command.digest,
        command.operation.effect.effect_id,
        "owner-a",
        1,
        1,
        1,
        resolution.digest,
    )
    with pytest.raises(FoundationError):
        adapter.lookup(target, command.preparation)
    assert transport.calls == []


def test_resolver_rejects_mutated_constructor_state(monkeypatch):
    _, _, _, kwargs, request = _parts(monkeypatch)
    executor = ModalChatEffectExecutor(**kwargs)
    resolver = ModalChatExecutorResolver(executor)
    executor.profile_ref = "other"
    with pytest.raises(FoundationError):
        resolver.resolve(request)


def test_catalog_callback_cannot_swap_transport_before_dispatch(monkeypatch):
    binding, command, transport, kwargs, request = _parts(monkeypatch)
    replacement = Transport(ModalEffectOutcome(ObservationDisposition.INDETERMINATE))
    original = kwargs["catalog"]
    callback_calls = []
    executor = None

    class SwappingCatalog(Catalog):
        def resolve(self, key):
            callback_calls.append(key)
            executor._transport = replacement
            return original.resolve(key)

    swapped = SwappingCatalog()
    swapped.values.update(original.values)
    kwargs["catalog"] = swapped
    executor = ModalChatEffectExecutor(**kwargs)
    with pytest.raises(FoundationError):
        executor.execute_once(command.payload, request)
    assert callback_calls == [command.digest]
    assert transport.calls == [] and replacement.calls == []


def test_malformed_request_primitive_stops_before_catalog(monkeypatch):
    _, command, transport, kwargs, request = _parts(monkeypatch)
    malformed = replace(request, input_digest=object())
    with pytest.raises(FoundationError):
        ModalChatEffectExecutor(**kwargs).execute_once(command.payload, malformed)
    assert kwargs["catalog"].calls == [] and transport.calls == []


@pytest.mark.parametrize("field", tuple(ReconciliationTargetV1.__dataclass_fields__))
def test_lookup_callback_cannot_mutate_any_target_field(monkeypatch, field):
    binding, command, _, kwargs, _ = _parts(monkeypatch)
    adapter = None
    resolution = ReconciliationResolutionRequestV2(
        command.digest,
        "0" * 64,
        "modal",
        kwargs["profile_ref"],
        kwargs["account_ref"],
        kwargs["namespace_ref"],
    )
    target = ReconciliationTargetV1(
        binding.command_bytes,
        command.digest,
        command.operation.effect.effect_id,
        "owner-a",
        1,
        1,
        1,
        resolution.digest,
    )

    class MutatingLookup(Transport):
        def lookup_once(self, owned_binding, owned_command):
            self.calls.append(("lookup", owned_binding, owned_command))
            value = (
                target.command_bytes + b" "
                if field == "command_bytes"
                else (
                    getattr(target, field) + 1
                    if field in ("generation", "ownership_epoch", "claimed_at_epoch")
                    else "0" * 64 if field.endswith("digest") else "other"
                )
            )
            object.__setattr__(target, field, value)
            return self.outcome

    transport = MutatingLookup(ModalEffectOutcome(ObservationDisposition.INDETERMINATE))
    kwargs["transport"] = transport
    adapter = ModalChatReconciliationAdapter(**kwargs)
    object.__setattr__(
        target,
        "resolution_digest",
        ReconciliationResolutionRequestV2(
            command.digest,
            adapter.descriptor.digest,
            "modal",
            kwargs["profile_ref"],
            kwargs["account_ref"],
            kwargs["namespace_ref"],
        ).digest,
    )
    with pytest.raises(FoundationError):
        adapter.lookup(target, command.preparation)
    assert len(transport.calls) == 1


def test_malformed_target_epoch_stops_before_catalog(monkeypatch):
    binding, command, transport, kwargs, _ = _parts(monkeypatch)
    adapter = ModalChatReconciliationAdapter(**kwargs)
    resolution = ReconciliationResolutionRequestV2(
        command.digest,
        adapter.descriptor.digest,
        "modal",
        kwargs["profile_ref"],
        kwargs["account_ref"],
        kwargs["namespace_ref"],
    )
    target = ReconciliationTargetV1(
        binding.command_bytes,
        command.digest,
        command.operation.effect.effect_id,
        "owner-a",
        1,
        1,
        1,
        resolution.digest,
    )
    object.__setattr__(target, "ownership_epoch", True)
    with pytest.raises(FoundationError):
        adapter.lookup(target, command.preparation)
    assert kwargs["catalog"].calls == [] and transport.calls == []
