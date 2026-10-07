"""Host semantic gate tests for Modal submit dispatch preparation."""

from types import SimpleNamespace

import pytest

from tests.execution.providers.test_modal_coordinator_launch import launch_case
from tests.execution.providers.modal_coordinator_fixtures import real_launch_bundle_case
from tuner.execution.foundation_v2.canonical import canonical_bytes, parse_canonical_object
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.providers.modal.coordinator_binding import ModalCommandBinding
from tuner.execution.providers.modal.coordinator_dispatch import ModalWorkerDispatch
from tuner.execution.providers.modal.coordinator_submit_preparation import (
    prepare_modal_submit_dispatch,
)
from tuner.training.methods.sft import SFTRecipe
from tuner.training.recipes import RecipeRegistry


def recipes():
    value = RecipeRegistry()
    value.register(SFTRecipe())
    return value


def arguments(case):
    harness, _, _, _, _, authority, authentic, _, envelope = case
    return dict(
        foundation_authenticator=harness.authenticator,
        assessment_authenticator=harness.foundation,
        binding_authority=authority, stage_verifier=authentic,
        launch_verifier=authentic, recipes=recipes(),
    ), envelope


def test_authenticated_launch_still_rejects_opaque_nonsemantic_bundle(monkeypatch):
    options, envelope = arguments(launch_case(monkeypatch))
    with pytest.raises(ValueError):
        prepare_modal_submit_dispatch(envelope, **options)


def test_semantic_bundle_gate_precedes_dispatch_build(monkeypatch):
    options, envelope = arguments(launch_case(monkeypatch))
    import tuner.execution.providers.modal.coordinator_submit_preparation as module
    observed = []

    def parsed(transport, *, binding, recipes):
        observed.append("parse")
        return SimpleNamespace(transport_bytes=transport, binding=binding)

    original = module.build_modal_worker_dispatch

    def built(*args, **kwargs):
        observed.append("build")
        return original(*args, **kwargs)

    monkeypatch.setattr(module.ModalCoordinatorBundle, "parse_transport", parsed)
    monkeypatch.setattr(module, "build_modal_worker_dispatch", built)
    result = prepare_modal_submit_dispatch(envelope, **options)
    assert type(result) is ModalWorkerDispatch
    assert observed == ["parse", "build"]


def test_real_semantic_bundle_reaches_dispatch_and_reopens_at_wire(monkeypatch):
    case = real_launch_bundle_case(monkeypatch)
    envelope = case["envelope"]
    harness = case["harness"]
    dispatch = prepare_modal_submit_dispatch(
        envelope, foundation_authenticator=harness.authenticator,
        assessment_authenticator=harness.foundation,
        binding_authority=case["authority"],
        stage_verifier=case["authenticator"],
        launch_verifier=case["authenticator"], recipes=case["recipes"],
    )
    parsed = __import__(
        "tuner.execution.providers.modal.coordinator_dispatch",
        fromlist=["parse_modal_worker_dispatch"],
    ).parse_modal_worker_dispatch(dispatch.canonical_bytes)
    admitted = __import__(
        "tuner.execution.providers.modal.coordinator_wire",
        fromlist=["admit_modal_launch_wire"],
    ).admit_modal_launch_wire(
        parsed.launch_claim, parsed.launch_claim_tag,
        case["material"].claim, case["material"].claim_tag,
        case["material"].bundle, expectation=parsed.expectation,
        verifier=case["authenticator"],
    )
    reopened = __import__(
        "tuner.execution.providers.modal.coordinator_bundle",
        fromlist=["ModalCoordinatorBundle"],
    ).ModalCoordinatorBundle.parse_transport(
        case["material"].bundle,
        binding=case["bundle"].binding, recipes=case["recipes"],
    )
    assert parsed.expectation.submit_command_bytes == envelope.submit_binding.command_bytes
    assert admitted.submit_command_bytes == envelope.submit_binding.command_bytes
    assert reopened == case["bundle"]


def test_launch_authentication_failure_stops_before_bundle_parse(monkeypatch):
    options, envelope = arguments(launch_case(monkeypatch))
    import tuner.execution.providers.modal.coordinator_submit_preparation as module
    observed = []
    monkeypatch.setattr(module.ModalCoordinatorBundle, "parse_transport",
                        lambda *a, **k: observed.append("parse"))
    options["launch_verifier"] = type("Reject", (), {"verify": lambda *args: False})()
    with pytest.raises(ValueError):
        prepare_modal_submit_dispatch(envelope, **options)
    assert observed == []


def test_bundle_binding_is_reconstructed_from_admitted_stage_claim(monkeypatch):
    options, envelope = arguments(launch_case(monkeypatch))
    import tuner.execution.providers.modal.coordinator_submit_preparation as module
    captured = {}

    def parsed(transport, *, binding, recipes):
        captured["binding"] = binding
        return SimpleNamespace(transport_bytes=transport, binding=binding)

    monkeypatch.setattr(module.ModalCoordinatorBundle, "parse_transport", parsed)
    prepare_modal_submit_dispatch(envelope, **options)
    stage = parse_exact_command(canonical_bytes(
        parse_canonical_object(envelope.stage_material.claim, name="claim")["command"]
    ))
    assert captured["binding"] == ModalCommandBinding(
        stage.canonical_bytes, envelope.submit_binding.preparation_snapshot,
        envelope.submit_binding.deployment_bytes,
    )


def test_exact_recipe_registry_is_required_before_host_admission(monkeypatch):
    options, envelope = arguments(launch_case(monkeypatch))
    options["recipes"] = object()
    with pytest.raises(TypeError, match="exact recipe registry"):
        prepare_modal_submit_dispatch(envelope, **options)
