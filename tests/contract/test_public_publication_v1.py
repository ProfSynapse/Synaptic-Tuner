"""Public contract for canonical artifact publication composition."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from synaptic_tuner.api import v1
from synaptic_tuner.api.v1.artifacts_facade import (
    ArtifactDestination,
    ArtifactsAPI,
    DestinationPage,
    PublicationPage,
    PublicationRef,
    PublicationRequest,
    PublicationResult,
    PublicationState,
    PublicationVerification,
)
from synaptic_tuner.api.v1.publication import (
    AuthenticatedDestinationInventoryV1,
    AuthenticatedPublicationReceiptV1,
    AuthenticatedPublicationTombstoneV1,
    DestinationArtifactV1,
    DestinationInventoryV1,
    PublicationCommandV1,
    PublicationOperationsV1,
    PublicationTransitionKernelV1,
)
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact


def _result() -> PublicationResult:
    return PublicationResult(
        "synaptic-publication-result/v1",
        PublicationRef("publication-1", "destination-1"),
        TrainingRunRef("run-1", "project-1"),
        PublicationState.VERIFIED,
        (VerifiedArtifact("model", "a" * 64, 2),),
    )


def _publication_codec_documents():
    artifact = {
        "role": "adapter", "path": "bundle/adapter",
        "sha256": "1" * 64, "size_bytes": 7,
    }
    inventory = {"artifacts": [artifact]}
    authenticated_inventory = {
        "inventory": inventory, "publication_id": "2" * 64,
        "command_digest": "3" * 64, "mutation_id": "4" * 64,
        "ownership_id": "5" * 64, "recorded_at": "2026-08-27T12:00:01Z",
        "authority_ref": "host-authority", "key_ref": "host-key",
        "tag": "6" * 64,
    }
    receipt = {
        "schema_version": "synaptic-publication-receipt/v1",
        "publication_id": "2" * 64, "command_digest": "3" * 64,
        "run": {"run_id": "run-1", "project_ref": "project-1"},
        "source_identity_digest": "7" * 64,
        "destination_ref": "opaque/local-like",
        "destination_identity_digest": "8" * 64,
        "mutation_id": "4" * 64, "claim_digest": "9" * 64,
        "ownership_id": "5" * 64, "inventory": authenticated_inventory,
        "recorded_at": "2026-08-27T12:00:01Z",
        "authority_ref": "host-authority", "key_ref": "host-key",
        "tag": "a" * 64,
    }
    tombstone = {
        "schema_version": "synaptic-publication-tombstone/v1",
        "publication_id": "2" * 64, "mutation_id": "4" * 64,
        "command_digest": "3" * 64, "claim_digest": "9" * 64,
        "destination_ref": "opaque/local-like",
        "destination_identity_digest": "8" * 64,
        "destination_configuration_digest": "b" * 64,
        "destination_policy_digest": "c" * 64,
        "destination_authority_ref": "host-authority",
        "destination_key_ref": "host-key", "fenced_ownership_id": "5" * 64,
        "recovery_permit_id": "d" * 64, "mutation_registry_digest": "e" * 64,
        "checked_at": "2026-08-27T12:00:02Z", "evidence_digest": "f" * 64,
        "authority_ref": "host-authority", "key_ref": "host-key",
        "tag": "0" * 64,
    }
    return (
        (DestinationArtifactV1, artifact),
        (DestinationInventoryV1, inventory),
        (AuthenticatedDestinationInventoryV1, authenticated_inventory),
        (AuthenticatedPublicationReceiptV1, receipt),
        (AuthenticatedPublicationTombstoneV1, tombstone),
    )


def _assert_exact_json_scalars(value) -> None:
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            _assert_exact_json_scalars(item)
    elif type(value) is list:
        for item in value:
            _assert_exact_json_scalars(item)
    else:
        assert type(value) in (str, int, type(None))


def test_root_exports_exact_canonical_publication_identities() -> None:
    assert v1.ArtifactsAPI is ArtifactsAPI
    assert v1.PublicationOperationsV1 is PublicationOperationsV1
    assert v1.PublicationTransitionKernelV1 is PublicationTransitionKernelV1
    assert PublicationOperationsV1.__module__ == "tuner.execution.coordinator_v1.publication"
    for removed in (
        "ArtifactPublicationReceipt", "ArtifactPublisher", "PublishedArtifact",
        "VerifiedArtifactDescriptor", "VerifiedArtifactSource",
    ):
        assert removed not in v1.__all__
        assert not hasattr(v1, removed)
    with pytest.raises(ModuleNotFoundError):
        __import__("synaptic_tuner.api.v1.artifacts")


def test_public_destination_port_exposes_its_exact_command_type() -> None:
    assert PublicationCommandV1.__module__ == "tuner.execution.coordinator_v1.publication"


@pytest.mark.parametrize(
    ("codec_type", "document"),
    _publication_codec_documents(),
    ids=lambda item: item.__name__ if isinstance(item, type) else None,
)
def test_publication_evidence_object_codecs_require_exact_objects_and_scalars(
    codec_type, document,
) -> None:
    parsed = codec_type.from_dict(document)
    _assert_exact_json_scalars(parsed.to_dict())

    class DictSubclass(dict):
        pass

    for hostile in (MappingProxyType(document), DictSubclass(document)):
        with pytest.raises(TypeError, match="exact object"):
            codec_type.from_dict(hostile)


def test_publication_evidence_object_codecs_reject_hostile_keys_without_callbacks() -> None:
    class Field(str):
        calls = 0

        def __hash__(self):
            type(self).calls += 1
            return str.__hash__(self)

        def __eq__(self, other):
            type(self).calls += 1
            raise RuntimeError("secret key callback")

    _, artifact = _publication_codec_documents()[0]
    role = artifact.pop("role")
    dict.__setitem__(artifact, Field("role"), role)
    Field.calls = 0
    with pytest.raises(TypeError, match="field names") as captured:
        DestinationArtifactV1.from_dict(artifact)
    assert captured.value.__cause__ is None
    assert Field.calls == 0

    _, inventory = _publication_codec_documents()[1]
    nested_artifact = inventory["artifacts"][0]
    path = nested_artifact.pop("path")
    dict.__setitem__(nested_artifact, Field("path"), path)
    Field.calls = 0
    with pytest.raises(TypeError, match="field names") as captured:
        DestinationInventoryV1.from_dict(inventory)
    assert captured.value.__cause__ is None
    assert Field.calls == 0

    _, receipt = _publication_codec_documents()[3]
    run = receipt["run"]
    run_id = run.pop("run_id")
    dict.__setitem__(run, Field("run_id"), run_id)
    Field.calls = 0
    with pytest.raises(TypeError, match="field names") as captured:
        AuthenticatedPublicationReceiptV1.from_dict(receipt)
    assert captured.value.__cause__ is None
    assert Field.calls == 0


def test_publication_evidence_object_codecs_reject_hostile_scalars_without_callbacks() -> None:
    class Text(str):
        calls = 0

        def __hash__(self):
            type(self).calls += 1
            raise RuntimeError("secret scalar callback")

        def __eq__(self, other):
            type(self).calls += 1
            raise RuntimeError("secret scalar callback")

        def __str__(self):
            type(self).calls += 1
            raise RuntimeError("secret scalar callback")

    cases = []
    _, artifact = _publication_codec_documents()[0]
    artifact["role"] = Text(artifact["role"])
    cases.append((DestinationArtifactV1, artifact))

    _, inventory = _publication_codec_documents()[1]
    inventory["artifacts"][0]["path"] = Text("bundle/adapter")
    cases.append((DestinationInventoryV1, inventory))

    _, authenticated_inventory = _publication_codec_documents()[2]
    authenticated_inventory["authority_ref"] = Text("host-authority")
    cases.append((AuthenticatedDestinationInventoryV1, authenticated_inventory))

    _, receipt = _publication_codec_documents()[3]
    receipt["run"]["run_id"] = Text("run-1")
    cases.append((AuthenticatedPublicationReceiptV1, receipt))

    _, tombstone = _publication_codec_documents()[4]
    tombstone["checked_at"] = Text("2026-08-27T12:00:02Z")
    cases.append((AuthenticatedPublicationTombstoneV1, tombstone))

    for codec_type, document in cases:
        Text.calls = 0
        with pytest.raises(TypeError, match="exact string") as captured:
            codec_type.from_dict(document)
        assert captured.value.__cause__ is None
        assert Text.calls == 0

    class Number(int):
        calls = 0

        def __lt__(self, other):
            type(self).calls += 1
            raise RuntimeError("secret numeric callback")

    _, artifact = _publication_codec_documents()[0]
    artifact["size_bytes"] = Number(7)
    with pytest.raises(TypeError, match="exact integer") as captured:
        DestinationArtifactV1.from_dict(artifact)
    assert captured.value.__cause__ is None
    assert Number.calls == 0


def test_publication_evidence_object_codecs_reject_nested_container_subclasses() -> None:
    class DictSubclass(dict):
        pass

    class ListSubclass(list):
        pass

    _, inventory = _publication_codec_documents()[1]
    inventory["artifacts"] = ListSubclass(inventory["artifacts"])
    with pytest.raises(ValueError, match="exact array"):
        DestinationInventoryV1.from_dict(inventory)

    _, receipt = _publication_codec_documents()[3]
    receipt["run"] = MappingProxyType(receipt["run"])
    with pytest.raises(TypeError, match="exact object"):
        AuthenticatedPublicationReceiptV1.from_dict(receipt)

    _, receipt = _publication_codec_documents()[3]
    receipt["inventory"] = DictSubclass(receipt["inventory"])
    with pytest.raises(TypeError, match="exact object"):
        AuthenticatedPublicationReceiptV1.from_dict(receipt)


def test_publication_result_parser_requires_exact_builtin_objects() -> None:
    document = _result().to_dict()
    with pytest.raises(TypeError, match="exact object"):
        PublicationResult.from_dict(MappingProxyType(document))  # type: ignore[arg-type]

    class DictSubclass(dict):
        pass

    with pytest.raises(TypeError, match="exact object"):
        PublicationResult.from_dict(DictSubclass(document))


def test_publication_result_parser_rejects_hostile_keys_before_callbacks() -> None:
    class Field(str):
        calls = 0

        def __hash__(self):
            type(self).calls += 1
            if type(self).calls > 1:
                raise RuntimeError("secret callback")
            return str.__hash__(self)

    document = _result().to_dict()
    state = document.pop("state")
    dict.__setitem__(document, Field("state"), state)
    with pytest.raises(TypeError, match="field names") as captured:
        PublicationResult.from_dict(document)
    assert captured.value.__cause__ is None
    assert Field.calls == 1


def test_artifacts_api_reconstructs_and_binds_callback_results() -> None:
    result = _result()
    request = PublicationRequest(TrainingRunRef("run-1", "project-1"), "destination-1")

    class Operations:
        def destinations(self):
            return DestinationPage((ArtifactDestination("destination-1", "Local"),))

        def publications(self, destination_ref):
            return PublicationPage((result,))

        def publish(self, supplied):
            assert supplied is not request
            return result

        def verify(self, supplied):
            return PublicationVerification(
                supplied, True, "2026-08-30T12:00:00Z"
            )

    api = ArtifactsAPI(Operations())
    assert api.destinations() == DestinationPage((ArtifactDestination("destination-1", "Local"),))
    assert api.publications("destination-1") == PublicationPage((result,))
    assert api.publish(request) == result
    assert api.verify(result.publication).verified is True


def test_artifacts_api_rejects_callback_destination_drift() -> None:
    result = _result()

    class Operations:
        def publish(self, request):
            return result

    request = PublicationRequest(TrainingRunRef("run-1", "project-1"), "other")
    with pytest.raises(ValueError, match="bind"):
        ArtifactsAPI(Operations()).publish(request)


@pytest.mark.parametrize("verb", ["publish", "verify"])
@pytest.mark.parametrize("raises", [False, True])
def test_artifacts_api_rejects_presented_input_mutation_on_return_and_raise(verb, raises) -> None:
    result = _result()
    request = PublicationRequest(result.run, result.publication.destination_ref)

    class Operations:
        def publish(self, supplied):
            object.__setattr__(supplied.run, "run_id", "changed")
            if raises:
                raise RuntimeError("collaborator detail")
            return result

        def verify(self, supplied):
            object.__setattr__(supplied, "publication_id", "changed")
            if raises:
                raise RuntimeError("collaborator detail")
            return PublicationVerification(supplied, True, "2026-08-30T12:00:00Z")

    api = ArtifactsAPI(Operations())
    invocation = api.publish if verb == "publish" else api.verify
    value = request if verb == "publish" else result.publication
    with pytest.raises(ValueError, match="input changed") as captured:
        invocation(value)
    if raises:
        pending = [captured.value]
        seen = set()
        while pending:
            error = pending.pop()
            if id(error) in seen:
                continue
            seen.add(id(error))
            assert type(error) is not RuntimeError
            assert "collaborator detail" not in str(error)
            pending.extend(item for item in (error.__cause__, error.__context__) if item is not None)


def test_publication_result_requires_exact_run_and_rejects_runless_v1_payload() -> None:
    result = _result()
    assert PublicationResult.from_dict(result.to_dict()).run == result.run
    runless = result.to_dict()
    del runless["run"]
    with pytest.raises(ValueError, match="missing fields: run"):
        PublicationResult.from_dict(runless)
    wrong = PublicationResult(
        result.schema_version, result.publication,
        TrainingRunRef("other", result.run.project_ref), result.state, result.artifacts,
    )

    class Operations:
        def publish(self, supplied):
            return wrong

    with pytest.raises(ValueError, match="bind"):
        ArtifactsAPI(Operations()).publish(PublicationRequest(result.run, "destination-1"))


def test_publication_contracts_detach_nested_run_and_reference_identities() -> None:
    run = TrainingRunRef("run-1", "project-1")
    publication = PublicationRef("publication-1", "destination-1")
    request = PublicationRequest(run, "destination-1")
    result = PublicationResult(
        "synaptic-publication-result/v1", publication, run,
        PublicationState.COMMITTED,
    )
    verification = PublicationVerification(
        publication, True, "2026-08-30T12:00:00Z",
    )
    assert request.run is not run
    assert result.run is not run
    assert result.publication is not publication
    assert verification.publication is not publication


def test_legacy_training_start_publication_surface_is_absent() -> None:
    from synaptic_tuner.api.v1.training import TrainingAPI, TrainingOperations
    from synaptic_tuner.api.v1.host import HostPorts

    assert "publish" not in TrainingAPI.__dict__
    assert "publish" not in TrainingOperations.__dict__
    assert "artifact_publisher" not in HostPorts.__dataclass_fields__


def test_secondary_host_v1_publication_protocols_are_absent() -> None:
    import synaptic_tuner.host.v1 as host_v1
    from synaptic_tuner.host.v1 import ports

    for name in ("ArtifactSource", "ArtifactPublisher"):
        assert name not in host_v1.__all__
        assert name not in ports.__all__
        assert not hasattr(host_v1, name)
        assert not hasattr(ports, name)
