from __future__ import annotations

import copy
import io
import json
import os
import pickle
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

from synaptic_tuner.api.v1.training_facade import TrainingAPI, TrainingRequest
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_sources import (
    LocalTrainingInputPathV1,
    OneUseTrainingInputUploadV1,
    PreparedTrainingInputIdentity,
    PreparedTrainingInputResultV1,
    PreparedTrainingInputV1,
    TrainingPreparationConfigV1,
)
from tests.contract.test_public_training_input_v1 import _document
from tests.dataset_prep.test_dataset_prep_v1 import _bundle, _config
from tuner.dataset_prep import (
    DatasetPublicationUncertainV1,
    DatasetPrepValidationError,
    LocalPreparedTrainingInputSource,
    ROW_SCHEMA_VERSION,
    prepare_dataset_v1,
)
from tuner.training.input_preparation import (
    DATASET_PREP_NORMALIZER_V1,
    DatasetPrepNormalizerConfigV1,
    DatasetPrepTrainingInputNormalizerV1,
    DatasetPreparedInputFormatVerifierV1,
    NormalizedTrainingInputV1,
    PrivatePreparedRootAttestationV1,
    TrainingInputPreparationServiceV1,
    default_dataset_format_verifiers_v1,
)


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _path_free_config(value) -> str:
    document = {
        "schema_version": value.schema_version,
        "source": {
            "bundle_path": str(value.source_bundle_path),
            "expected_bundle_digest": value.expected_bundle_digest,
        },
        "format": value.format,
        "projection": value.projection.to_dict(),
        "ordering": value.ordering.to_dict(),
        "split": value.split.to_dict(),
    }
    document["source"] = {"expected_bundle_digest": value.expected_bundle_digest}
    return _canonical(document)


def _preparation_config(dataset_config) -> TrainingPreparationConfigV1:
    return TrainingPreparationConfigV1(
        request_id="request-a",
        project_ref="project-a",
        canonical_training_json=TrainingInputV1.from_dict(_document()).canonical_json(),
        normalizer_config=DatasetPrepNormalizerConfigV1(_path_free_config(dataset_config)),
    )


def _stable_identity(path: Path) -> tuple[int, int, int, int, int]:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or path.resolve(strict=True) != path.absolute():
        raise ValueError("test root is not a stable plain directory")
    return info.st_dev, info.st_ino, info.st_mode, 0, 0


class _ExplicitTestRootAuthority:
    """Test-only stand-in for an OS-specific private-root attestor."""

    def attest(self, path: Path) -> PrivatePreparedRootAttestationV1:
        path.mkdir(mode=0o700, exist_ok=True)
        return PrivatePreparedRootAttestationV1(path, _stable_identity(path))

    def verify(self, attestation: PrivatePreparedRootAttestationV1) -> None:
        if _stable_identity(attestation.path) != attestation.identity:
            raise ValueError("private root identity changed")


def _service(tmp_path: Path, bundle_path: Path) -> TrainingInputPreparationServiceV1:
    return TrainingInputPreparationServiceV1(
        prepared_root=(tmp_path / "private-prepared").resolve(),
        normalizers={
            DATASET_PREP_NORMALIZER_V1: DatasetPrepTrainingInputNormalizerV1(
                allowed_source_roots=(bundle_path.parent.resolve(),)
            )
        },
        format_verifiers=default_dataset_format_verifiers_v1(),
        root_authority=_ExplicitTestRootAuthority(),
    )


def _prepare_reconciled(service, source, config):
    try:
        return service.prepare(source, config)
    except DatasetPublicationUncertainV1:
        return service.prepare(source, config)


def _prepared_dataset(tmp_path: Path):
    structure, bundle = _bundle(tmp_path / "source")
    config = _config(bundle, structure)
    try:
        prepared = prepare_dataset_v1(config, tmp_path / "private-prepared")
    except DatasetPublicationUncertainV1:
        prepared = prepare_dataset_v1(config, tmp_path / "private-prepared")
    return bundle, config, prepared


def test_path_preparation_verifies_bytes_before_returning_safe_canonical_value(tmp_path):
    structure, bundle = _bundle(tmp_path / "source")
    dataset_config = _config(bundle, structure)
    result = _prepare_reconciled(
        _service(tmp_path, bundle.path),
        LocalTrainingInputPathV1(bundle.path.resolve()),
        _preparation_config(dataset_config),
    )

    assert type(result) is PreparedTrainingInputResultV1
    identity = result.prepared.identity
    request = TrainingInputV1.from_json(result.prepared.request.canonical_json)
    assert request.dataset.ref == identity.ref
    assert identity.ref == f"prepared://sha256/{identity.revision}"
    assert identity.execution_binding_fields() == {
        "prepared_input_ref": identity.ref,
        "prepared_input_revision": identity.revision,
        "prepared_input_content_digest": identity.content_digest,
        "prepared_input_size_bytes": identity.size_bytes,
        "prepared_input_format": identity.format,
    }
    lease = result.retained_source.open_lease()
    stream = lease.take_stream()
    assert len(stream.read()) == identity.size_bytes
    lease.close()
    public = _canonical(result.prepared.to_dict())
    assert str(bundle.path) not in public
    assert "dataset.jsonl" not in public
    assert "Alpha" not in public
    for value in (result, result.retained_source):
        with pytest.raises(TypeError):
            pickle.dumps(value)
        with pytest.raises(TypeError):
            copy.copy(value)


def test_training_api_prepare_is_the_provider_neutral_delegation(tmp_path):
    structure, bundle = _bundle(tmp_path / "source")
    service = _service(tmp_path, bundle.path)

    class Operations:
        def prepare(self, source, config):
            return service.prepare(source, config)

    class Clock:
        def now(self):
            return "2026-09-22T12:00:00Z"

    result = _prepare_reconciled(
        TrainingAPI(Operations(), clock=Clock()),
        LocalTrainingInputPathV1(bundle.path.resolve()),
        _preparation_config(_config(bundle, structure)),
    )
    assert result.prepared.identity == result.retained_source.identity


def test_closed_dataset_config_rejects_locators_duplicates_and_nonfinite_values(tmp_path):
    structure, bundle = _bundle(tmp_path / "source")
    document = json.loads(_path_free_config(_config(bundle, structure)))
    document["source"]["bundle_path"] = str(bundle.path)
    with pytest.raises(ValueError, match="path-free source"):
        DatasetPrepNormalizerConfigV1(_canonical(document))
    with pytest.raises(ValueError, match="duplicate"):
        DatasetPrepNormalizerConfigV1('{"source":{},"source":{}}')
    with pytest.raises(ValueError, match="non-finite"):
        DatasetPrepNormalizerConfigV1('{"source":{"expected_bundle_digest":NaN}}')


def test_prepare_fails_closed_without_private_root_authority(tmp_path):
    structure, bundle = _bundle(tmp_path / "source")
    service = TrainingInputPreparationServiceV1(
        prepared_root=(tmp_path / "private").resolve(),
        normalizers={
            DATASET_PREP_NORMALIZER_V1: DatasetPrepTrainingInputNormalizerV1(
                allowed_source_roots=(bundle.path.parent.resolve(),)
            )
        },
        format_verifiers=default_dataset_format_verifiers_v1(),
    )
    with pytest.raises(ValueError, match="authority is required"):
        service.prepare(
            LocalTrainingInputPathV1(bundle.path.resolve()),
            _preparation_config(_config(bundle, structure)),
        )


def test_allowed_source_root_replacement_is_rejected(tmp_path):
    structure, bundle = _bundle(tmp_path / "admitted")
    service = _service(tmp_path, bundle.path)
    admitted = bundle.path.parent.resolve()
    displaced = admitted.with_name("admitted-displaced")
    admitted.rename(displaced)
    admitted.mkdir()
    with pytest.raises(ValueError, match="root identity changed"):
        service.prepare(
            LocalTrainingInputPathV1(bundle.path.resolve()),
            _preparation_config(_config(bundle, structure)),
        )


def test_hardlinked_source_member_is_rejected(tmp_path):
    structure, bundle = _bundle(tmp_path / "source")
    linked = tmp_path / "manifest-hardlink.json"
    try:
        os.link(bundle.path / "manifest.json", linked)
    except OSError as error:
        pytest.skip(f"hardlinks unavailable: {error}")
    with pytest.raises(ValueError, match="hardlinked"):
        _service(tmp_path, bundle.path).prepare(
            LocalTrainingInputPathV1(bundle.path.resolve()),
            _preparation_config(_config(bundle, structure)),
        )


@dataclass(frozen=True)
class _CustomConfig:
    normalizer_ref: str


def _custom_request(normalizer_ref: str) -> TrainingPreparationConfigV1:
    return TrainingPreparationConfigV1(
        "request-a", "project-a",
        TrainingInputV1.from_dict(_document()).canonical_json(),
        _CustomConfig(normalizer_ref),
    )


def test_fabricated_normalizer_candidates_and_unknown_formats_fail_during_prepare(tmp_path):
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()

    class OutsideNormalizer:
        def prepare(self, source, *, config, output_root):
            return NormalizedTrainingInputV1(outside, ROW_SCHEMA_VERSION)

    service = TrainingInputPreparationServiceV1(
        prepared_root=(tmp_path / "private").resolve(),
        normalizers={"hostile/v1": OutsideNormalizer()},
        format_verifiers=default_dataset_format_verifiers_v1(),
        root_authority=_ExplicitTestRootAuthority(),
    )
    with pytest.raises(ValueError, match="outside the private"):
        service.prepare(
            OneUseTrainingInputUploadV1(io.BytesIO(b"payload")),
            _custom_request("hostile/v1"),
        )

    class UnknownFormatNormalizer:
        def prepare(self, source, *, config, output_root):
            candidate = output_root / "candidate"
            candidate.mkdir()
            return NormalizedTrainingInputV1(candidate.resolve(), "unknown/v1")

    service = TrainingInputPreparationServiceV1(
        prepared_root=(tmp_path / "private-two").resolve(),
        normalizers={"unknown/v1": UnknownFormatNormalizer()},
        format_verifiers=default_dataset_format_verifiers_v1(),
        root_authority=_ExplicitTestRootAuthority(),
    )
    with pytest.raises(ValueError, match="format is unsupported"):
        service.prepare(
            OneUseTrainingInputUploadV1(io.BytesIO(b"payload")),
            _custom_request("unknown/v1"),
        )


def test_retained_publication_tampering_fails_before_prepare_returns(tmp_path):
    _bundle_value, _dataset_config, prepared = _prepared_dataset(tmp_path)
    delegate = DatasetPreparedInputFormatVerifierV1(ROW_SCHEMA_VERSION)

    class TamperingVerifier:
        format = ROW_SCHEMA_VERSION

        def verify(self, path):
            verified = delegate.verify(path)
            (path / "dataset.jsonl").chmod(0o600)
            with (path / "dataset.jsonl").open("ab") as stream:
                stream.write(b"tampered\n")
            return verified

        def source(self, path, identity):
            return LocalPreparedTrainingInputSource(path, identity)

    class ExistingNormalizer:
        def prepare(self, source, *, config, output_root):
            source.take_stream().close()
            return NormalizedTrainingInputV1(prepared.path.resolve(), ROW_SCHEMA_VERSION)

    service = TrainingInputPreparationServiceV1(
        prepared_root=(tmp_path / "private-prepared").resolve(),
        normalizers={"existing/v1": ExistingNormalizer()},
        format_verifiers={ROW_SCHEMA_VERSION: TamperingVerifier()},
        root_authority=_ExplicitTestRootAuthority(),
    )
    with pytest.raises(Exception, match="changed|digest|invalid|match"):
        service.prepare(
            OneUseTrainingInputUploadV1(io.BytesIO(b"payload")),
            _custom_request("existing/v1"),
        )


def test_prepare_rejects_hardlinked_prepared_publication_before_return(tmp_path):
    _bundle_value, _dataset_config, prepared = _prepared_dataset(tmp_path)
    external = tmp_path / "external-dataset.jsonl"

    class HardlinkingNormalizer:
        def prepare(self, source, *, config, output_root):
            source.take_stream().close()
            os.link(prepared.path / "dataset.jsonl", external)
            return NormalizedTrainingInputV1(prepared.path.resolve(), ROW_SCHEMA_VERSION)

    service = TrainingInputPreparationServiceV1(
        prepared_root=(tmp_path / "private-prepared").resolve(),
        normalizers={"hardlink/v1": HardlinkingNormalizer()},
        format_verifiers=default_dataset_format_verifiers_v1(),
        root_authority=_ExplicitTestRootAuthority(),
    )
    try:
        with pytest.raises(DatasetPrepValidationError):
            service.prepare(
                OneUseTrainingInputUploadV1(io.BytesIO(b"payload")),
                _custom_request("hardlink/v1"),
            )
    finally:
        if external.exists():
            external.chmod(0o600)
            external.unlink()


def test_prepare_rejects_invalid_registered_format_identity(tmp_path):
    _bundle_value, _dataset_config, prepared = _prepared_dataset(tmp_path)
    invalid_format = "format//v1"

    class InvalidFormatNormalizer:
        def prepare(self, source, *, config, output_root):
            source.take_stream().close()
            return NormalizedTrainingInputV1(prepared.path.resolve(), invalid_format)

    class InvalidFormatVerifier:
        format = invalid_format

        def verify(self, path):
            raise AssertionError("invalid format crossed the public grammar boundary")

        def source(self, path, identity):  # pragma: no cover - identity rejects first
            return LocalPreparedTrainingInputSource(path, identity)

    service = TrainingInputPreparationServiceV1(
        prepared_root=(tmp_path / "private-prepared").resolve(),
        normalizers={"invalid-format/v1": InvalidFormatNormalizer()},
        format_verifiers={invalid_format: InvalidFormatVerifier()},
        root_authority=_ExplicitTestRootAuthority(),
    )
    with pytest.raises(ValueError, match="logical identifier"):
        service.prepare(
            OneUseTrainingInputUploadV1(io.BytesIO(b"payload")),
            _custom_request("invalid-format/v1"),
        )


def test_one_use_upload_snapshots_closes_bounds_and_detaches_caller_stream():
    caller = io.BytesIO(b"immutable")
    upload = OneUseTrainingInputUploadV1(caller)
    assert caller.closed
    assert upload.take_stream().read() == b"immutable"
    with pytest.raises(ValueError, match="already consumed"):
        upload.take_stream()
    for operation in (pickle.dumps, copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            operation(upload)

    too_large = io.BytesIO(b"12345")
    with pytest.raises(ValueError, match="byte limit"):
        OneUseTrainingInputUploadV1(too_large, maximum_bytes=4)
    assert too_large.closed

    concurrent = OneUseTrainingInputUploadV1(io.BytesIO(b"atomic"))

    def take_upload():
        try:
            return concurrent.take_stream().read()
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _value: take_upload(), range(32)))
    assert [value for value in results if value is not None] == [b"atomic"]


def _prepared_contract() -> tuple[PreparedTrainingInputIdentity, str]:
    digest = "a" * 64
    identity = PreparedTrainingInputIdentity(
        f"prepared://sha256/{digest}", digest, "b" * 64, 1, ROW_SCHEMA_VERSION,
    )
    document = _document()
    document["dataset"] = {"ref": identity.ref}
    return identity, TrainingInputV1.from_dict(document).canonical_json()


def test_prepared_value_requires_strict_canonical_training_input():
    identity, canonical = _prepared_contract()
    prepared = PreparedTrainingInputV1(identity, TrainingRequest("request", "project", canonical))
    assert prepared.to_dict()["request"]["training_input"] == json.loads(canonical)
    assert "canonical_json" not in prepared.to_dict()["request"]

    malformed = (
        json.dumps(json.loads(canonical), indent=2),
        canonical[:-1] + ',"dataset":{"ref":"' + identity.ref + '"}}',
        canonical[:-1] + ',"extra":1}',
        canonical[:-1] + ',"extra":NaN}',
    )
    for value in malformed:
        with pytest.raises(ValueError):
            PreparedTrainingInputV1(identity, TrainingRequest("request", "project", value))

    path_document = json.loads(canonical)
    path_document["dataset"] = {"ref": "C:\\private\\dataset.jsonl"}
    with pytest.raises(ValueError):
        PreparedTrainingInputV1(
            identity,
            TrainingRequest("request", "project", _canonical(path_document)),
        )

    for unsafe_format in (
        ".", "..", "format/.", "format/..", "format//v1", "../format",
        "file://dataset", "C:/private", "C:\\private\\dataset.jsonl",
        "/format", "format/", "customer prose format", "a" * 65,
    ):
        with pytest.raises(ValueError, match="logical identifier"):
            PreparedTrainingInputIdentity(
                identity.ref, identity.revision, identity.content_digest,
                identity.size_bytes, unsafe_format,
            )
