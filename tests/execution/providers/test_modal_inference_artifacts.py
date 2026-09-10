"""Provider-free tests for exact mounted Modal inference artifact transport."""

from __future__ import annotations
from dataclasses import replace
import hashlib, os
from pathlib import Path
import pytest
from tuner.execution.providers.modal.contracts import (
    ArtifactMemberV1,
    ArtifactRole,
    provider_entry_identity,
)
from tuner.execution.providers.modal.inference_artifacts import (
    ModalInferenceArtifactError,
    ModalMountedInferenceArtifactReader,
)
from tuner.inference.retrieved_model import _materialize_admitted_sft_model, _open_root
from tuner.inference import retrieved_model
from tests.inference.test_retrieved_model import _fixture


def _case(tmp_path):
    run, values, operations = _fixture(tmp_path)
    effect = "effect-submit"
    volume = "volume-artifacts"
    output = tmp_path / "operations" / effect / "output"
    output.mkdir(parents=True)
    members = []
    for role in sorted(values):
        data = values[role]
        path = f"operations/{effect}/output/{role}"
        (output / role).write_bytes(data)
        members.append(
            ArtifactMemberV1(
                ArtifactRole(role),
                path,
                len(data),
                hashlib.sha256(data).hexdigest(),
                provider_entry_identity(volume, path, len(data)),
            )
        )
    fd = _open_root(tmp_path)
    reader = ModalMountedInferenceArtifactReader(
        root=tmp_path,
        root_fd=fd,
        run=run,
        artifact_volume_id=volume,
        effect_id=effect,
        artifacts=operations.inventory,
        members=tuple(members),
    )
    return run, values, operations, tuple(members), fd, reader


def test_mounted_reader_materializes_same_metadata_without_runs_api(tmp_path):
    run, _, operations, _, fd, reader = _case(tmp_path)
    try:
        result = _materialize_admitted_sft_model(
            run=run,
            artifacts=operations.inventory,
            root=tmp_path,
            root_fd=fd,
            read_artifact=reader.read_artifact,
        )
        assert (
            result.model_ref,
            result.model_revision,
            result.tokenizer_revision,
            result.model_kind,
        ) == ("example/model", "c" * 40, "c" * 40, "lora")
        assert result.artifacts == operations.inventory
        os.fstat(fd)
    finally:
        os.close(fd)


@pytest.mark.parametrize("field", ("role", "hash", "size", "entry", "volume", "effect"))
def test_constructor_rejects_identity_mismatch_before_body_read(tmp_path, field):
    run, _, operations, members, fd, _ = _case(tmp_path)
    changed = list(members)
    kwargs = {}
    if field == "volume":
        kwargs["artifact_volume_id"] = "other-volume"
    elif field == "effect":
        kwargs["effect_id"] = "other-effect"
    else:
        item = changed[0]
        updates = (
            {"role": ArtifactRole.TOKENIZER}
            if field == "role"
            else (
                {"sha256": "0" * 64}
                if field == "hash"
                else (
                    {"size": item.size + 1}
                    if field == "size"
                    else {"provider_entry_id": "entry-other"}
                )
            )
        )
        changed[0] = replace(item, **updates)
    try:
        with pytest.raises(ModalInferenceArtifactError):
            ModalMountedInferenceArtifactReader(
                root=tmp_path,
                root_fd=fd,
                run=run,
                artifact_volume_id=kwargs.get("artifact_volume_id", "volume-artifacts"),
                effect_id=kwargs.get("effect_id", "effect-submit"),
                artifacts=operations.inventory,
                members=tuple(changed),
            )
        os.fstat(fd)
    finally:
        os.close(fd)


@pytest.mark.parametrize("attack", ("extra", "symlink", "hardlink", "fifo"))
def test_reader_rejects_nonexact_or_redirected_output(tmp_path, attack):
    run, _, operations, _, fd, reader = _case(tmp_path)
    output = tmp_path / "operations" / "effect-submit" / "output"
    target = output / "final_model"
    if attack == "extra":
        (output / "extra").write_bytes(b"x")
    else:
        target.unlink()
        if attack == "symlink":
            target.symlink_to(output / "tokenizer")
        elif attack == "hardlink":
            os.link(output / "tokenizer", target)
        else:
            os.mkfifo(target)
    try:
        with pytest.raises(
            ModalInferenceArtifactError, match="^modal_inference_artifact_invalid$"
        ):
            tuple(reader.read_artifact(run, operations.inventory[0]).iter_bytes())
        os.fstat(fd)
    finally:
        os.close(fd)


@pytest.mark.parametrize("attack", ("replace", "truncate", "overflow", "wronghash"))
def test_stream_rechecks_file_identity_and_exact_bytes(tmp_path, attack):
    run, _, operations, _, fd, reader = _case(tmp_path)
    path = tmp_path / "operations" / "effect-submit" / "output" / "final_model"
    stream = reader.read_artifact(run, operations.inventory[0])
    try:
        iterator = stream.iter_bytes()
        if attack == "replace":
            next(iterator)
            data = path.read_bytes()
            path.unlink()
            path.write_bytes(data)
            with pytest.raises(
                ModalInferenceArtifactError, match="^modal_inference_artifact_invalid$"
            ):
                tuple(iterator)
        else:
            if attack == "truncate":
                path.write_bytes(path.read_bytes()[:-1])
            elif attack == "overflow":
                path.write_bytes(path.read_bytes() + b"x")
            else:
                data = bytearray(path.read_bytes())
                data[0] ^= 1
                path.write_bytes(data)
            with pytest.raises(
                ModalInferenceArtifactError, match="^modal_inference_artifact_invalid$"
            ):
                tuple(iterator)
        os.fstat(fd)
    finally:
        os.close(fd)


@pytest.mark.parametrize("field", ("run", "role", "hash", "size"))
def test_request_mismatch_is_rejected_before_open(tmp_path, field):
    run, _, operations, _, fd, reader = _case(tmp_path)
    artifact = operations.inventory[0]
    if field == "run":
        run = replace(run, run_id="other")
    elif field == "role":
        artifact = replace(artifact, role="other")
    elif field == "hash":
        artifact = replace(artifact, sha256="0" * 64)
    else:
        artifact = replace(artifact, size_bytes=artifact.size_bytes + 1)
    try:
        with pytest.raises(ModalInferenceArtifactError):
            reader.read_artifact(run, artifact)
        os.fstat(fd)
    finally:
        os.close(fd)


def test_partial_stream_close_releases_leaf_and_preserves_borrowed_root(tmp_path):
    run, _, operations, _, fd, reader = _case(tmp_path)
    stream = reader.read_artifact(run, operations.inventory[0])
    iterator = stream.iter_bytes()
    next(iterator)
    iterator.close()
    os.fstat(fd)
    os.close(fd)


def test_reader_rejects_ancestor_replacement_and_root_metadata_mutation(tmp_path):
    run, _, operations, _, fd, reader = _case(tmp_path)
    operations_dir = tmp_path / "operations"
    operations_dir.rename(tmp_path / "old-operations")
    operations_dir.mkdir()
    try:
        with pytest.raises(
            ModalInferenceArtifactError, match="^modal_inference_artifact_invalid$"
        ):
            tuple(reader.read_artifact(run, operations.inventory[0]).iter_bytes())
        object.__setattr__(reader, "_effect_id", "other")
        with pytest.raises(
            ModalInferenceArtifactError, match="^modal_inference_artifact_invalid$"
        ):
            tuple(reader.read_artifact(run, operations.inventory[0]).iter_bytes())
    finally:
        os.close(fd)


def test_many_extra_members_deny_before_leaf_open(tmp_path, monkeypatch):
    run, _, operations, _, fd, reader = _case(tmp_path)
    output = tmp_path / "operations" / "effect-submit" / "output"
    for index in range(100):
        (output / f"extra-{index}").write_bytes(b"x")
    real = os.open
    real_scandir = os.scandir
    leaf = []
    scanned = []

    def opened(path, *args, **kwargs):
        if path == "final_model":
            leaf.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", opened)

    class BoundedEntries:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            self.iterator = iter(self.value.__enter__())
            return self

        def __exit__(self, *args):
            return self.value.__exit__(*args)

        def __iter__(self):
            return self

        def __next__(self):
            item = next(self.iterator)
            scanned.append(item.name)
            return item

    monkeypatch.setattr(os, "scandir", lambda path: BoundedEntries(real_scandir(path)))
    try:
        with pytest.raises(ModalInferenceArtifactError):
            tuple(reader.read_artifact(run, operations.inventory[0]).iter_bytes())
    finally:
        os.close(fd)
    assert leaf == [] and len(scanned) == 6


@pytest.mark.parametrize("field", ("_effect_id", "_artifact_volume_id"))
def test_retained_scalar_mutation_is_rejected(tmp_path, field):
    run, _, operations, _, fd, reader = _case(tmp_path)
    object.__setattr__(reader, field, "other")
    try:
        with pytest.raises(ModalInferenceArtifactError):
            tuple(reader.read_artifact(run, operations.inventory[0]).iter_bytes())
    finally:
        os.close(fd)


def test_output_ancestor_swap_after_first_yield_is_rejected(tmp_path):
    run, _, operations, _, fd, reader = _case(tmp_path)
    output = tmp_path / "operations" / "effect-submit" / "output"
    iterator = reader.read_artifact(run, operations.inventory[0]).iter_bytes()
    next(iterator)
    output.rename(output.parent / "old-output")
    output.mkdir()
    try:
        with pytest.raises(ModalInferenceArtifactError):
            tuple(iterator)
    finally:
        os.close(fd)


@pytest.mark.parametrize("primary", (False, True))
def test_owned_descriptor_close_failures_attempt_both_and_preserve_primary(
    tmp_path, monkeypatch, primary
):
    run, _, operations, _, fd, reader = _case(tmp_path)
    real_open = os.open
    real_close = os.close
    owned = []
    closed = []

    def opened(path, *args, **kwargs):
        value = real_open(path, *args, **kwargs)
        if path == "final_model" or (
            path == "output" and not any(kind == "output" for kind, _ in owned)
        ):
            owned.append((path, value))
        return value

    def close(value):
        real_close(value)
        if value in {item for _, item in owned}:
            closed.append(value)
            raise OSError("private close")

    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "close", close)
    iterator = reader.read_artifact(run, operations.inventory[0]).iter_bytes()
    next(iterator)
    if primary:
        object.__setattr__(reader, "_effect_id", "changed")
    try:
        with pytest.raises(
            ModalInferenceArtifactError, match="^modal_inference_artifact_invalid$"
        ):
            tuple(iterator)
    finally:
        real_close(fd)
    assert {item for _, item in owned}.issubset(closed)


def test_cleanup_control_interrupt_outranks_prior_ordinary_close_failure(
    tmp_path, monkeypatch
):
    run, _, operations, _, fd, reader = _case(tmp_path)
    real_open = os.open
    real_close = os.close
    owned = []
    closed = []

    def opened(path, *args, **kwargs):
        value = real_open(path, *args, **kwargs)
        if path == "final_model" or (
            path == "output" and not any(kind == "output" for kind, _ in owned)
        ):
            owned.append((path, value))
        return value

    def close(value):
        real_close(value)
        if value in {item for _, item in owned}:
            closed.append(value)
            if len(closed) == 1:
                raise OSError("ordinary")
            raise KeyboardInterrupt

    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "close", close)
    iterator = reader.read_artifact(run, operations.inventory[0]).iter_bytes()
    next(iterator)
    try:
        with pytest.raises(KeyboardInterrupt):
            tuple(iterator)
    finally:
        real_close(fd)
    assert {item for _, item in owned}.issubset(closed)


def test_materialization_abort_closes_mounted_stream_descriptors(tmp_path, monkeypatch):
    run, _, operations, _, fd, reader = _case(tmp_path)

    def mounted_descriptors():
        result = []
        for name in os.listdir("/proc/self/fd"):
            try:
                target = os.readlink("/proc/self/fd/" + name)
            except OSError:
                continue
            if str(tmp_path / "operations" / "effect-submit" / "output") in target:
                result.append(target)
        return result

    monkeypatch.setattr(retrieved_model.os, "write", lambda descriptor, data: 0)
    try:
        with pytest.raises(OSError, match="short artifact write"):
            _materialize_admitted_sft_model(
                run=run,
                artifacts=operations.inventory,
                root=tmp_path,
                root_fd=fd,
                read_artifact=reader.read_artifact,
            )
        os.fstat(fd)
    finally:
        os.close(fd)
    assert mounted_descriptors() == []
