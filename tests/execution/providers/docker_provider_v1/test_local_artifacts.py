"""Real local files and signed reader chain; daemon commands are scripted only."""
from dataclasses import replace
import hashlib
import hmac
import json
import os
import time

import pytest

from tuner.cloud.derived_training_image import DockerLaunchAuthority, _validate_docker_inputs
from tuner.cloud.hf_training_image_lock import CommandResult
from tuner.execution.providers.docker_provider_v1.cli_transport import DockerCLISessionV1
from tuner.execution.providers.docker_provider_v1.local_artifacts import DockerLocalArtifactsV1
from tuner.execution.providers.docker_provider_v1.model import (
    PreparedDockerPlanV1, DockerEffectIdentityV1, labels_for,
    DockerArtifactInventoryRequestV1, DockerArtifactReadRequestV1,
    AuthenticatedDockerArtifactInventoryV1, DockerArtifactChunkV1, DockerArtifactEOFV1,
    DockerProviderError,
)

CID = 'f' * 64
KEY = b'deterministic-test-evidence-key-only'


def inventory_tag(content):
    return hmac.new(KEY, b'inventory:' + content.content_digest.encode(), hashlib.sha256).hexdigest()


def eof_tag(value):
    raw = json.dumps({field: getattr(value, field) for field in value.__dataclass_fields__
                      if field != 'tag'}, sort_keys=True).encode()
    return hmac.new(KEY, b'eof:' + raw, hashlib.sha256).hexdigest()


class Signer:
    def inventory(self, content):
        return AuthenticatedDockerArtifactInventoryV1(content, 'test-only', 'key-v1', inventory_tag(content))
    def eof(self, value):
        signed = replace(value, authority_ref='test-only', key_ref='key-v1')
        return replace(signed, tag=eof_tag(signed))
    def authenticate_inventory(self, value):
        return (type(value) is AuthenticatedDockerArtifactInventoryV1 and
                value.authority_ref == 'test-only' and value.key_ref == 'key-v1' and
                hmac.compare_digest(value.tag, inventory_tag(value.content)))
    def authenticate_eof(self, value):
        return (type(value) is DockerArtifactEOFV1 and value.authority_ref == 'test-only' and
                value.key_ref == 'key-v1' and hmac.compare_digest(value.tag, eof_tag(value)))


def transport(tmp_path, plan, labels):
    executable, config = tmp_path / 'docker', tmp_path / 'config'
    executable.write_bytes(b'test-only CLI authority, never executed')
    config.mkdir()
    calls = []
    def runner(spec):
        calls.append(spec.argv)
        args = spec.argv[5:]
        if args[0] == 'info':
            output = 'daemon-exact'
        elif args[:2] == ('image', 'inspect'):
            output = json.dumps(plan.profile.image.image_digest) + '\n[]'
        elif args[:2] == ('container', 'ls'):
            output = CID
        elif args[:2] == ('container', 'inspect'):
            output = '\n'.join(map(json.dumps, [CID, '/' + labels.container_name,
                plan.profile.image.image_digest, DockerCLISessionV1._labels(labels), 'exited', 0]))
        else:
            raise AssertionError('mutation command forbidden')
        return CommandResult(output.encode())
    return DockerCLISessionV1(authority=DockerLaunchAuthority({}, _validate_docker_inputs(executable, config)),
        endpoint='unix:///fixture.sock', daemon_id='daemon-exact',
        image_refs={plan.profile.image.image_ref: plan.profile.image.image_digest},
        source_roots={}, artifact_roots={}, environment={}, deadline=time.monotonic() + 60,
        runner=runner, profile=plan.profile), calls


@pytest.fixture
def stack(tmp_path, profile, plan, run):
    prepared = PreparedDockerPlanV1(profile, run.project_ref, run.run_id, plan.plan_fingerprint,
                                   plan.basis.source_digest, 'a' * 64)
    labels = labels_for(DockerEffectIdentityV1('b' * 64, 'submit-effect', 'submit', prepared))
    root = tmp_path / 'artifacts'
    root.mkdir()
    (root / 'result.json').write_bytes(b'fixture-result')
    control, calls = transport(tmp_path, prepared, labels)
    signer = Signer()
    port = DockerLocalArtifactsV1(plan=prepared, labels=labels, container_ref=CID, control=control,
        root=root, role_paths={'result': 'result.json'}, deadline=time.monotonic() + 60,
        sign_inventory=signer.inventory, sign_eof=signer.eof)
    request = DockerArtifactInventoryRequestV1(labels, 'c' * 64, 1, profile.profile_digest,
        prepared.digest, profile.artifacts.digest, profile.roots.artifact_ref)
    yield port, request, root, signer, calls
    port.close()


def read_request(inventory):
    value = inventory.content
    entry = value.entries[0]
    return DockerArtifactReadRequestV1(value.labels, value.content_digest, entry.descriptor.role,
        entry.relative_path, entry.descriptor.size_bytes, entry.descriptor.size_bytes,
        entry.descriptor.sha256, entry.file_identity_digest, value.generation,
        value.profile_digest, value.prepared_plan_digest, value.artifact_contract_digest, value.artifact_root_ref)


def test_inventory_and_stream_are_signed_and_exact(stack):
    port, request, _, signer, calls = stack
    inventory = port.artifact_inventory(request)
    assert signer.authenticate_inventory(inventory)
    events = list(port.iter_artifact_events(read_request(inventory)))
    assert b''.join(e.data for e in events if type(e) is DockerArtifactChunkV1) == b'fixture-result'
    assert signer.authenticate_eof(events[-1])
    assert not any('create' in c or 'start' in c for c in calls)


@pytest.mark.parametrize('mutation', ['replace', 'symlink', 'hardlink', 'truncate', 'ancestor'])
def test_changed_file_between_inventory_and_stream_rejects(stack, tmp_path, mutation):
    port, request, root, _, _ = stack
    inventory = port.artifact_inventory(request)
    path = root / 'result.json'
    if mutation == 'ancestor':
        root.rename(tmp_path / 'old-root')
        root.mkdir()
        path.write_bytes(b'fixture-result')
    elif mutation == 'truncate':
        path.write_bytes(b'x')
    else:
        path.rename(root / 'original')
        if mutation == 'replace':
            path.write_bytes(b'fixture-result')
        elif mutation == 'symlink':
            path.symlink_to('original')
        else:
            os.link(root / 'original', path)
    with pytest.raises(DockerProviderError):
        list(port.iter_artifact_events(read_request(inventory)))


@pytest.mark.parametrize('size', [0, 1_048_576, 1_048_577])
def test_file_bounds_and_empty_file(stack, size):
    port, request, root, signer, _ = stack
    (root / 'result.json').write_bytes(b'x' * size)
    if size > port._plan.profile.artifacts.maximum_artifact_bytes:
        with pytest.raises(DockerProviderError):
            port.artifact_inventory(request)
        return
    events = list(port.iter_artifact_events(read_request(port.artifact_inventory(request))))
    assert sum(len(e.data) for e in events if type(e) is DockerArtifactChunkV1) == size
    assert signer.authenticate_eof(events[-1])


@pytest.mark.parametrize('mode', ['normal', 'large', 'bad_eof'])
def test_actual_reader_authenticated_coordinator_artifact_chain(tmp_path, profile, plan, run, seams, monkeypatch, mode):
    """Reuse existing admitted coordinator fixture; no reader auth bypass."""
    from tests.execution.providers.docker_provider_v1 import test_effects as fixture
    from tuner.execution.providers.docker_provider_v1.reader import DockerProviderRunReaderV1
    from tuner.execution.coordinator_v1.model import ProviderRunReadRequestV1
    if mode == 'large':
        from tuner.execution.providers.docker_provider_v1.model import DockerArtifactContractV1
        from synaptic_tuner.api.v1.planning import TrainingPlan
        profile = replace(profile, artifacts=DockerArtifactContractV1(('result',), 8 * 1024**2, 8 * 1024**2))
        plan = TrainingPlan(plan.schema_version, replace(plan.basis, artifact_policy_digest=profile.artifacts.digest), plan.provider_plan)
    original_authority = fixture.ReadAuthority
    original_read_port = fixture.ReadPort
    retained = {}

    class Evidence(original_authority, Signer):
        authenticate_inventory = Signer.authenticate_inventory
        authenticate_eof = Signer.authenticate_eof

    def factory(selected_profile, catalog, binding_authority, authorization,
                evidence_authority, reader, *, observed_at):
        submit = next(value.content for value in (catalog.resolve(key) for key in catalog.values)
                      if value.content.effect_kind == 'submit')
        labels = labels_for(submit.identity)
        root = tmp_path / 'full-chain-artifacts'
        root.mkdir()
        (root / 'result.json').write_bytes(b'fixture-result')
        native_root = tmp_path / 'native-cli'
        native_root.mkdir()
        native, calls = transport(native_root, submit.plan, labels)
        local = DockerLocalArtifactsV1(plan=submit.plan, labels=labels, container_ref=CID,
            control=native, root=root, role_paths={'result': 'result.json'},
            deadline=time.monotonic() + 60, sign_inventory=evidence_authority.inventory,
            sign_eof=evidence_authority.eof)
        # Existing coordinator mock uses container-1. Exact A lookup returns CID;
        # preserve the original scoped provider reference only at fixture boundary.
        # To avoid changing that reference, script its existing control create/start
        # results to use CID before running the coordinator below.
        class Composite:
            lookup = local.lookup
            artifact_inventory = local.artifact_inventory
            iter_artifact_events = local.iter_artifact_events
            logs = reader.logs  # unrelated existing log fixture, not B1 capability
        class Gate:
            calls = 0
            def authenticate(self, request):
                self.calls += 1
                self.last = request
                if type(request) is not ProviderRunReadRequestV1:
                    return False
                rebuilt = ProviderRunReadRequestV1(**{f: getattr(request, f) for f in request.__dataclass_fields__})
                return (rebuilt == request and request.run == run and
                        request.submit_command_bytes == submit.command_bytes and
                        authorization.authenticate(request) is True)
        gate = Gate()
        actual = DockerProviderRunReaderV1(selected_profile, catalog, binding_authority,
            gate, evidence_authority, Composite(), observed_at=observed_at)
        retained.update(local=local, actual=actual, gate=gate, calls=calls)
        return actual

    _, _, _, control, _ = seams
    original_create, original_start = control.create_once, control.start_once
    def create(**kwargs):
        value = original_create(**kwargs)
        return replace(value, container_ref=CID)
    def start(container_ref, labels):
        value = original_start(container_ref, labels)
        return replace(value, container_ref=CID)
    monkeypatch.setattr(control, 'create_once', create)
    monkeypatch.setattr(control, 'start_once', start)
    monkeypatch.setattr(fixture, 'ReadAuthority', Evidence)
    monkeypatch.setattr(fixture, 'DockerProviderRunReaderV1', factory)
    assert original_read_port is fixture.ReadPort
    try:
        fixture.test_both_opaque_profiles_run_unchanged_coordinator_foundation_and_operations(
            profile, plan, run, seams, 'lifecycle')
        assert type(retained['actual']) is DockerProviderRunReaderV1
        assert retained['gate'].calls >= 3
        if mode in ('large', 'bad_eof'):
            request = retained['gate'].last
            local = retained['local']
            content = b'escaped\n\x00' * 250000 if mode == 'large' else b'fixture-result'
            (local._root / 'result.json').write_bytes(content)
            manifest = retained['actual'].artifacts(request)
            if mode == 'bad_eof':
                original = local._sign_eof
                local._sign_eof = lambda value: replace(original(value), tag='0' * 64)
                with pytest.raises(DockerProviderError):
                    list(retained['actual'].iter_artifact_bytes(request, manifest, 'result', maximum_bytes=8 * 1024**2))
            else:
                chunks = list(retained['actual'].iter_artifact_bytes(request, manifest, 'result', maximum_bytes=8 * 1024**2))
                assert len(chunks) >= 2
                assert all(len(chunk) <= 1048576 for chunk in chunks)
                assert b''.join(chunks) == content
        assert not any('create' in c or 'start' in c for c in retained['calls'])
    finally:
        if 'local' in retained:
            retained['local'].close()


def test_mutation_after_first_chunk_no_eof(stack):
    port, request, root, _, _ = stack
    inventory = port.artifact_inventory(request)
    stream = port.iter_artifact_events(read_request(inventory))
    assert type(next(stream)) is DockerArtifactChunkV1
    (root / 'result.json').write_bytes(b'altered-result')
    with pytest.raises(DockerProviderError):
        list(stream)


@pytest.mark.parametrize('phase', ['hash', 'stream', 'inventory_signer', 'eof_signer'])
def test_exhausted_deadline_rejects_before_success(stack, monkeypatch, phase):
    port, request, _, signer, _ = stack
    if phase == 'hash':
        original = os.read
        def read(fd, size):
            value = original(fd, size)
            port._deadline = time.monotonic() - 1
            return value
        monkeypatch.setattr(os, 'read', read)
        with pytest.raises(DockerProviderError):
            port.artifact_inventory(request)
        return
    if phase == 'inventory_signer':
        def delayed(value):
            port._deadline = time.monotonic() - 1
            return signer.inventory(value)
        port._sign_inventory = delayed
        with pytest.raises(DockerProviderError):
            port.artifact_inventory(request)
        return
    inventory = port.artifact_inventory(request)
    if phase == 'eof_signer':
        def delayed(value):
            port._deadline = time.monotonic() - 1
            return signer.eof(value)
        port._sign_eof = delayed
        with pytest.raises(DockerProviderError):
            list(port.iter_artifact_events(read_request(inventory)))
    else:
        stream = port.iter_artifact_events(read_request(inventory))
        assert type(next(stream)) is DockerArtifactChunkV1
        port._deadline = time.monotonic() - 1
        with pytest.raises(DockerProviderError):
            list(stream)


def test_fifo_rejects_without_blocking(stack):
    port, request, root, _, _ = stack
    path = root / 'result.json'
    path.unlink()
    os.mkfifo(path)
    started = time.monotonic()
    with pytest.raises(DockerProviderError):
        port.artifact_inventory(request)
    assert time.monotonic() - started < 2


def test_total_bounds_across_roles(tmp_path, profile, plan, run):
    from tuner.execution.providers.docker_provider_v1.model import DockerArtifactContractV1
    profile = replace(profile, artifacts=DockerArtifactContractV1(('first', 'second'), 10, 10))
    prepared = PreparedDockerPlanV1(profile, run.project_ref, run.run_id, plan.plan_fingerprint,
                                   plan.basis.source_digest, 'a' * 64)
    labels = labels_for(DockerEffectIdentityV1('b' * 64, 'submit-effect', 'submit', prepared))
    root = tmp_path / 'artifacts'
    root.mkdir()
    (root / 'first').write_bytes(b'123456')
    (root / 'second').write_bytes(b'123456')
    control, _ = transport(tmp_path, prepared, labels)
    signer = Signer()
    with DockerLocalArtifactsV1(plan=prepared, labels=labels, container_ref=CID, control=control,
            root=root, role_paths={'first': 'first', 'second': 'second'}, deadline=time.monotonic() + 60,
            sign_inventory=signer.inventory, sign_eof=signer.eof) as port:
        request = DockerArtifactInventoryRequestV1(labels, 'c' * 64, 1, profile.profile_digest,
            prepared.digest, profile.artifacts.digest, profile.roots.artifact_ref)
        with pytest.raises(DockerProviderError):
            port.artifact_inventory(request)


def test_close_invalidates_reads(stack):
    port, request, _, _, _ = stack
    port.close()
    with pytest.raises(DockerProviderError):
        port.artifact_inventory(request)


@pytest.mark.parametrize('mutation', ['file', 'root'])
def test_eof_signer_mutation_cannot_emit_success(stack, tmp_path, mutation):
    port, request, root, signer, _ = stack
    inventory = port.artifact_inventory(request)
    def mutate(value):
        signed = signer.eof(value)
        if mutation == 'file':
            (root / 'result.json').write_bytes(b'altered-result')
        else:
            root.rename(tmp_path / 'old-root')
            root.mkdir()
        return signed
    port._sign_eof = mutate
    stream = port.iter_artifact_events(read_request(inventory))
    assert type(next(stream)) is DockerArtifactChunkV1
    with pytest.raises(DockerProviderError):
        list(stream)
