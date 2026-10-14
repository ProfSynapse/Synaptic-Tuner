"""Real local Git identity, with no provider, model or wheel-builder execution."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

from tuner.execution.providers.modal import runtime_build as build
from tuner.training import modal_local_preparation as local


INSPECTOR = Path(build.__file__).resolve().parents[4] / 'tuner/runtime/modal_build_inspector.py'


def source(tmp_path):
    root = tmp_path / 'engine'
    target = root / 'tuner/runtime/modal_build_inspector.py'
    target.parent.mkdir(parents=True)
    target.write_bytes(INSPECTOR.read_bytes())
    for args in (['init', '-q'], ['add', '.'], ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture']):
        subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)
    head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], check=True, capture_output=True).stdout.decode().strip()
    return root, head


def profile(monkeypatch):
    selected = SimpleNamespace(packaged_runtime={'wheel': {}, 'bootstrap': []}, packages=[], name='fixture', canonical_sha256='a'*64, base_image='image@sha256:'+'b'*64)
    monkeypatch.setattr(build, 'load_profile', lambda _: selected)
    return selected


def wheel(path, raw):
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('tuner/runtime/modal_build_inspector.py', raw)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_git_source_selects_exact_commit_without_importing_checkout(tmp_path, monkeypatch):
    profile(monkeypatch)
    root, head = source(tmp_path)
    intent = build.plan_modal_build_material(Path('fixture'), source_root=root)
    assert intent['engine_source_commit'] == head
    assert intent['inspector_sha256'] == hashlib.sha256(INSPECTOR.read_bytes()).hexdigest()
    assert not Path(build.__file__).resolve().is_relative_to(root)


def test_changed_source_inspector_rejects_before_claim_or_sdk(tmp_path, monkeypatch):
    profile(monkeypatch)
    root, _ = source(tmp_path)
    intent = build.plan_modal_build_material(Path('fixture'), source_root=root)
    (root/'tuner/runtime/modal_build_inspector.py').write_bytes(b'changed')
    calls = []
    with pytest.raises(ValueError, match='source identity|inspector differs'):
        build.capture_modal_build_candidate(sdk=SimpleNamespace(__version__='1.5.4'), client=object(), profile_path=Path('fixture'), build_claim=lambda: calls.append('claim'), app_name='fixture', environment_name='fixture', expected_intent_digest=intent['intent_digest'], source_root=root)
    assert calls == []


def test_wrong_wheel_inspector_rejects_local_material(tmp_path, monkeypatch):
    selected = profile(monkeypatch)
    path = tmp_path/'synaptic_tuner-1.1.0-py3-none-any.whl'
    digest = wheel(path, b'changed')
    with pytest.raises(build.ModalBuildStageFailure):
        build.prepare_capture_build_inputs(selected, tmp_path, path, digest)
    assert not (tmp_path/'build-inputs.json').exists()


def test_actual_wheel_member_matches_installed_policy(tmp_path, monkeypatch):
    selected = profile(monkeypatch)
    selected.python_executable = '/python'
    path = tmp_path/'synaptic_tuner-1.1.0-py3-none-any.whl'
    digest = wheel(path, INSPECTOR.read_bytes())
    result = build.prepare_capture_build_inputs(selected, tmp_path, path, digest)
    assert result[0]['wheel']['sha256'] == digest


def test_local_preparation_retains_real_source_and_rejects_dirty_or_replaced_root(tmp_path, monkeypatch):
    selected = profile(monkeypatch)
    selected.python_executable = '/python'
    root, head = source(tmp_path)
    monkeypatch.setattr(local, 'load_profile', lambda _: selected)
    monkeypatch.setattr(local, 'validate_local_launcher', lambda: None)
    monkeypatch.setattr(local, 'bootstrap_paths', lambda *a, **kw: {})
    seen = []
    def built(source_root, staged, **kwargs):
        seen.append((source_root, kwargs['expected_source_commit']))
        path = staged/'synaptic_tuner-1.1.0-py3-none-any.whl'
        return path, wheel(path, INSPECTOR.read_bytes())
    monkeypatch.setattr(local, 'prepare_current_source_wheel', built)
    intent = build.plan_modal_build_material(Path('fixture'), source_root=root)
    holder = local.prepare_local_inputs(Path('fixture'), root=tmp_path/'private', expected_intent=intent['intent_digest'], source_root=root)
    try:
        assert seen == [(root, head)]
        holder.revalidate(Path('fixture'), intent['intent_digest'], source_root=root)
        with pytest.raises(local.ModalLocalPreparationUnavailable):
            holder.revalidate(Path('fixture'), intent['intent_digest'], source_root=tmp_path)
        (root/'tuner/runtime/modal_build_inspector.py').write_bytes(b'dirty')
        with pytest.raises(local.ModalLocalPreparationUnavailable):
            holder.revalidate(Path('fixture'), intent['intent_digest'])
        (root/'tuner/runtime/modal_build_inspector.py').write_bytes(INSPECTOR.read_bytes())
        root.rename(tmp_path/'moved')
        root.mkdir()
        with pytest.raises(local.ModalLocalPreparationUnavailable):
            holder.revalidate(Path('fixture'), intent['intent_digest'])
    finally:
        holder.close()


def test_missing_git_and_dirty_selected_source_reject(tmp_path, monkeypatch):
    profile(monkeypatch)
    missing = tmp_path/'missing-git'
    missing.mkdir()
    with pytest.raises(ValueError, match='source identity'):
        build.plan_modal_build_material(Path('fixture'), source_root=missing)
    root, _ = source(tmp_path)
    # Dirty tracked input other than the inspector must also reject planning.
    (root/'tuner/runtime/modal_build_inspector.py').write_bytes(INSPECTOR.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='clean engine source'):
        build.plan_modal_build_material(Path('fixture'), source_root=root)


def test_restricted_metrics_source_scope_is_not_extended(tmp_path):
    from tuner.training.modal_metrics_qualification import current_clean_commit, MetricsQualificationUnavailable
    root, _ = source(tmp_path)
    assert build.current_clean_source_commit(root)
    with pytest.raises(MetricsQualificationUnavailable):
        current_clean_commit(root)


def test_changed_wheel_does_not_claim_or_access_sdk(tmp_path, monkeypatch):
    profile(monkeypatch)
    root, _ = source(tmp_path)
    intent = build.plan_modal_build_material(Path('fixture'), source_root=root)
    def built(*args, **kwargs):
        path = args[1]/'synaptic_tuner-1.1.0-py3-none-any.whl'
        return path, wheel(path, b'changed')
    monkeypatch.setattr(build, 'prepare_current_source_wheel', built)
    calls = []
    with pytest.raises(build.ModalBuildStageFailure):
        build.capture_modal_build_candidate(sdk=SimpleNamespace(__version__='1.5.4'), client=object(), profile_path=Path('fixture'), build_claim=lambda: calls.append('claim'), app_name='fixture', environment_name='fixture', expected_intent_digest=intent['intent_digest'], source_root=root)
    assert calls == []


def test_capture_rechecks_same_real_source_before_claim(tmp_path, monkeypatch):
    selected = profile(monkeypatch)
    selected.python_executable = '/python'
    root, _ = source(tmp_path)
    real_plan = build.plan_modal_build_material
    intent = real_plan(Path('fixture'), source_root=root)
    observed = []
    def planning(path, **kwargs):
        observed.append(kwargs.get('source_root'))
        return real_plan(path, **kwargs)
    monkeypatch.setattr(build, 'plan_modal_build_material', planning)
    def built(source_root, staged, **kwargs):
        assert source_root == root
        path = staged/'synaptic_tuner-1.1.0-py3-none-any.whl'
        return path, wheel(path, INSPECTOR.read_bytes())
    monkeypatch.setattr(build, 'prepare_current_source_wheel', built)
    # Provider image/capture are scripted; local Git, policy and wheel checks run.
    calls, files = [], {}
    class Image:
        object_id = 'im-SourceFixture'
        def entrypoint(self, value):
            return self
        def add_local_file(self, path, target, **kwargs):
            files[target] = Path(path).read_bytes()
            return self
        def run_commands(self, command):
            return self
    def image(reference):
        calls.append('sdk')
        return Image()
    def output(sandbox):
        document = {'schema_version': 'synaptic-modal-build-capture/v1',
            'image_id': Image.object_id, 'inspector_sha256': intent['inspector_sha256'],
            'measured': {'build_inputs_digest': hashlib.sha256(files['/opt/synaptic-runtime/build-inputs.json']).hexdigest(),
                'package': {'digest': json.loads(files['/opt/synaptic-runtime/build-inputs.json'])['wheel']['sha256']}}}
        return (json.dumps(document, sort_keys=True, separators=(',', ':'))+'\n').encode()
    monkeypatch.setattr(build, '_build_image_with_failure_id', lambda image, app: image)
    monkeypatch.setattr(build, '_capture_output', output)
    monkeypatch.setattr(build, '_cleanup_sandbox', lambda sandbox: calls.append('cleanup'))
    sdk = SimpleNamespace(__version__='1.5.4', Image=SimpleNamespace(from_registry=image),
        Sandbox=SimpleNamespace(create=lambda *a, **kw: object()))
    result = build.capture_modal_build_candidate(sdk=sdk, client=object(), profile_path=Path('fixture'), build_claim=lambda: calls.append('claim'), build_app=object(), app_name='fixture', environment_name='fixture', expected_intent_digest=intent['intent_digest'], source_root=root)
    assert result.image_id == Image.object_id
    assert observed == [root, root]
    assert calls == ['claim', 'sdk', 'cleanup']
