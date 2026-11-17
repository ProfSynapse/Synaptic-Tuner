"""Offline REST v2 contract and billing cleanup tests for the skill launcher."""
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path('fine-tuning/scripts/runpod_run_job.py')


@pytest.fixture
def launcher(monkeypatch):
    spec = importlib.util.spec_from_file_location('runpod_run_job', ROOT / '.skills' / SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    monkeypatch.delenv('HF_TOKEN', raising=False)
    return module


@pytest.fixture
def args(launcher, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['runpod_run_job.py', '--run-tag', 'offline-test',
        '--repo-url', 'https://github.com/test/repo.git', '--commit', 'a' * 40,
        '--wrapper', 'job.sh'])
    return launcher.parse_args()


def test_v2_create_spec(launcher, args):
    args.allowed_cuda = '12.8, 12.9'
    args.min_download = 700  # legacy callers must not send the removed field
    spec = launcher.build_pod_spec(args)
    assert spec == {
        'name': 'offline-test', 'image': args.image,
        'gpu': {'id': args.gpu, 'count': 1, 'allowedCudaVersions': ['12.8', '12.9']},
        'cloud': 'COMMUNITY', 'disk': 60,
        'mounts': {'persistent': {'size': 20, 'path': '/workspace'}},
        'env': {'RUN_TAG': 'offline-test', 'HF_HUB_DISABLE_XET': '1',
                'HF_HUB_ENABLE_HF_TRANSFER': '0'},
        'entrypoint': ['bash', '-lc'], 'cmd': [launcher.build_job_command(args)],
    }


def test_image_defaults_and_no_cuda(launcher, args):
    args.no_entrypoint_override = True
    spec = launcher.build_pod_spec(args)
    assert 'entrypoint' not in spec and 'cmd' not in spec
    assert spec['gpu'] == {'id': args.gpu, 'count': 1}


@pytest.mark.parametrize('method,path,body,response', [
    ('POST', '/pods', {'name': 'test', 'gpu': {'id': 'gpu', 'count': 1}},
     b'{"id":"pod-test","status":"PROVISIONING","cost":0.44}'),
    ('DELETE', '/pods/pod-test', None, b''),
])
def test_v2_http_contract(launcher, monkeypatch, method, path, body, response):
    opened = MagicMock()
    opened.return_value.__enter__.return_value.read.return_value = response
    monkeypatch.setattr(launcher.urllib.request, 'urlopen', opened)
    result = launcher.api(method, path, 'fake-key', body)
    request = opened.call_args.args[0]
    assert request.full_url == 'https://api.runpod.io/v2' + path
    assert request.method == method
    assert request.get_header('Authorization') == 'Bearer fake-key'
    assert request.data == (json.dumps(body).encode() if body else None)
    assert result == (json.loads(response) if response else {})


@pytest.mark.parametrize('outcome', ['probe', 'exited', 'timeout', 'boot_fail', 'poll_exception'])
def test_cleanup_on_every_lifecycle_exit(launcher, args, monkeypatch, outcome):
    calls = MagicMock(return_value={'id': 'pod-test', 'cost': 0.44})
    monkeypatch.setattr(launcher, 'api', calls)
    poll = MagicMock(return_value={'status': 'EXITED', 'uptime': 1})
    monkeypatch.setattr(launcher, 'poll_pod', poll)
    if outcome == 'probe':
        args.probe_only = True
        poll.return_value = {'status': 'RUNNING', 'uptime': 5}
    elif outcome == 'timeout':
        args.timeout_min = 0
    elif outcome == 'boot_fail':
        poll.return_value = {'status': 'RUNNING', 'uptime': 0}
        monkeypatch.setattr(launcher, 'BOOT_TIMEOUT_S', -1)
    elif outcome == 'poll_exception':
        poll.side_effect = KeyboardInterrupt
    if outcome == 'poll_exception':
        with pytest.raises(KeyboardInterrupt):
            launcher.run_once(args, 'fake-key', 1)
    else:
        assert launcher.run_once(args, 'fake-key', 1) == {
            'probe': 0, 'exited': 0, 'timeout': 1, 'boot_fail': 'boot_fail'}[outcome]
    assert calls.call_args_list[0].args == ('POST', '/pods', 'fake-key', launcher.build_pod_spec(args))
    assert calls.call_args_list[-1].args == ('DELETE', '/pods/pod-test', 'fake-key')


def test_cleanup_failure_prevents_next_host(launcher, args, monkeypatch):
    calls = MagicMock(side_effect=[{'id': 'pod-test'}, RuntimeError('delete failed'),
                                  RuntimeError('delete failed'), RuntimeError('delete failed')])
    monkeypatch.setattr(launcher, 'api', calls)
    monkeypatch.setattr(launcher, 'parse_args', lambda: args)
    monkeypatch.setenv('RUNPOD_API_KEY', 'fake-key')
    monkeypatch.setattr(launcher, 'poll_pod', lambda *_: {'status': 'RUNNING', 'uptime': 0})
    monkeypatch.setattr(launcher, 'BOOT_TIMEOUT_S', -1)
    with pytest.raises(RuntimeError, match='Could not terminate pod pod-test'):
        launcher.main()
    assert [c.args[0] for c in calls.call_args_list] == ['POST', 'DELETE', 'DELETE', 'DELETE']


def test_termination_retries_transient_error(launcher, monkeypatch):
    calls = MagicMock(side_effect=[RuntimeError('transient'), {}])
    monkeypatch.setattr(launcher, 'api', calls)
    launcher.terminate('fake-key', 'pod-test')
    assert calls.call_count == 2


def test_legacy_download_flag_warns_and_dry_run_is_offline(launcher, args, monkeypatch, capsys):
    args.min_download = 700
    args.dry_run = True
    monkeypatch.setenv('HF_TOKEN', 'secret-fixture-token')
    monkeypatch.setattr(launcher, 'parse_args', lambda: args)
    opened = MagicMock(side_effect=AssertionError('must be offline'))
    monkeypatch.setattr(launcher.urllib.request, 'urlopen', opened)
    assert launcher.main() == 0
    output = capsys.readouterr()
    assert '--min-download is ignored' in output.err
    assert 'secret-fixture-token' not in output.out
    assert json.loads(output.out)['env']['HF_TOKEN'] == '***'
    opened.assert_not_called()


def test_skill_copies_identical():
    for relative in (SCRIPT, Path('fine-tuning/reference/runpod-jobs.md')):
        canonical = (ROOT / '.skills' / relative).read_bytes()
        for tree in ('.claude/skills', '.agents/skills'):
            assert (ROOT / tree / relative).read_bytes() == canonical
