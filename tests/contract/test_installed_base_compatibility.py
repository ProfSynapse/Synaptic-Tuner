"""Portable installed-base consumer; no provider or process-runtime qualification.

Run under isolated Python with --checkout-root and --expectations, or use pytest
with SYNAPTIC_COMPAT_CHECKOUT and SYNAPTIC_COMPAT_EXPECTATIONS. Ordinary source
collection skips these installed-only assertions. The base consumer needs no
pytest or optional SDK/ML/UI dependencies.
"""
from contextlib import contextmanager, redirect_stdout
import hashlib
import importlib
import importlib.resources
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import textwrap

_OPTIONAL = ('modal', 'huggingface_hub', 'docker', 'torch', 'transformers',
             'unsloth', 'vllm', 'sentence_transformers', 'numpy', 'pandas',
             'sklearn', 'rich', 'e2b', 'e2b_code_interpreter', 'mlx', 'mlx_lm')


def _context():
    checkout = os.environ.get('SYNAPTIC_COMPAT_CHECKOUT')
    expectations = os.environ.get('SYNAPTIC_COMPAT_EXPECTATIONS')
    if not checkout or not expectations:
        import pytest
        pytest.skip('requires the explicit installed-base consumer environment')
    return Path(checkout).resolve(), json.loads(Path(expectations).read_text(encoding='utf-8'))


def _installed(module, checkout):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve()), path
    assert not path.is_relative_to(checkout), path


@contextmanager
def _no_effects():
    def forbidden(*args, **kwargs):
        raise AssertionError('base consumer attempted process or network effects')
    before = (subprocess.Popen, socket.socket.connect, socket.socket.connect_ex,
              socket.create_connection, socket.getaddrinfo)
    subprocess.Popen = socket.socket.connect = socket.socket.connect_ex = forbidden
    socket.create_connection = socket.getaddrinfo = forbidden
    try:
        yield
    finally:
        (subprocess.Popen, socket.socket.connect, socket.socket.connect_ex,
         socket.create_connection, socket.getaddrinfo) = before


def test_formal_exports_and_base_origins():
    checkout, _ = _context()
    assert all(importlib.util.find_spec(name) is None for name in _OPTIONAL)
    with _no_effects():
        api = importlib.import_module('synaptic_tuner.api.v1')
        for name in api.__all__:
            assert getattr(api, name) is not None, name
        for name in ('synaptic_tuner', 'synaptic_tuner.api.v1', 'tuner.cli.main',
                     'Evaluator.config_loader', 'SynthChat.config.config_loader',
                     'shared.tool_catalog_resources'):
            _installed(importlib.import_module(name), checkout)
    assert not any(name.partition('.')[0] in _OPTIONAL for name in sys.modules)


def test_installed_schemas_and_catalog_bytes():
    _, expected = _context()
    with _no_effects():
        schemas = importlib.resources.files('schemas')
        actual = {item.name for item in schemas.iterdir() if item.name.endswith('.schema.json')}
        schema_expected = {name.split('/', 1)[1] for name in expected if name.startswith('schemas/')}
        assert actual == schema_expected
        for name, digest in expected.items():
            package, relative = name.split('/', 1)
            raw = importlib.resources.files(package).joinpath(relative).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == digest, name
            assert isinstance(json.loads(raw), dict), name
        from shared.environments.tool_executor import cli_command_catalog
        from shared.validation.parsing.cli_commands import load_cli_command_catalog
        from shared.validation.dataset_validator import load_tool_schemas
        from SynthChat.scripts.embed_tool_schemas import load_tool_schemas as load_embed_catalog
        catalogs = importlib.resources.files('synaptic_tuner').joinpath('runtime')
        with importlib.resources.as_file(catalogs.joinpath('cli-first-tool-schemas.json')) as path:
            assert cli_command_catalog() == load_cli_command_catalog(path)
        assert cli_command_catalog()
        assert load_tool_schemas() == json.loads(catalogs.joinpath('Tools/tool_schemas.json').read_bytes())
        assert load_embed_catalog() == json.loads(catalogs.joinpath('cli-first-tool-schemas.json').read_bytes())


def test_real_installed_configuration_loaders():
    checkout, _ = _context()
    with _no_effects():
        import Evaluator
        import SynthChat
        from Evaluator.config_loader import ConfigLoader as EvaluationConfigLoader
        from SynthChat.generator import ScenarioLoader
        from SynthChat.config.config_loader import ConfigLoader as ScopeConfigLoader
        _installed(Evaluator, checkout)
        _installed(SynthChat, checkout)
        evaluation = EvaluationConfigLoader(Path(Evaluator.__file__).parent / 'config')
        assert evaluation.tool_schema
        assert evaluation.load_all_scenarios(evaluation.load_eval_run().scenarios)
        assert ScenarioLoader(Path(SynthChat.__file__).parent / 'scenarios').list_scenarios()
        assert ScopeConfigLoader().load().scopes
    assert not any(name.partition('.')[0] in _OPTIONAL for name in sys.modules)


def test_help_has_no_optional_import_or_process_network_effect():
    _context()
    child = textwrap.dedent('''
        import importlib.util, runpy, socket, subprocess, sys
        optional = %r
        assert all(importlib.util.find_spec(name) is None for name in optional)
        def forbidden(*args, **kwargs):
            raise AssertionError('installed help attempted process/network effects')
        subprocess.Popen = forbidden
        socket.socket.connect = forbidden
        socket.socket.connect_ex = forbidden
        socket.create_connection = forbidden
        socket.getaddrinfo = forbidden
        sys.argv = [sys.argv[1], '--help']
        try:
            runpy.run_module(sys.argv[0], run_name='__main__')
        except SystemExit as error:
            assert error.code == 0
        else:
            raise AssertionError('help did not exit')
        assert not any(name.partition('.')[0] in optional for name in sys.modules)
        assert 'shared.ui' not in sys.modules
    ''' % (_OPTIONAL,))
    for module in ('tuner.cli.main', 'Evaluator.cli'):
        result = subprocess.run([sys.executable, '-I', '-B', '-c', child, module],
                                capture_output=True, text=True, check=True, timeout=30)
        assert 'usage:' in result.stdout.lower(), module


def test_native_diagnostic_has_no_project_or_compute_effects():
    checkout, _ = _context()
    from tuner.cli.main import main
    _installed(importlib.import_module('tuner.handlers.native_training_observation'), checkout)
    output = io.StringIO()
    with _no_effects(), redirect_stdout(output):
        try:
            main(['doctor', 'native-training', '--json',
                  '--manifest', 'nonexistent-native-doctor/synaptic.yaml'])
        except SystemExit as error:
            assert error.code == 0
        else:
            raise AssertionError('diagnostic did not exit')
    report = json.loads(output.getvalue())
    assert report['status'] == 'DIAGNOSTIC_ONLY'
    assert report['platform']['system']['status'] == 'observed'
    assert report['native_apple_candidate'] == (
        report['platform']['system'].get('value') == 'Darwin'
        and report['platform']['architecture'].get('value', '').lower() == 'arm64'
    )
    assert not any(name.partition('.')[0] in _OPTIONAL for name in sys.modules)


def test_installed_console_and_project_manifest():
    _context()
    from tuner.project.manifest import load_project_manifest
    scratch = Path(os.environ['SYNAPTIC_COMPAT_EXPECTATIONS']).resolve().parent
    # This is an actual console smoke; the effect guards belong to the fresh
    # help children above. No process-runtime/platform execution is exercised.
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        manifest = Path(directory) / 'synaptic.yaml'
        manifest.write_text("schema_version: synaptic-project/v1\n"
                            "project: {id: compatibility-consumer, name: Compatibility consumer}\n"
                            "engine: {requires: '>=1', api: v1}\n", encoding='utf-8')
        assert load_project_manifest(manifest).project_id == 'compatibility-consumer'
        console = Path(sys.executable).parent / ('synaptic.exe' if os.name == 'nt' else 'synaptic')
        result = subprocess.run([str(console), '--project-root', directory,
                                 'capabilities', 'list', '--json'], cwd=directory,
                                capture_output=True, text=True, check=True, timeout=30)
        assert json.loads(result.stdout)['success'] is True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout-root', required=True)
    parser.add_argument('--expectations', required=True)
    args = parser.parse_args()
    os.environ['SYNAPTIC_COMPAT_CHECKOUT'] = args.checkout_root
    os.environ['SYNAPTIC_COMPAT_EXPECTATIONS'] = args.expectations
    for check in (test_formal_exports_and_base_origins, test_installed_schemas_and_catalog_bytes,
                  test_real_installed_configuration_loaders,
                  test_help_has_no_optional_import_or_process_network_effect,
                  test_native_diagnostic_has_no_project_or_compute_effects,
                  test_installed_console_and_project_manifest):
        check()
    print('Installed base compatibility checks passed')
