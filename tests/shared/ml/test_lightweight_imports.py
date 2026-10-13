"""Optional ML dependencies must not gate lightweight evaluator/contracts."""
import subprocess
import sys
import textwrap

import pytest


BLOCK_OPTIONAL = '''
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pandas', 'numpy', 'torch', 'sklearn'}:
            raise ModuleNotFoundError('blocked optional dependency: ' + fullname,
                                      name=fullname.split('.')[0])
sys.meta_path.insert(0, BlockOptional())
'''


def run_isolated(code):
    result = subprocess.run([sys.executable, '-c', BLOCK_OPTIONAL + textwrap.dedent(code)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return result


def test_retrieval_public_contract_and_cli_help_without_optional_ml():
    result = run_isolated('''
        import runpy
        import shared.ml
        from shared.ml.retrieval_metrics import QueryResult
        import synaptic_tuner.api.v1
        from synaptic_tuner.api.v1 import EvaluationAPI, TrainingAPI
        assert not any(name in sys.modules for name in ('pandas', 'numpy', 'torch', 'sklearn'))
        import subprocess, socket
        def forbidden(*args, **kwargs):
            raise AssertionError('help attempted process creation or network access')
        subprocess.Popen = forbidden
        socket.socket.connect = forbidden
        socket.create_connection = forbidden
        sys.argv = ['Evaluator.cli', '--help']
        try:
            runpy.run_module('Evaluator.cli', run_name='__main__')
        except SystemExit as error:
            assert error.code == 0
        assert not any(name in sys.modules for name in ('pandas', 'numpy', 'torch', 'sklearn'))
        assert 'shared.ui' not in sys.modules
    ''')
    assert 'usage:' in result.stdout


@pytest.mark.parametrize(('export', 'dependency'), [('load_dataset', 'pandas'), ('compute_metrics', 'numpy')])
def test_requested_optional_export_retains_missing_dependency_error(export, dependency):
    run_isolated(f'''
        import shared.ml
        assert '{export}' in dir(shared.ml) and '{export}' in shared.ml.__all__
        try:
            getattr(shared.ml, '{export}')
        except ModuleNotFoundError as error:
            assert error.name == '{dependency}'
        else:
            raise AssertionError('missing feature dependency was hidden')
        assert '{export}' not in shared.ml.__dict__
        try:
            shared.ml.unknown_feature
        except AttributeError:
            pass
        else:
            raise AssertionError('unknown export accepted')
    ''')


def test_optional_exports_preserve_real_function_identity_when_dependencies_exist():
    pytest.importorskip('pandas')
    pytest.importorskip('numpy')
    from shared.ml import load_dataset, compute_metrics
    from shared.ml.data_loaders import load_dataset as expected_loader
    from shared.ml.metrics import compute_metrics as expected_metrics
    assert load_dataset is expected_loader
    assert compute_metrics is expected_metrics


def test_remote_worker_policy_without_torch_but_gpu_use_requires_it():
    run_isolated("""
        from shared.experiment_tracking.runtime_autotune import recommend_eval_max_workers, cuda_headroom_bytes
        assert recommend_eval_max_workers(backend='vllm', requested_max_workers=0, cpu_count=8) == 4
        assert recommend_eval_max_workers(backend='ollama', requested_max_workers=2, cpu_count=8) == 2
        assert 'torch' not in sys.modules
        try:
            cuda_headroom_bytes()
        except ModuleNotFoundError as error:
            assert error.name == 'torch'
        else:
            raise AssertionError('GPU dependency failure hidden')
    """)


def test_no_dashboard_keeps_final_rich_summary_without_install_or_network():
    pytest.importorskip('rich')
    result = run_isolated("""
        import importlib.abc, tempfile, pathlib, socket, subprocess
        from unittest.mock import patch
        class BlockAnimation(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] == 'asciimatics':
                    raise ModuleNotFoundError('optional animation absent', name='asciimatics')
        sys.meta_path.insert(0, BlockAnimation())
        def forbidden(*args, **kwargs):
            raise AssertionError('evaluation attempted install or network')
        subprocess.Popen = forbidden
        socket.socket.connect = forbidden
        socket.create_connection = forbidden
        import Evaluator.cli as cli
        with tempfile.TemporaryDirectory() as directory:
            output = str(pathlib.Path(directory) / 'results.json')
            with patch.object(cli, 'create_client', return_value=object()), patch('shared.experiment_tracking.registry.RunRegistry'), patch('tuner.project.context.find_git_superproject', return_value=None):
                cli.main(['--backend','llamacpp','--model','model.gguf','--scenario','tool_prompts.yaml',
                    '--limit','1','--dry-run','--no-dashboard','--output',output])
        from shared.ui import animations
        assert animations.ASCIIMATICS_AVAILABLE is False
    """)
    assert 'Schema Pass Rate' in result.stdout
