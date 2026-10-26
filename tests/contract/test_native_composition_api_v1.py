"""Optional native composition must stay importable without ML or host effects."""

import subprocess
import sys


def test_explicit_native_composition_import_needs_no_ml_storage_or_effects():
    code = r'''
import importlib.abc, socket, subprocess, sys
blocked = {"mlx", "mlx_lm", "torch", "transformers", "modal", "huggingface_hub", "sqlite3"}
class NoOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked:
            raise AssertionError("native composition loaded an optional dependency")
sys.meta_path.insert(0, NoOptional())
def no_effect(*args, **kwargs):
    raise AssertionError("native composition attempted a host effect")
subprocess.Popen = socket.socket.connect = socket.create_connection = no_effect
from synaptic_tuner.api.v1 import native_mlx
assert native_mlx.__all__ and len(native_mlx.__all__) == len(set(native_mlx.__all__))
assert all(getattr(native_mlx, name) is not None for name in native_mlx.__all__)
assert not blocked.intersection(name.partition(".")[0] for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
