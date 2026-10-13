"""Fixed -I script entry, resolved from the supervisor's installed module file."""
import sys
from pathlib import Path

# Checkout root or installed site-packages root, never the caller's CWD/PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve(strict=True).parents[2]))
from shared.environments.local_worker import _run_child

if __name__ == '__main__':
    _run_child()
