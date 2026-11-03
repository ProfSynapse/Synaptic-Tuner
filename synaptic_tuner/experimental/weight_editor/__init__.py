"""Experimental, model-independent weight-editor broker and recorder.

Not a Tuner API v1 promise. The core owns admission, budgets, write-ahead
dispatch, reconciliation and a hash-chained transcript; callers supply the
trusted ``ModelAdapter`` and ``Executor`` ports. See
``docs/experimental/weight-editor.md``.
"""

__version__ = "0.2.0-experimental"
EXPERIMENTAL = True
