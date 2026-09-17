"""Reference implementations of the public v1 host ports and facades.

Location: ``synaptic_tuner/api/v1/reference/__init__.py``.

Everything under this subpackage is an implementation, never a contract. It is
deliberately absent from ``_LAZY_MODULE_ATTRIBUTES`` in ``api/v1/__init__.py``
and from ``_FORMAL_EXPORTS``: like ``api/v1/docker.py`` and ``api/v1/modal.py``
it is importable only by explicit submodule path, so importing
``synaptic_tuner.api.v1`` never loads it and the package import-closure gate
stays intact. A host writes
``from synaptic_tuner.api.v1.reference import compose_reference_host``.

This ``__init__`` imports nothing from ``tuner.*``: ``stores.py`` is
contract-only and both import-closure gates prove that importing it loads no
engine module, which requires the package init to stay pure. The composition
names below resolve lazily (PEP 562) from the modules that own them.

Modules:
- ``stores.py``: in-memory ``DurableRecordStorePort`` and ``DurableStreamStorePort``.
- ``provider_family.py``: ``ProviderFamilyV1`` and the provider-neutral
  ``compose_family_coordinator`` extracted from the Modal composition.
- ``repositories.py``: the five coordinator store ports over the host record
  and stream stores, plus the durable foundation effect ledger.
- ``authority.py``: engine authorities derived from one named secret.
- ``training.py``, ``runs.py``, ``artifacts.py``: the three facade
  implementations and the host authorization bridge.
- ``evaluation.py``: the ``EvaluationOperations`` implementation over
  ``Evaluator/`` (local backends only) and its host ports.
- ``composition.py``: ``compose_reference_host`` and ``ReferenceComposition``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_ATTRIBUTES = {
    "ProviderFamilyV1": "provider_family",
    "ReferenceHostPortsV1": "provider_family",
    "ReferenceAuthorityV1": "authority",
    "compose_reference_authority": "authority",
    "ReferenceRequestPortsV1": "training",
    "ReferenceEvaluationPortsV1": "evaluation",
    "EvaluationBackendRegistryV1": "evaluation",
    "LocalHttpEvaluatorBackendV1": "evaluation",
    "UnmeteredPaidBackendV1": "evaluation",
    "DirectoryEvaluationArtifactSinkV1": "evaluation",
    "compose_reference_evaluation": "evaluation",
    "ReferenceComposition": "composition",
    "compose_reference_host": "composition",
    "compose_reference_stores": "composition",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_ATTRIBUTES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module_name}"), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRIBUTES))


__all__ = sorted(_LAZY_ATTRIBUTES)
