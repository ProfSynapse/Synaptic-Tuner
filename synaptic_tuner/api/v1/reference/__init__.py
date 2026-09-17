"""Reference implementations of the public v1 host ports and facades.

Location: ``synaptic_tuner/api/v1/reference/__init__.py``.

Everything under this subpackage is an implementation, never a contract. It is
deliberately absent from ``_LAZY_MODULE_ATTRIBUTES`` in ``api/v1/__init__.py``
and from ``_FORMAL_EXPORTS``: like ``api/v1/docker.py`` and ``api/v1/modal.py``
it is importable only by explicit submodule path, so importing
``synaptic_tuner.api.v1`` never loads it and the package import-closure gate
stays intact. A host writes, for example,
``from synaptic_tuner.api.v1.reference.stores import InMemoryDurableRecordStoreV1``.

Modules:
- ``stores.py``: in-memory ``DurableRecordStorePort`` and ``DurableStreamStorePort``.

Later slices add the provider-neutral composition function and the family
reference operations here.
"""

__all__: list[str] = []
