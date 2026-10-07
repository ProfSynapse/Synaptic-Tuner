# Modal bootstrap closure audit

This provider-free contract test audits the candidate coordinator deployment's
image-baked engine closure against the checked-in Modal runtime lock. It is an
assertion surface, not a lock generator: it never edits the inventory, hashes,
or source tree.

The bounded static walker starts at `coordinator_deployment.py`, follows owned
`tuner` and `synaptic_tuner` imports at every lexical scope, resolves relative
`from` aliases such as `from . import worker_source`, and includes each package
initializer that Python executes while importing a submodule. Imports guarded
solely by `TYPE_CHECKING` are excluded. Function-local runtime imports remain
included, notably model snapshot preparation. The packaged offline-worker
manifest is asserted separately because it is loaded as package data rather
than through Python import syntax. Two reviewed lazy-symbol targets are also
declared explicitly: `tuner.training.TrainingService` resolves through the
package export map to `tuner/training/service.py`, which imports
`tuner/training/resolution.py`.

This is not generic dynamic-import discovery. It does not prove arbitrary
data-driven `import_module` calls, plugins, entry points, or reflective module
selection. After the rich-contract move, the reviewed candidate covers 91
Python files plus the one offline-worker manifest resource. Tests make omission of either explicit lazy
target fail rather than silently reducing that bounded set.

The lock must cover every resulting source/resource path with unique explicit
members, bind `deployment_wrapper` to the candidate coordinator deployment,
and omit the replaced legacy `remote.py` and `producer.py` worker entries. A
selection whose wrapper digest changes must fail the existing runtime-lock
validator.

The fixed declaration has 97 unique members: those 92 bootstrap inputs, the
launcher dependency lock and SFT runtime entrypoint, and three conservatively
retained public API integrity pins (`synaptic_tuner/api/v1/context.py`,
`execution.py`, `sources.py`). These three are not claimed as currently reached
bootstrap imports. Removing them was declined by automated safety review; their
coverage is retained rather than narrowed. The schema, runtime policy and
maintenance map agree on every key and exact path.

This bootstrap closure is not the separately authenticated 66-member offline
trainer closure. Some paths intentionally overlap, but the former describes
code baked into the coordinator image while the latter describes the exact
engine/trainer tree cloned and retained for child execution. Neither audit is
live Modal, credential, training, model-quality, or cloud qualification.

The original ten-member lock failed this audit. The lead updated the lock
schema, explicit inventory map, hashes and wrapper digest together; the static
audit remains a separate check from exact file hashes and provider execution.
