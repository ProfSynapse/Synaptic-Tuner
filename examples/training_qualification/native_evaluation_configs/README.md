# Native evaluation examples

The [bounded M4 comparison](../../../docs/review/native-mlx-evaluation-qualification.md)
records 40 cases each for the pinned base and two saved adapters on `edc4c7aa`.
All 120 generations stopped naturally. Reference loss improved on each adapter's
own dataset; all three models passed 0/20 historical command checks. Semantic
quality and current Nexus API behavior remain unqualified.
Read [the contract](../../../docs/plans/native-mlx-evaluation.md) before use.
All example placeholders require actual reviewed private paths/digests; no
example admission or permissive execution authority is supplied.

`preparation.example.json` selects both existing immutable publications and all
20 validation rows from each. It exports 40 identical ordered prompts for the
base and each saved adapter, preserving previous turns and separating final
targets. The conversation assertion measures a nonempty completion boundary,
not semantic quality. The Nexus assertions measure exact historical tool name
and command-string fidelity; equivalent quoting may fail, and current Nexus
API correctness is not established. Reference loss is reported separately.

Create private byte-identical publication copies (0400 sealed originals or 0600
copies inside 0700 roots), leaving canonical originals untouched. Do not chmod
shared cache ancestors. Copy the preparation config to 0600, replace its paths
and choose a new private output root. The existing host canonicalizer can write
a create-only canonical 0600 copy with `--destination`; this also applies to the
host/scoring configs and unsigned admission candidate. It performs no ML.
Then, with an existing CPU environment:

```sh
python -B examples/training_qualification/native_host.py canonicalize /PRIVATE/preparation-edited.json --destination /PRIVATE/preparation.json
python -B examples/training_qualification/native_evaluation_cases.py prepare /PRIVATE/preparation.json
python tuner.py check-contamination --train-data /PRIVATE/PREPARED-CASES/training-for-contamination.jsonl --eval-text /PRIVATE/PREPARED-CASES/heldout-prompts.jsonl --report-dir /PRIVATE/NEW-CONTAMINATION-REPORT --json
```

Retain and inspect contamination findings; do not silently remove data after
training to make scores appear independent. The existing profiles show held-out
references up to 48 supervised tokens for conversations and 1,329 for Nexus.
The explicit 1,536 completion-token cap covers those references; it does not
guarantee natural completion. The 4,096 context and 900-second worker bounds are
finite proposals, not a speed/memory qualification. Review before execution.

`host.example.json` declares a pinned base-model profile. An adapter selection
uses `adapter_file: {"path": "/PRIVATE/final_model.artifact", "sha256": "ACTUAL_HASH"}`
for the original verified native model role. The host checks archive members,
base/model binding, adapter configuration and finite tensors. It stages only
its own new private copy after authority is verified. It never merges weights,
downloads a model, changes a package environment or submits training.

From exact reviewed source on the Mac, using its existing private interpreter:

```sh
python -B examples/training_qualification/native_evaluation.py inspect /PRIVATE/host.json
python -B examples/training_qualification/native_evaluation.py admission-candidate /PRIVATE/host.json --expires-at REVIEWED_FRESH_UTC
```

Canonicalize and independently review the returned candidate, selection and
current host clock, then sign its exact bytes with the original operator key
outside the Mac. Only the canonical admission and detached signature are copied
to their new private paths. The private key is never transferred. Execution
requires an externally supplied public-key pin as well as the matching config:

```sh
python -B examples/training_qualification/native_evaluation.py execute /PRIVATE/host.json --trusted-root-key-sha256 EXTERNAL_PIN
python -B examples/training_qualification/native_evaluation.py readback /PRIVATE/host.json --trusted-root-key-sha256 EXTERNAL_PIN
python -B examples/training_qualification/native_evaluation.py export /PRIVATE/host.json --trusted-root-key-sha256 EXTERNAL_PIN --destination /PRIVATE/NEW-EXPORT
```

Use separate fresh attempt/state/admission identities for the base and each
adapter. A failed or uncertain attempt remains consumed; no retry/adoption is
implicit. Readback/export can use original authority after start expiry. Copy
the complete private export to the CPU host with unchanged bytes and modes.
It contains public signature material, cases, ordered outputs and a host receipt;
it does not contain weights, secrets or a portable provider execution attestation.
Its manifest digest must match the operator's original export observation.

Populate `scoring.example.json` with that exact export/scoring digest, label and
explicit validator declarations. Scoring runs in the existing CPU evaluator
environment, so the Mac MLX runtime does not need evaluator dependencies:

```sh
python -B examples/training_qualification/native_evaluation_cases.py score /PRIVATE/scoring.json --trusted-root-key-sha256 EXTERNAL_PIN
python -m Evaluator.compare --reference /PRIVATE/BASE/evaluation_results.json --candidate everyday=/PRIVATE/EVERYDAY/evaluation_results.json --candidate nexus=/PRIVATE/NEXUS/evaluation_results.json --output /PRIVATE/comparison.json --markdown /PRIVATE/comparison.md
```

The existing evaluator consumes exact retained prompts/responses with no tool,
model, network or judge execution during scoring. Admission and artifact
integrity checks do not independently attest model execution. Comparison bases
describe equivalent scoring inputs and controls, not model authenticity or
quality. Report length limits, failures, contamination, historical-format
fidelity and token-weighted reference loss separately. Retain complete private
generations for review before drawing task-quality conclusions.
