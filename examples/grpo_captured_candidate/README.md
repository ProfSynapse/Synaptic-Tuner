# Captured GRPO candidate

This is an unqualified, three-step Modal GRPO candidate. The named runtime
profile retains the **396-distribution CPU base observation**, not a fabricated
augmented-image inventory. No GPU, model compatibility, parameter update,
reward variation or writing-quality qualification is implied.

The base capture is `33bb4fa0eec173dda12882a5c5d34c5d643f2b92f1dc32addd3a6009de879ac7`,
from reviewed source `bde909346278d9ec4399a8244f5948440c4b5d8f` and builder
`2025.06`. It used script-copy-only derived image inspection; registry-base byte
identity was not attested. Actual CPython 3.11.14, its executable hash, library
roots, Torch 2.9.0+cu128/CUDA build 12.8, TRL 0.24.0 and the full inventory were
observed. Torch was imported for its CUDA build metadata; no model was loaded.
The original CLI exited zero with empty stderr. That exact source refuses success
unless termination is requested and polling confirms cleanup; this is not an
independent subsequent provider cleanup observation.

The build profile is host/source-owned and excluded from its referenced wheel.
Its real accepted wheel anchor is
`30d5f1b1b92f6826c2b624ca88daa32d0e8437595adfd93f689ae813d7fad863`.
The normal current-source path builds the clean selected source and replaces this
anchor with the **actual new wheel digest** in the authenticated build material.
It does not pretend to know a future wheel hash. Existing pinned bootstrap wheels
are reused. They replace base h2/hpack and add Modal dependencies; the final image
needs its own complete observed inventory and CPU GRPO qualification. The base
inventory must never be relabeled as that final observation.

## Prepare the actual example publication

The checked-in `rows.jsonl` contains four explicit user-message prompts. The reward
is the configured response regex for the word blue. It scores actual returned
completions once. Some or all groups may have equal rewards; the configuration
does **not** guarantee a parameter change or reward variation.

The GRPO publication extension is a host Python API; `prepare-dataset` is not a
GRPO CLI shortcut. Use the existing preparation service, not a copied normalizer.
Create a private consuming project and copy the declared source at 0600:

```sh
install -d -m 700 /absolute/project/private-grpo-source
install -d -m 700 /absolute/project/.tracking /absolute/project/.tracking/datasets
install -m 600 examples/grpo_captured_candidate/rows.jsonl /absolute/project/private-grpo-source/rows.jsonl
install -m 600 examples/grpo_captured_candidate/recipe.yaml /absolute/project/recipe.yaml
```

With the installed engine and the exact declared source bytes:

```python
from pathlib import Path
import yaml
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_sources import LocalTrainingInputPathV1, TrainingPreparationConfigV1
from tuner.training.grpo_preparation import GRPONormalizerConfigV1, GRPOTrainingInputNormalizerV1, GRPOPreparedInputFormatVerifierV1
from tuner.training.input_preparation import TrainingInputPreparationServiceV1, PosixPrivatePreparedRootAuthorityV1

project = Path('/absolute/project')
source = project / 'private-grpo-source'
recipe = yaml.safe_load((project / 'recipe.yaml').read_text())
public = TrainingInputV1.from_dict(recipe['training_input'])
normalizer = GRPOTrainingInputNormalizerV1(allowed_source_roots=(source,))
config = GRPONormalizerConfigV1('rows.jsonl', 'prompt')
verifier = GRPOPreparedInputFormatVerifierV1()
service = TrainingInputPreparationServiceV1(
    prepared_root=project / '.tracking/datasets',
    normalizers={config.normalizer_ref: normalizer},
    format_verifiers={verifier.format: verifier},
    root_authority=PosixPrivatePreparedRootAuthorityV1(),
)
prepared = service.prepare(
    LocalTrainingInputPathV1(source),
    TrainingPreparationConfigV1('candidate-request', 'candidate-project', public.canonical_json(), config),
)
assert prepared.prepared.identity.ref == public.dataset.ref
assert prepared.prepared.identity.revision == recipe['dataset']['expected_digest']
prepared.retained_source.open_lease().close()
```

`preparation.json` records exact source SHA/size and the independently recomputed
semantic publication identity. These are different commitments. The normalizer
preserves original JSONL bytes and publishes under the recipe's content-addressed
`.tracking/datasets/dataset-<semantic digest>` directory. Changing prompts requires
a fresh publication and recipe commitment; do not silently repair hashes.

## Normal planning and qualification boundary

Use the named profile through the existing normal path:

```sh
python tuner.py train --job-config /absolute/project/recipe.yaml --project-root /absolute/project --prepare-local --json
python tuner.py train --job-config /absolute/project/recipe.yaml --project-root /absolute/project --plan --json
```

Local preparation may fetch pinned bootstrap/source-wheel inputs when absent;
review those downloads separately. A preplan is core/build intent, not a final
runtime release. The selected engine source must be clean and match the runtime
policy. The effectful launcher requires the existing hash-pinned CPython 3.11.14 /
Modal 1.5.4 environment. Use the explicit nontransforming builder setting
`MODAL_IMAGE_BUILDER_VERSION=2025.06` for the command process.

Image build/CPU qualification (`--qualify`) and later GPU execution require
separate reviewed exact source, account/environment, fresh attempt, resource and
cleanup choices. None is authorized by this example. **GRPO quote flags are
unsupported**; use the existing read-only provider rate workflow with the exact
recipe resources when preparing an effect plan. L40S/900 seconds and 100 USD minor
units are candidate configuration choices, not current price evidence or an
invoice cap. The child-only 300-second deadline starts after model preparation;
it does not bound image build or the complete provider job.

The optional G4 policy bounds the entire supported trainable population and
actual reward values (16 MiB, 256 tensors, 4096 values), with 1 MiB transfer chunks.
Native verified completion is distinct from `changed_with_grouped_variation`.
Unchanged parameters, missing grouped evidence or zero variation remain truthful
`inconclusive` without automatic retry or promotion. No raw weights/reward vectors
are retained as measurement records. The model/tokenizer revisions are identical
by declaration and must be independently verified on the execution machine.
