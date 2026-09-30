# Config-first Modal training plan

> Status: APPROVED — Joseph Rosenbaum, 2026-09-22
> Scope: one Qwen 3.5 4B SFT recipe through Modal, saved LoRA adapter, and verified retrieval
> Branch at planning: `feat/submodule-cloud-api-v1-windows` at `193933a5`

## Decision to review

Make `python tuner.py train --job-config <recipe.yaml>` the intended user entry
point for a prepared dataset and a named runtime profile. The command is a
proposed interface; it is not currently runnable for Modal. The user chooses a
recipe and Modal as the provider. Synaptic Tuner resolves and authenticates the
runtime, prepares or reuses private input, plans through the public training API,
and runs the job. The user supplies no image, wheel, repository, registry, or
signing identifier.

This proposal changes one requirement in the
[previously approved packaged-runtime plan](packaged-modal-runtime-training-plan.md):
an externally published final OCI image and user-selected registry/signing
authority need not gate the first Modal training path. A pinned base image and
exact local package inputs can instead be built into a cached Modal image during
controlled runtime preparation. The previous published-OCI release form remains
valid. Existing packaged contracts, worker seams, input preparation, provider
authority, and the `developer_integration` route remain useful. This is a plan
change for approval, not a claim that the new route is qualified.

The immediate goal is one successful optimization run with a recoverable adapter.
Generalized releases for other providers, registry CI, signing policy, and a
broad model catalog can follow evidence from that run. The host continues to own
configuration, credentials, grants, durable coordinator/Foundation records,
prepared inputs, and final artifacts.

## Evidence and starting point

- The existing [Qwen recipe](../../Trainers/recipes/qwen35_4b_32k_prompt_completion.yaml)
  already names `qwen35-sft-v1`, Qwen revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, 32,768 context, a prepared
  `dataset.local_file`, preassigned splits, completion loss, and LoRA rank 32.
  Keep this vocabulary and profile name. The recipe currently selects
  `local_docker` and `dry_run: true`; a Modal run requires explicit
  provider/target adaptation and real optimization.
- The local full-dataset dry run admitted 220 rows, with 181 training and 39
  evaluation rows. Its lineage is
  `toolset-training-artifacts/runs/local_docker/sft/qwen35-4b-32k-prompt-completion/20260921_182421/training_lineage.json`.
  A dry run proves admission/configuration, not optimizer execution or saved
  weights. The reviewed base inventory has 327 pinned distributions, CPython
  3.12.3, Torch 2.11.0, and Transformers 5.17.0; the exact inventory and image
  digest are release inputs, not values to infer from the profile label.
- [Native Modal qualification](../review/modal-native-training-qualification.md)
  proves that an earlier source-backed consumer trained and retrieved five
  artifacts. Its model, runtime, and source contract differ from this Qwen
  profile, so it does not qualify the proposed path or 32K memory fit.
- Modal 1.5.4 image construction is supported by the locally inspected wheel
  and by the existing
  [inference runtime capture](../../scripts/capture_modal_inference_runtime.py#L890),
  which uses `Image.from_registry`, copied local files, build commands, and
  `build(app)`. [Inference transport](../../tuner/execution/providers/modal/inference_transport.py#L575)
  also uses provider image identity. Modal's
  [Image SDK](https://modal.com/docs/sdk/py/latest/Image) and
  [existing-images guide](https://modal.com/docs/guide/existing-images)
  describe these surfaces. This supports a build hypothesis; the training
  worker and deployment still require live proof.
- The present [release contract](../../tuner/runtime/releases.py#L324)
  requires an immutable final OCI reference matching `image_digest` at line
  365. [Modal release deployment](../../tuner/execution/providers/modal/runtime_release_deployment.py#L1042)
  currently pulls that final OCI image. A new build-material form therefore
  requires an explicit versioned contract and adapter change. Substituting the
  base image reference for the final image identity would make the release
  commitment false.
- [TrainingAPI](../../synaptic_tuner/api/v1/training_facade.py#L213) exposes
  `prepare`, while [CoordinatorTrainingService](../../tuner/training/coordinator_service.py#L35)
  and [reference composition](../../synaptic_tuner/api/v1/reference/composition.py#L128)
  do not yet wire that operation into the default host flow.
  [The `train` handler](../../tuner/handlers/train_handler.py#L151) does not
  consume `--job-config`. [Local run handling](../../tuner/handlers/local_run_handler.py#L1437)
  is Docker-specific: its recipe loader is reusable, its launcher is not.

## User and authority flow

1. A recipe loader reads the existing YAML shape. `job.runtime_profile:
   qwen35-sft-v1` resolves an allowlisted profile containing the pinned base
   digest, Python/dependency inventory, compatible model revision, package
   source, and build policy. Recipe fields supply model, dataset, training,
   LoRA, and artifact preferences. Provider is `modal`; no arbitrary executable
   or image reference is accepted from the recipe.
2. The first path consumes the already verified prepared v2 dataset and
   manifest via the existing
   [dataset publication boundary](../../tuner/dataset_prep/publication.py#L1288).
   `dataset.local_file` may locate that authorized prepared pair, but a bare
   arbitrary JSONL file cannot silently gain provenance. Uploads can use the
   existing ingestion/preparation path later. Do not add an ad hoc importer to
   get this smoke through.
3. The host calls `prepare` and reuses
   [input preparation](../../tuner/training/input_preparation.py#L440) to
   produce a stable private content identity. Paths, filenames, prose, and
   upload handles stay outside canonical launch authority.
   [Packaged compilation](../../tuner/training/packaged_compilation.py),
   [packaged boundary](../../tuner/training/packaged_boundary.py), and
   [Modal composition](../../tuner/execution/providers/modal/packaged_composition.py#L130)
   carry the admitted workload and material into existing
   coordinator/Foundation/store ports.
4. The public flow remains `load -> resolve -> plan -> preflight -> start`,
   with preparation before canonical planning. A standalone default composition
   should use the existing generic store ports; a consuming host such as
   Syntunia can inject its implementations. No engine-owned lifecycle database
   or second launch API is needed.
5. The host builds or resolves the exact Modal runtime material, binds the
   returned provider image/deployment identity, stages the prepared dataset,
   and makes one effect-authorized submission. Observation, reconciliation, and
   artifact streaming use the authenticated call/effect identity. Ambiguous
   submission grants no replay. Credentials stay in named provider secrets or
   host-only preparation, never in config, argv, canonical records,
   diagnostics, or the offline trainer child.

## Small runtime-contract change

Add a closed, versioned runtime-material variant beside the existing published-OCI form:

| Material | Canonical release commitment | Provider binding |
| --- | --- | --- |
| Published OCI | Final immutable OCI reference and digest, package and closure, runtime inventory | Exact provider deployment/image facts |
| Modal build material | Pinned base image digest, hashed build recipe and wheel/bootstrap inputs, exact package and worker closure, captured final Python/distribution inventory | Actual returned Modal image ID and deployment facts |

The build-material release is not self-certifying. The host verifies pinned
inputs before build; the Modal preparation captures and checks the final runtime
inventory and package closure; the provider binding commits the actual returned
image/deployment identity to the release. A cached result is reusable only
while these commitments and the current deployment match. Do not fabricate a
final OCI digest or treat the pinned base as the finished runtime.

Review and update the release schema, parser, lock/policy inventory, and tests
together. Keep the old published-OCI variant readable and executable for its
intended route. The older Modal consumer lock is CPython 3.11.14 with a
different Torch stack; changing only its image string or rerunning that consumer
would not establish this CPython 3.12.3 Qwen profile. For this profile, use the
same pinned base and install the exact wheel/bootstrap during cached Modal image
preparation, not inside a training job.

As planned behavior, engine preparation obtains the hash-pinned bootstrap wheels
and builds or reuses the exact engine wheel through the existing workflow. The
user supplies no wheels; each training job performs no dependency resolution,
and this development proof has no external publication prerequisite.

The reviewed base currently has four preexisting `pip check` conflicts. Record
that baseline, compare the derived overlay against it, and reject new conflicts
attributable to the overlay. A blanket clean `pip check` gate would reject the
known base without diagnosing the new material. Keep model weights and private
data out of the image; [model snapshot preparation](../../tuner/execution/providers/modal/model_snapshot.py)
handles exact model revision and private cache identity.

## Delivery sequence and owners

The main orchestrator owns cross-lane coordination and accepts independent handoffs. Parallel work starts only after a brief shared contract decision on the runtime-material variant, recipe mapping, and prepared-input admission. No Git branch change is needed.

1. **Shared contract decision.** Write the minimal schema migration and compatibility rules for both runtime-material variants; identify the exact profile/base and release evidence. Check that no existing lock or current release invariant is silently bypassed. Output is a reviewed contract note and file-level ownership for the two lanes.
2. **A — config and host wiring, Sol backend.** Own the recipe/CLI branch, `TrainingAPI.prepare` operation wiring, default standalone composition through generic stores, and prepared v2 dataset admission. Reuse the recipe fields and profile name above. Unit/contract checks cover wrong profile, wrong model revision, unverified dataset, `dry_run` accidentally enabled, secret-bearing values, and config-to-canonical workload agreement. This lane does not own Modal image creation.
3. **B — cached runtime material and Modal binding, Sol DevOps.** Own the versioned release/schema change, exact pinned build inputs, cached Modal image construction and returned-image binding, compatibility with the existing published-OCI form, and bounded deployment self-check. The build may use Modal's `from_registry`, copied wheel inputs, build commands, `build(app)`, and `from_id` only behind authenticated release preparation. This lane does not own recipe parsing or dataset provenance.
4. **Integration.** Join A and B through existing packaged composition and worker seams. First produce a read-only resolved plan showing profile, dataset identity, Qwen revision, workload digest, runtime material, provider binding requirement, estimated resource request, and artifact policy. A plan command must have no provider mutation. Resolve any contract mismatch before a build or submission.
5. **First Modal build and CPU self-check.** Build the cached image once from pinned inputs, capture its actual inventory, authenticate its provider identity, and run a bounded worker self-check with no training data or GPU. Verify the worker is installed/importable and the deployment uses the intended image. This qualifies construction and dispatch wiring only.
6. **Bounded GPU optimizer smoke.** Use the same Qwen revision, rank-32 adapter, 32K configuration, and prepared dataset contract. Admit all 220 rows, execute one or two actual optimizer steps, save the adapter and exact five-artifact inventory, stream and rehash the archive, then reload the saved adapter for bounded inference. Measure a representative longest sequence and GPU memory without silently truncating the declared context. One successful CPU check or a structurally valid archive cannot stand in for these results.
7. **Full 220-row Modal run.** After smoke acceptance, run the 181-train/39-eval split to completion, verify training/evaluation metrics and artifact identities, retrieve and reload the adapter, and preserve host-owned lineage. Then continue the original eval, GGUF conversion, and chat roadmap as separate qualification steps; no conversion or serving success is inferred from SFT completion.

An independent Sol reviewer should focus on the changed release material/schema, build-to-binding identity, prepared-input authority, secret isolation, and one-submission/no-replay behavior. Reuse accepted audits of unchanged packaged seams; rerun broader gates only when a changed boundary creates a concrete regression risk. No live provider call or paid training occurs in this planning review.

## Acceptance evidence and limits

- The proposed command accepts the existing recipe shape with a Modal target, resolves `qwen35-sft-v1`, and rejects arbitrary runtime/image/source inputs. A local plan shows the exact Qwen revision, prepared v2 dataset identity, split counts, and canonical workload before any provider effect.
- Runtime preparation proves exact base/build/wheel inputs, final package and Python inventory, actual Modal image/deployment identity, and no new dependency conflicts over the recorded base baseline. Published-OCI release behavior and the developer integration route remain covered by compatibility tests.
- CPU self-check proves image creation, installed worker, and bounded dispatch. GPU smoke proves real optimizer metrics, nontruncated representative 32K handling, saved LoRA adapter, authenticated artifact streaming, and successful reload/inference.
- Preserve the five existing artifact roles: `workload.json`, `training_lineage.json`, `training_metrics.json`, `final_model.tar`, and `tokenizer.tar`. The current provider limits are 192 MiB per member and 256 MiB aggregate. Qwen output size is not yet measured; adjust a limit only after observed evidence and reviewed policy, never by silently dropping an artifact.
- The full run proves the 181/39 split, terminal result, metrics, exact inventory, durable host lineage, and a reloaded usable adapter. It does not by itself qualify GGUF, serving, HF Jobs, or RunPod.

Open live questions are the derived builder's Python/runtime compatibility, Modal deployment and worker behavior, 32K GPU fit and cost, and saved adapter size/reload. Treat these as measured gates, not reasons to require an external registry first. This is medium-to-high integration work, likely around 10–15 source/schema files plus focused tests, rather than a one-line CLI change.

## Approval and state transition

Approve this proposal before treating it as the active PACT architecture. On approval, the main orchestrator should dispatch the shared contract decision, then the two nonoverlapping Sol lanes above, then independent review and staged qualification. Update the active PACT ledger and reconcile the previous publication DAG and any vault journal only after approval. Preserve the current branch and the old approved plan as historical evidence; do not silently rewrite its status.

After the workflow succeeds, update the canonical project-local `.skills/fine-tuning`
guidance from the observed command and evidence, then sync its `.agents/skills`
and `.claude/skills` copies. No skill packaging is planned.
