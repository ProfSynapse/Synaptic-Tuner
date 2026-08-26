# Synaptic Tuner Runtime-Asset Boundary Inventory

**Status:** Publication gate defined; self-contained wheel publication blocked

**Prepared for:** Submodule-first agent engine, DAG nodes R and J0

**Scope date:** 2026-08-16

**Executable contract:** `tests/contract/test_runtime_assets.py`

## Decision

Synaptic Tuner can expose editable/source-install packaging and the `synaptic`
console without claiming that its wheel is a complete engine installation. A
self-contained wheel must not be published yet.

The current runtime still reads configuration catalogs, schemas, trainer entry
scripts, evaluator assets, SynthChat assets, MechInterp templates, cloud
wrappers, requirements files, and container build contexts from checkout-relative
paths. A wheel containing only importable Python packages can build and install
successfully while omitting those dependencies.

Publication becomes eligible only when all of the following are true:

1. `pyproject.toml` explicitly opts in with
   `[tool.synaptic-tuner.runtime-assets] self-contained-wheel = true`;
2. the runtime resolves assets through an installed-resource boundary rather
   than assuming a source checkout;
3. CI supplies the built wheel through `SYNAPTIC_WHEEL_UNDER_TEST` and runs the
   runtime-asset contract with `SYNAPTIC_REQUIRE_SELF_CONTAINED_WHEEL=1`;
4. every required asset expanded by the contract is present in the wheel at
   either its current checkout-relative path or the reserved
   `synaptic_tuner/runtime/<checkout-relative-path>` layout;
5. an isolated environment with no source checkout on `PYTHONPATH` passes the
   representative console, configuration, capability, local, and cloud dry-run
   smokes from the approved plan.

An ordinary `[build-system]` table, a distribution name, or a successful
`python -m build` is not a self-contained claim. That distinction allows Node B
to land the public facade and editable console without misrepresenting the
runtime boundary.

## J0 Bootstrap Capsule Boundary

Node J0 adds a deliberately separate, unpublished bootstrap artifact. It is
not a wheel and does not change the self-contained-wheel decision above. A
capsule is reusable for one exact committed engine revision and contains only:

- exact Git-object bytes for `tuner/cloud/bootstrap_core.py`;
- exact Git-object bytes for `tuner/cloud/bootstrap_capsule.py`; and
- `synaptic-bootstrap-capsule/v1` manifest bytes binding those two paths,
  their sizes, modes, SHA-256 digests, fixed limits, and the engine commit.

The manifest schema is
`schemas/synaptic-bootstrap-capsule-v1.schema.json`. The schema is an engine
source/runtime asset used to validate the format; it is not embedded as a
third capsule member.

The capsule never contains a source lock, checkout policy, credential,
configuration, plug-in, dataset, prompt, rubric, workload input, or run
output. A later provider launcher transports source-lock and policy JSON as
separate files, binds each file with a launcher-supplied SHA-256 digest, and
passes their paths and digests to the already verified capsule entrypoint.
J0 implements no provider upload, volume, client, authentication, network, or
paid-compute behavior.

After J has independently bound the expected capsule-manifest digest, its
trusted launcher invokes the verified copy with this J0-owned wire contract:

```text
python -I tuner/cloud/bootstrap_capsule.py _run-verified \
  --source-lock <path> --source-lock-sha256 <lowercase-sha256> \
  --checkout-policy <path> --checkout-policy-sha256 <lowercase-sha256> \
  --destination <empty-checkout-root>
```

The two JSON files contain only primitive canonical source-lock and checkout
policy documents. Remote credentials remain opaque environment-backed
references; controlled SSH executable, agent-socket, and known-hosts values
are explicit policy inputs. Local callback resolvers remain adapters in
`tuner/cloud/checkout.py` and are never serialized.

The transport-neutral verifier accepts a capsule root plus an expected
manifest digest. It permits only the exact two regular-file members above,
enforces per-file and aggregate limits, rejects unsafe/duplicate paths and
links or other special files, copies authenticated bytes into per-invocation
private scratch, re-verifies the copies, and cleans scratch on success and
failure. Only the verified copy authenticates the separate external input
bytes and then imports the shared bootstrap core. Local checkout and capsule
checkout therefore execute the same reconstruction implementation.

The builder disables Git replacement-object lookup for every revision, tree,
and blob read, so a requested literal commit cannot be substituted through
`refs/replace`; the shared reconstruction runner independently forces that
same setting for every runtime Git command. Runtime parsing is fail-closed:
the remote entrypoint rejects duplicate JSON keys at any object depth and
requires the complete canonical `synaptic-source-lock/v1` envelope, exact
policy and manifest/member key sets, real JSON booleans, exact bounded JSON
integers for every manifest limit, size, and mode, canonical member order, and
case-insensitively unique submodule paths. Lexical path inspection rejects
POSIX links and Windows reparse points/junctions before capsule reads, scratch
writes, external-input reads, checkout writes, or Git runner calls. Scratch
cleanup failure is itself a failure; when an integrity or execution error is
already active, that primary error remains authoritative and receives a safe
cleanup-failure note.

## Method

The inventory was produced from four complementary checks:

1. Search Python consumers for checkout-derived roots, `__file__` traversal,
   subprocess script paths, default config directories, and Docker contexts.
2. Enumerate tracked non-module assets under `schemas/`, `Trainers/`,
   `Evaluator/`, `SynthChat/`, `MechInterp/`, `configs/`, `Tools/`, and
   `docker/`.
3. Trace the normal CLI handlers for training, local Docker, evaluation,
   generation, MechInterp, model conversion, and cloud launch.
4. Separate assets needed to execute supported capabilities from examples,
   historical material, generated output, private project inputs, and external
   dependencies.

The executable contract intentionally uses an explicit family table. It is not
an exhaustive snapshot of the repository. New runtime families must be added
deliberately, while unrelated documentation and generated artifacts do not
cause churn.

Observed tracked non-Python material was approximately 1.30 MB before wheel
compression, dominated by legacy Evaluator prompt data. The required v1 asset
set is smaller because archived prompts, generated SynthChat outputs, notebooks,
and project data are excluded. Size alone is therefore not the blocker; path
semantics and complete capability coverage are.

## Classification Rules

| Classification | Meaning | Wheel implication |
|---|---|---|
| Package-resource candidate | Declarative data or a template that supported Python code reads at runtime | Must be packaged and resolved with an installed-resource API for a self-contained wheel |
| Engine-checkout-only resource | Executable/helper currently invoked by repository-relative filename, or a source tree copied/mounted as a unit | Must be converted to a packaged module/entry point or explicitly retained in an external runtime checkout |
| Provider/container resource | Build context, bootstrap dependency list, or provider launch template | Must be packaged for the claimed provider capability or supplied by a separately versioned provider/container artifact |
| Optional/deferred | Example, archive, operator convenience file, generated output, or external tool | Excluded from the core wheel gate unless a later capability contract promotes it |
| Host/project owned | Dataset, project config, project plug-in, prompt/rubric override, credential reference, or run output | Must remain outside the engine wheel |

## Package-Resource Candidates

These assets are declarative engine defaults or discoverable catalogs. The
contract expands the listed patterns and requires their source members to exist.
A self-contained wheel must contain every expanded member.

| Family | Repository paths | Consumers | Why it is runtime material |
|---|---|---|---|
| Project contract schemas | `schemas/synaptic-*.schema.json` | `tuner.project.manifest`, config/source-lock validation | Host manifests and resolved/source-lock documents cannot be validated from an installed engine without their schemas |
| Training method catalog | `Trainers/methods.yaml` | cloud/training handlers and display logic | Defines method identifiers and labels outside Python |
| Trainer defaults and registries | `Trainers/{sft,kto,dpo,grpo,embedding,ace_step}/configs/**/*.yaml` | trainer loaders and local/cloud commands | Supplies default trainer settings, tiers, rewards, fitness rules, and model registries |
| Job recipe catalog | `Trainers/recipes/*.yaml` | `local-run`, `cloud-run`, recipe discovery | Config-first executable job definitions; the CLI currently discovers them under the engine root |
| Cloud defaults and experiment catalog | `Trainers/cloud/cloud_config.yaml`, `Trainers/cloud/experiments/*.yaml` | HF Jobs, Modal, RunPod, hardware planning, experiment orchestration | Supplies image profiles, provider settings, dependency overlays, hardware tiers, and checked-in experiment specs |
| Evaluator configuration | `Evaluator/config/**/*.yaml` | evaluator CLI/handlers, cloud evaluation, environment validator | Supplies run presets, display rules, response types, tool/environment schemas, scenarios, rubrics, and templates |
| Evaluator recipes | `Evaluator/recipes/*.yaml` | evaluation recipe discovery | Checked-in declarative evaluation entry points |
| SynthChat core configuration | selected `SynthChat/config/*.yaml` | generation, improve, sanitize, validation, format resolver | Provides defaults, formats, labels, privacy profiles, settings, and validation behavior |
| SynthChat rubric catalog | `SynthChat/rubrics/*.yaml`, `SynthChat/rubrics/*.example` | rubric repository, validators, generation/improvement | Built-in generic judging and data-quality policies |
| SynthChat scenario catalog | `SynthChat/scenarios/*.yaml` | generation and agentic scenario discovery | Built-in generic generation definitions |
| MechInterp templates | `MechInterp/configs/templates/*.yaml`, `*.json`, `*.jsonl` | `mechinterp list-configs`, pipeline and stage loaders | Example pipeline documents are an advertised CLI-discoverable runtime surface |
| Engine workflow presets | `configs/flywheel/*.yaml`, `configs/prompt_optimization/*.yaml`, `configs/lora_surgery.yaml`, `configs/transcript_import/default.yaml` | experiment loop, prompt optimization, surgery, transcript import | Config-first defaults and reusable checked-in workflows |
| Tool schema catalogs | `cli-first-tool-schemas.json`, `Tools/tool_schemas.json` | shared environment executor/validator and schema utilities | The environment executor resolves the root schema by filename |

### Deliberate exclusions within these trees

- `SynthChat/config/targets_*.json` is run/project input, not an engine default.
- `SynthChat/rubrics/archived/**` is historical compatibility material.
- `Evaluator/prompts/archived/**` is legacy prompt data and accounts for most
  of the Evaluator non-code byte size.
- `Trainers/notebooks/**` is educational material, not CLI runtime data.
- Any dataset referenced by a recipe remains host/project input even when the
  recipe itself is an engine example.

## Engine-Checkout-Only Resources

These files are currently run by path or depend on being surrounded by the
source tree. Packaging Python modules alone does not preserve that behavior.

| Family | Current resources | Current coupling | Required resolution |
|---|---|---|---|
| Trainer entry points | `Trainers/sft/train_sft.py`, `kto/train_kto.py`, `dpo/train_dpo.py`, `grpo/train_grpo.py`, `grpo/train_env_grpo.py`, `embedding/train_embedding.py`, `ace_step/train_ace_step.py`, `mlx_sft_mac/train_sft.py` | Local and provider commands construct `Trainers/.../train_*.py`; several mutate `sys.path` to reach adjacent `src/` and the repo root | Convert to packaged callable entry points with resource-backed defaults, or require an exact engine checkout |
| Trainer implementation islands | adjacent `src/**`, `configs/config_loader.py`, `Trainers/shared/**` | Entry scripts import siblings after path injection | Include as real packages or preserve the checkout tree |
| Modal v1 deployment and worker | `tuner/execution/providers/modal/deployment_v1.py`, `remote.py`, `producer.py`, `Trainers/sft/runtime_v1.py` | The fixed deployment reconstructs the exact dual checkout and verifies all packaged runtime-lock digests before invoking SFT | Ship the provider modules and runtime lock, then reconstruct the exact authenticated engine checkout |
| RunPod synchronization | `Trainers/cloud/runpod_sync.py` | RunPod jobs use the checked-out helper and colocated `cloud_config.yaml` | Package with provider adapter or preserve exact checkout |
| Cloud evaluation entry points | `Evaluator/cloud_hf_job.py`, `Evaluator/cloud_hf_job_vllm.py` | HF Jobs selects these import/module entry points after cloning source | Include the complete `Evaluator` package and assets, or preserve exact checkout |
| MechInterp Modal wrapper | `MechInterp/cloud/modal_runner.py` | Handler passes the checkout-relative file to `modal run` | Package/provision as a provider entry point or preserve exact checkout |
| CLI compare helper | `Tools/compare_runs.py` | Router currently executes `Path("Tools/compare_runs.py")` relative to process cwd | Replace with a packaged callable; current behavior is not installed-wheel safe |
| Conversion helpers | `Tools/convert_to_webllm.py`, `scripts/cloud_gguf_convert.py` | Model conversion/cloud commands use checked-in scripts | Package as commands/modules or preserve exact checkout |
| Complete engine source sent to Docker/cloud | current local copy/bind flow and provider Git checkout | Local Docker stages `/workspace/repo`; cloud providers clone the exact pushed commit | Replace with the approved `/workspace/engine` runtime layout before any installed-only claim |

`Tools/split_synthchat_dataset.py` is referenced conditionally by
`tuner/handlers/generate_handler.py` but is not present in the tracked tree. The
handler falls back when absent, so it is not a required asset. It should either
remain an explicitly optional plug-in/helper or be removed from the advertised
installed capability path; it must not be silently listed as packaged.

## Provider and Container Resources

| Family | Repository paths | Consumer/boundary | Decision |
|---|---|---|---|
| MechInterp runner build context | `docker/mechinterp-runner/Dockerfile`, `entrypoint.sh`, `print_provenance.py` | Docker build; the Dockerfile copies both adjacent files | Treat the three files as one versioned provider asset. Missing any member invalidates the context |
| Warm vLLM Space template | `Trainers/cloud/spaces/vllm_warm/Dockerfile.tmpl`, `entrypoint.sh`, `sync_bucket_prefix.py` | `manage_space.py` renders/deploys the template | Package/provision the complete template directory for that capability |
| ACE-STEP image definition | `Trainers/ace_step/Dockerfile`, `requirements.txt` | Purpose-built local/cloud image and dependency island | Provider/container asset; retain exact pins and source provenance |
| Evaluator dependency overlay | `Evaluator/requirements.txt` | Cloud evaluator installs `-r Evaluator/requirements.txt` | Provider bootstrap asset unless dependencies move to an explicit package extra/image |
| Trainer dependency overlays | `Trainers/{sft,kto,dpo,grpo,embedding,mlx_sft_mac}/requirements.txt` | Setup/provider bootstrap | Provider/runtime-profile assets unless replaced by package extras or pinned images |
| Trainer setup launchers | `Trainers/{sft,kto,dpo}/setup.sh`, `Trainers/mlx_sft_mac/run.sh` | Manual/local environment bootstrap | Provider/operator assets, not needed for the minimal console |
| Root dependency overlays | `requirements-cloud.txt`, `requirements-flywheel.txt` | Cloud/flywheel environment provisioning | Provider/runtime-profile assets |

The provider category may ultimately be delivered separately from the Python
wheel, but that artifact must be versioned and selected in provenance. A full
“self-contained Synaptic Tuner wheel” claim cannot omit it while advertising
those provider capabilities.

## Optional, Deferred, and External Assets

| Asset | Classification rationale |
|---|---|
| `run.sh`, `run.ps1`, `setup_env.sh`, `setup_env.ps1`, `Tools/run_*.{sh,ps1}` | Operator convenience wrappers; console/API replacements are preferred for installed use |
| `Trainers/activate_unsloth_latest.*` | Development environment helper, not a deterministic runtime contract |
| `Trainers/notebooks/**` | Educational/interactive examples |
| `Trainers/archive/**`, `Evaluator/prompts/archived/**`, `SynthChat/rubrics/archived/**` | Historical material; package only for an explicit compatibility feature |
| `SynthChat/output/**`, `Evaluator/results/**`, `Evaluator/interactions/**`, `.tracking/**` | Generated state/output; never package |
| `SynthChat/content_200_targets.json`, `targets_essay.json`, `test_content_targets.json`, `SynthChat/config/targets_*.json` | Example or project-specific generation inputs |
| `Trainers/llama.cpp/**` and the `llama-cli` binary | External build/dependency. The tracked repository does not currently vendor this tree |
| Model weights, adapters, datasets, caches, credentials, bucket contents | External or project-owned inputs; record references and hashes, never include secrets or private data in a wheel |
| `.skills/**`, `.agents/skills/**`, `.claude/skills/**` | Agent/operator guidance distributed with source; not Python runtime data |

## Host/Project-Owned Boundary

The following remain outside every engine distribution:

- `synaptic.yaml` belonging to a consumer project;
- project configs, datasets, prompts, rubrics, plug-ins, governance files, and
  private fixtures;
- `.synaptic/artifacts`, `.synaptic/state`, `.synaptic/tracking`,
  `.synaptic/cache`, and `.synaptic/tmp`;
- credentials and credential-bearing URLs;
- explicitly exported, project-reviewed summaries.

The engine may ship generic examples. Execution must distinguish those examples
from host-owned inputs through `ProjectContext` and resolved source metadata.

## Executable Gate Contract

`tests/contract/test_runtime_assets.py` has three layers:

1. **Inventory integrity:** every required family resolves to at least one
   tracked source member, and critical exact members exist.
2. **Boundary integrity:** output/state/private-data paths are not admitted to
   the self-contained wheel manifest.
3. **Publication mode:** an explicit self-contained claim requires an actual
   `.whl`; every required member must exist in that archive. CI can force this
   mode even when the claim is absent, which fails closed.

Source/editable verification:

```text
python -m pytest tests/contract/test_runtime_assets.py -q
```

Future publication verification:

```text
SYNAPTIC_REQUIRE_SELF_CONTAINED_WHEEL=1 \
SYNAPTIC_WHEEL_UNDER_TEST=dist/synaptic_tuner-<version>-py3-none-any.whl \
python -m pytest tests/contract/test_runtime_assets.py -q
```

PowerShell uses `$env:SYNAPTIC_REQUIRE_SELF_CONTAINED_WHEEL = "1"` and
`$env:SYNAPTIC_WHEEL_UNDER_TEST = "..."` before the same pytest command.

The accepted installed layout is intentionally narrow:

- the current repository-relative member path, for assets inside installed
  top-level packages; or
- `synaptic_tuner/runtime/<repository-relative-member>` for a consolidated
  resource bundle.

Supporting arbitrary wheel paths would make the test pass without proving the
runtime knows where to find the files.

## Current Gate Result

| Claim | Result |
|---|---|
| Editable/source console support | Allowed, subject to its separate contract test |
| Build sdist/wheel for inspection | Allowed; do not publish |
| Self-contained wheel | Blocked until explicit claim, installed-resource resolution, real-wheel asset proof, and isolated smokes pass |
| External-runtime-checkout distribution | Architecturally allowed later, but it must declare and verify the exact runtime checkout rather than call itself self-contained |
| Provider/container distribution | Allowed only as a separately versioned, provenance-recorded artifact with complete build/template contexts |
| J0 committed bootstrap capsule | Allowed as a local deterministic build artifact; unpublished, code-only, provider-neutral, and not a self-contained wheel claim |

## Follow-Up Owners

- **Node B:** keep editable/source packaging separate from the self-contained
  marker; do not opt in during the public-facade commit.
- **Nodes F/G:** route discovery, trainer, Evaluator, SynthChat, and MechInterp
  assets through `ProjectContext` and explicit engine-resource resolution.
- **Node J0:** retain the exact two-member committed capsule and keep per-run
  inputs external and hash-bound; do not opt into wheel publication.
- **Nodes J/K/L:** reconstruct `/workspace/engine` and provider assets from
  exact source locks; do not assume `/workspace/repo` is both host and engine.
- **Node N:** build a wheel without publishing, run this test in forced
  publication mode, install it in a fresh environment, and keep publication
  absent until all acceptance gates are green.
