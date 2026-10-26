# API v1 integration handoff

## Recovered continuation — 2026-10-26

The previous `api-v1-coordination-20261020` worktree is absent; its Git branch
and MLX branches survived. Work resumes on `resume/api-v1-mlx-20261026` from
the latest MLX consumer source. Bounded raw-text and assistant-only
prompt/completion runs on the actual M4 now pass public training, native
verification and five-artifact delivery; see
the [exact source and evidence](../review/native-mlx-sft-qualification.md).
This does not open Syntunia's native capability. Release branch integration,
final package/CI review and product-owned training lifecycle joins remain.
The older handoff below retains its dated scope and historical evidence.

The integration branch is `feat/submodule-cloud-api-v1`. The Windows working
branch is `feat/submodule-cloud-api-v1-windows`; its reviewed work was
merged into the integration branch. Read the actual branch and worktree state
before continuing. This note records the handoff scope, not a production
readiness certification.

## Current priority

Prepare the provider-neutral API v1 for production use by consuming
applications, including Syntunia. First complete and qualify live training
metrics through the supported API. Then assess the normal consumer journey:
configuration, validation, submission, progress, artifact retrieval, error
handling and lifecycle ownership. Distinguish proven behavior from implemented
but unqualified paths and genuinely missing capabilities.

The proposed ending-outline dataset transformation and four-model training
comparison are paused. Do not rebuild private datasets or launch those full
training runs as part of this handoff. Paid cloud qualification needs an
explicit scope and approval. No live training was started for the publication
of this handoff.

## Approved qualification attempt: blocked by local build inputs

The separately approved `metrics-20261012-01` attempt on source `9c7392f2`
failed before image build, integrated CPU qualification or GPU training. The
first invocation rejected local state-root mode 0775 before consuming authority;
after correction to 0700, the next invocation consumed the metrics and build
claims and failed with closed `BUILD_INPUTS / INVALID` at
`runtime_build.prepare_build_inputs`. Three Volumes, two Secrets and the build
App had already been provisioned. This was a failed approved attempt with
provider effects, not a provider-free preflight. No numeric delivery evidence
or transport/capability promotion resulted. Read-only exact-name lookup later
observed the retained build App stopped with zero tasks; no cleanup mutation
was performed.

The exact local cause was reproduced: all nine hash-pinned bootstrap wheel
files under `Trainers/image_profiles/qwen35_4b_packaged_sft_3360351c` were absent;
the first required wheel read failed at that input boundary. Installing the
37-package launcher closure does not provide these build wheel artifacts.
At that failed source, the derived-image workflow checked/copied artifacts already
present; no reusable checked-in bootstrap preparation surface was available.

The repair adds automatic exact hash-pinned bootstrap acquisition and local
source-wheel/input validation on the normal standalone path before credentials,
claims or provisioning. Optional
`train --job-config <recipe> --prepare-local --json` runs the same local preparation without provider effects; see the
[local preparation workflow](../../.skills/fine-tuning/reference/modal-jobs.md#standalone-local-preparation-and-cold-cache).
`--plan` reports cache readiness without network access. Launcher pins remain
POSIX CPython 3.11.14 and Modal SDK 1.5.4. This implementation description does
not establish the two successful live runs required for reliability acceptance.
Retain the consumed claims and existing resources;
this failure grants no replay, deletion or adoption authority. The user subsequently authorized bounded fix/test/live iteration for this session;
fresh attempts retain distinct reviewed source/admission and claims after the
preparation/preflight gap is resolved. This authority does not replay the consumed
attempt or authorize resource cleanup. See the exact
[attempt checkpoint](training-progress-streaming.md#approved-attempt-checkpoint-2026-10-12).

## Local preparation and second attempt

Cold and warm local preparation passed on source `840949af`, verifying all
nine pinned bootstrap wheels with unchanged attempt journal/catalog and no
provider effects. The retained `local-preparation-proof-20261012` evidence
establishes this local behavior. See the
[preparation and second-attempt record](training-progress-streaming.md#preparation-and-second-attempt-evidence-2026-10-12)
for the exact source and admission.

`metrics-20261012-02` passed build and integrated CPU qualification. Its final
ledger retained 32 numeric records plus one exhaustion record across nine
growing pages, each read bracketed by pending observations of the exact call.
It subsequently failed at `RUN_WORKER_SFT_ARTIFACT`; native artifact verification
was not completed. The result remains `inconclusive` without acknowledged
durable qualification publication. It is not one of the two required successful
same-code runs. Preserve the consumed claims/resources and use a reviewed distinct
admission after a confirmed repair; ordinary logs remain disabled.

The real captured-execution/native-verifier path reproduced a child-transport
digest mismatch: final verification had omitted the optional metric descriptor.
The repair authenticates complete saved canonical child bytes, reconstructs the
strict optional descriptor and retains full evidence equality. This reproduced
bug is consistent with the closed attempt-02 stage, without asserting a remote
exception that was not retained. Provider-free tests and independent review
passed with preserved runtime pins. The repaired source `01383fa8`
subsequently completed attempt 03; its restricted delivery evidence is recorded
[here](training-progress-streaming.md#accepted-restricted-delivery-evidence-2026-10-12).
Attempts 03 and 04 subsequently passed independent audit on the same unchanged
`01383fa81c8ed3847903095456e9b80350cd28a6` source. Both delivered 32 numeric
records before completion with acknowledged durable reports and five verified
native artifacts. This meets the bounded fixture's two-run acceptance condition;
this later documentation-only commit does not change the source that was tested.

## Metrics: restricted GPU evidence and conditional CPU-backed logs

See [training-progress-streaming.md](training-progress-streaming.md) for the
existing numeric observation contract, signed transport and limitations.

- The standalone host now admits the shared `PROGRESS_PURPOSE` for verification
  of remote signed progress. Its regression test also rejects unknown signing
  purposes. This does not enable the provider logs capability.
- Modal logs are enabled only by an exact matching, durably retained reader-issued CPU transport proof; missing, invalid or inconclusive proof keeps logs disabled.
- The standalone launch loop consumes public log pages while pending only when
  capture is configured and the selected provider advertises logs. Modal
  uses that consumer only when the per-call CPU proof passes the composition gate.
  The consumer retains the last non-null cursor, checks canonical bytes on
  duplicate run/sequence records and writes private per-run JSONL evidence.
- Attempts 03 and 04 completed independently accepted restricted delivery proofs
  on the same source with acknowledged durable publication and native artifact
  verification on historical source `01383fa8`. The CPU SDK pair below separately qualifies bounded complete-byte transport; live partial-record behavior remains unqualified.
- Optional telemetry is best-effort; a missing tail is not evidence of zero
  loss, job completion, or permission to replay training.

The polling consumer uses existing typed observations and public logs. Its
retained cursor and deduplication state are process-local; JSONL retention does
not restore the coordinator/Foundation workflow or authorize a reconnect/replay.
Transient read failures retain the cursor and retry; permanent validation,
conflict, storage or local-budget failures disable polling independently of
training. A failed local write/fsync can leave a valid prefix and a partial
trailing line; there is no automatic repair or reopen. No manuscript
text, credentials or arbitrary remote exception text belongs in this lane.

The [bounded qualification proposal](training-progress-streaming.md#restricted-live-qualification-workflow-2026-10-12)
now describes the restricted `train --qualify-metrics` operation and its exact
canonical admission. Ordinary composition remains disabled. Its synthetic fixture
and provider-free checks establish local readiness for review, not live transport
qualification or paid approval. Existing CPU and native-training modes retain
their narrower purpose.

The current standalone route creates nonce-unique apps and Volumes for each
attempt. The fixed-name restrictions of the older minimal `examples/modal_chat`
consumer do not establish a collision in this route.

## Other retained work

The config-only [local Gemma serving example](../../examples/model_chat/local_vllm/gemma4/README.md)
retains exact public model revisions, immutable serving image and observed
package inventory. It records bounded local long-context generation trials,
not Gemma training qualification, full-window saturation, reliability or prose
quality. Do not treat that QAT serving profile as a BF16 training profile.

Private datasets, source manuscripts, attempt journals, experiment responses,
credentials and local research notes are not included in this branch handoff.
A fresh checkout can inspect the implementation and run provider-free tests;
it does not automatically have those private artifacts or live host authority.

## Working rules

Read `AGENTS.md` and the relevant canonical `.skills/` documentation. Preserve
unrelated edits, use existing CLIs and adapters, and keep runtime/model choices
configuration-driven. Diagnose with exact retained evidence and targeted tests.
Do not invent new launcher or lifecycle infrastructure where an existing
contract suffices. Merge readiness is distinct from production qualification.

## Consumer journey assessment (2026-10-12)

These judgments describe the checked-in API and standalone reference host, not
Syntunia product integration or universal provider readiness. Test references
are provider-free evidence unless the row identifies retained live qualification.

| Boundary | Implemented/tested evidence | Qualification or product gap |
|---|---|---|
| Configuration | Public `TrainingAPI.load/resolve/plan`; closed Modal recipe/profile mapping and opt-in metric capture. `test_public_training_api_v1.py`, `test_modal_host_requests.py`, `test_modal_standalone_runner.py`, `test_observation_capture_config.py`. | Host supplies source/data/profile/credentials and immutable prepared input. A fresh checkout has no private live authority; candidate model profiles require their own qualification. |
| Validation | Canonical contracts, source and exact runtime/image binding, prepared-data admission, preflight and unstartable-preflight rejection. | `--plan` is provider-free; `--quote` observes current scoped GPU rates; `--qualify` executes a CPU package proof. None proves GPU/model compatibility, all-in billing bounds or metric delivery. |
| Submission | Public coordinator start, one-shot durable claim before effects, retained identity and uncertain-response rejection; `test_modal_host_effects.py` and standalone fake transcript. Native attempt H is retained live training evidence for its exact source/configuration. | New source changes require current release/CPU checks. No automatic replay or general restart restoration. The host owns durable grants and full coordinator/Foundation persistence. |
| Progress | Signed numeric capture and bounded authenticated public projection; `test_modal_packaged_progress.py`, `test_modal_host_progress.py`. Capability-gated consumer retains local records while pending. | Modal `logs` requires exact matching per-call CPU proof; absent/invalid/inconclusive proof is false. GPU attempts 03/04 qualify restricted delivery on `01383fa8`; CPU attempts 05/06 separately qualify bounded SDK complete-byte delivery on `9252e924`. General transport and partial-write behavior remain unqualified. Missing tails do not prove completion or freshness. Browser charts and subscriptions belong to the product host. |
| Artifacts | `RunsAPI.verify/artifacts/stream`; exact five-artifact terminal inventory, bounded chunking, hash checks and create-only local publication; `test_modal_host_reader.py`, `test_modal_host_download.py`. Native H qualified observe/artifact streaming for its exact attempt. | New source/runtime combinations need revalidation. A local download failure never permits another training submission. Sanitized substitutes are test fixtures, not the observed artifact bytes. |
| Errors | Closed phase/class/location projections and bounded known identities; hostile exception suppression and preserved consumed claims have regression coverage in `test_modal_host_diagnostic_projection.py` and standalone tests. | Phase evidence locates a boundary; it does not establish cause or grant retry. Product hosts must map these codes into actionable UI while preserving exact diagnostic evidence. |
| Lifecycle | Generic coordinator/Foundation and consumer-owned catalogs; public observe/outcome/verification composition. | Modal cancel/reconcile/cost-quote remain disabled. Standalone journals prevent replay but do not restore all in-memory workflow state. Multi-user service persistence, browser sessions and product lifecycle UX are host integration work. |

The bounded reliability cycle is complete for the reviewed synthetic fixture:
automatic pinned preparation passed cold/warm local checks, the captured-child
verification defect was reproduced and repaired, and two independently accepted
live runs completed on the same exact source. Preserve every failed and successful
claim and its resources. Continue provider-free consumer integration while keeping
external provider limitations and broader production requirements explicit. A pre-completion
delivery result alone does not qualify partial writes or promote public logs;
retain that separate live transport/review gate.
Dataset transformation and four-model comparison remain paused.

## CPU SDK transport pair (2026-10-12)

CPU attempts 05/06 passed on unchanged source
`9252e924985b7bf41430e57eb49429f4501b59bf`: each delivered eight synthetic numeric
records plus one fixture status while the exact call was pending and the
terminal receipt absent, then verified the create-only collision and exact final
inventory after completion/control commit. This CPU-only evidence does not
requalify GPU training or native artifacts on that source. See the
[exact CPU evidence](training-progress-streaming.md#accepted-cpu-sdk-transport-evidence-2026-10-12).
Ordinary logs remain conditional on each run's matching opaque proof and durable
capture; historical successes or retained JSON do not confer global authority.
