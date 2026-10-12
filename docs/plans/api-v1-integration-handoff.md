# API v1 integration handoff

The integration branch is `feat/submodule-cloud-api-v1`. The Windows working
branch is `feat/submodule-cloud-api-v1-windows`; its reviewed work is being
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

## Metrics: implemented is not live-qualified

See [training-progress-streaming.md](training-progress-streaming.md) for the
existing numeric observation contract, signed transport and limitations.

- The standalone host now admits the shared `PROGRESS_PURPOSE` for verification
  of remote signed progress. Its regression test also rejects unknown signing
  purposes. This does not enable the provider logs capability.
- Modal's public logs capability remains disabled.
- The standalone launch loop still waits for provider completion; it does not
  yet poll public log pages and persist numeric records locally while pending.
- A live proof of pre-completion visibility, including background Volume
  commits and partial-record handling, remains outstanding.
- Optional telemetry is best-effort; a missing tail is not evidence of zero
  loss, job completion, or permission to replay training.

The next implementation should reuse the existing public logs API and typed
observations. Retain cursors correctly across temporarily empty pages, dedupe
by run and sequence with canonical-byte comparison, and save per-run records
for later analysis. State restart/reconnection limitations honestly. Do not
expose manuscript text, credentials, or arbitrary remote exception text.

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
