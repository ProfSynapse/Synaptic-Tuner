# Dedicated smoke consumer

This branch is a minimal consuming project, not the engine development branch.
Do not merge it into main or an engine feature branch. Do not touch EHR.

- Read the pinned engine's canonical `.skills/fine-tuning/SKILL.md` and its
  `reference/modal-jobs.md` before any training or chat operation.
- Treat the engine submodule as read-only; make engine changes in the engine
  feature worktree, review them, then deliberately update the published pin.
- Use the existing public TrainingAPI/RunsAPI and bounded run-chat adapter.
  The example in `synaptic-tuner/examples/modal_chat` supplies one-shot
  attempt claims and result persistence, not a complete authenticated host.
- Keep consumer configuration, data, authorities and state here. Private output
  belongs under ignored `.synaptic/`; credential values belong in neither
  source, command arguments, logs nor the example database.
- Use only the dedicated `synaptic-smoke-v1` Modal environment. Never inspect
  or alter old unrelated environments, apps or objects.
- A published gitlink is not a training grant or runtime qualification. Require
  current source/deployment/quote evidence, the final reviewed inference image,
  real authenticated run artifacts, and bounded exact-instance cleanup.
- Never retry an ambiguous submit or erase an attempt claim to resubmit.
  User-approved small smokes share the cumulative $10 approval threshold.
  Stop for materially broader scope, irreversible actions, or spending over it.
