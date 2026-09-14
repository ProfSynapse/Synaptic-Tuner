# Dedicated smoke consumer

This branch is a minimal consuming project, not the engine development branch.
Do not merge it into main or an engine feature branch. Do not touch EHR.

- Read the pinned engine's canonical `.skills/fine-tuning/SKILL.md` and its
  `reference/modal-jobs.md` before any training or chat operation.
- Treat the engine submodule as read-only; make engine changes in the engine
  feature worktree, review them, then deliberately update the published pin.
- Use the existing public TrainingAPI/RunsAPI and bounded run-chat adapter.
  The example in `synaptic-tuner/examples/modal_chat` supplies one-shot
  authenticated host composition, permanent attempt claims and result persistence.
  Use its checked-in `launch.py`; do not add a throwaway cloud launcher.
- Keep consumer configuration, data, authorities and state here. Private output
  belongs under ignored `.synaptic/`; credential values belong in neither
  source, command arguments, logs nor the example database.
- Use only the dedicated `synaptic-smoke-v1` Modal environment. Never inspect
  or alter old unrelated environments, apps or objects.
- A published gitlink is not a training grant or runtime qualification. Require
  current source/deployment/quote evidence. `qualify-training` records native
  authenticated training and artifact evidence without claiming public chat.
  `train-chat` additionally requires enabled qualified public capabilities,
  the final reviewed inference image and bounded exact-instance cleanup.
- Never retry an ambiguous submit or erase an attempt claim to resubmit.
  The operator approved autonomous bounded Modal smoke iteration on 2026-09-14,
  superseding the earlier routine $10 approval threshold. Keep configured
  resource limits and timeouts; that approval is not unlimited production spend.
  Stop for materially broader scope or serious irreversible uncertainty.
