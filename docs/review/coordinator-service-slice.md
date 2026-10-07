# Coordinator training service slice

`CoordinatorTrainingService` is the concrete engine implementation of the
provider-neutral public training operations protocol. It composes the existing
request loader, resolver, planning, planning-store, clock, and durable
`TrainingCoordinatorV1` ports. It owns no provider identifier, database,
credential, grant, signing key, or legacy Modal lifecycle value.

Loading preserves the caller's exact canonical JSON text. Resolution must bind
the returned request and project identifiers to that request. Planning derives
the existing `TrainingPlanBasisV1`, provider context, and `TrainingPlan`; it
does not introduce another fingerprint domain. Context is retained before the
plan, and both are read back exactly. A `False` put result is an idempotent
restart only when the retained value is identical; malformed store responses or
substitution fail closed.

Preflight reloads the plan and context, re-describes the current provider, and
requires the descriptor and context digests to remain exact. The returned
preflight is reconstructed, plan-bound, and unexpired using only the public
clock's `now()` method. Start validates the caller's original preflight again;
it never obtains a replacement. It delegates once to the actual coordinator
and reconstructs and checks the returned durable workflow before projecting a
`TrainingStart`.

`TrainingStart.accepted=True` means the coordinator durably accepted the
workflow and returned its exact run identity. It does not claim that training
succeeded or even that provider submission is final: a durable
`STAGE_RECONCILE_REQUIRED` or `SUBMIT_RECONCILE_REQUIRED` workflow is still an
accepted start whose subsequent semantics belong to `RunsAPI`.

This slice adds no public export or provider composition. Public cutover and
the separate migration of rich internal planning DTO ownership remain outside
its scope.
