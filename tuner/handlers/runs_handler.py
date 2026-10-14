"""Read an exact retained workflow through public RunsAPI."""

from pathlib import Path
import re

from tuner.handlers.base import BaseHandler


class _CurrentCancellationGrant:
    """The explicit local operator request, scoped only to this retained run."""

    def __init__(self, run):
        from synaptic_tuner.api.v1.execution import ExecutionGrant

        self._run = run
        import hashlib

        token = hashlib.sha256(
            (run.project_ref + "\0" + run.run_id).encode()
        ).hexdigest()
        self._grant = ExecutionGrant("modal-current-cancel-" + token[:32])

    def authorize(self, requirements):
        from synaptic_tuner.api.v1.training_facade import AuthorizationRequirement

        if requirements != (AuthorizationRequirement("training.cancel", False),):
            raise ValueError("current cancellation request unavailable")
        return self._grant

    def bind(self, grant, *, operation, requirements):
        from tuner.execution.foundation_v2.commands import CancelCommandV2

        if (
            type(operation) is not CancelCommandV2
            or grant != self.authorize(requirements)
            or operation.preparation.run_id != self._run.run_id
            or operation.preparation.project_ref != self._run.project_ref
        ):
            raise ValueError("current cancellation request differs from retained run")
        return grant


class RunsHandler(BaseHandler):
    @property
    def name(self):
        return "runs"

    def can_handle_direct_mode(self):
        return True

    def handle(self):
        try:
            action = getattr(self.args, "subcommand", None)
            run_id = getattr(self.args, "run", None)
            profile = getattr(self.args, "modal_profile", None)
            environment = getattr(self.args, "modal_environment", None)
            role = getattr(self.args, "artifact_role", None)
            destination = getattr(self.args, "dest", None)
            reason = getattr(self.args, "reason", None)
            if (
                action
                not in {"show", "observe", "verify", "download", "cancel", "reconcile"}
                or type(run_id) is not str
                or run_id == "latest"
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id) is None
                or (
                    action == "cancel"
                    and (type(reason) is not str or not reason.strip())
                )
                or (action != "cancel" and reason is not None)
                or not profile
                or not environment
                or (action == "download" and (not role or not destination))
                or (
                    action != "download"
                    and (role is not None or destination is not None)
                )
            ):
                raise ValueError
            from synaptic_tuner.api.v1.results import TrainingRunRef
            from tuner.training.modal_local_preparation import training_state_root
            from tuner.training.modal_host_authority import UTCClock
            from tuner.training.modal_retained_reader import (
                open_retained_modal_runs,
                open_retained_modal_lifecycle,
            )

            run = TrainingRunRef(run_id, "standalone-training")
            root = training_state_root(create=False) / run_id
            import modal

            mutation = action in {"cancel", "reconcile"}
            opener = (
                open_retained_modal_lifecycle if mutation else open_retained_modal_runs
            )
            with opener(
                root=root,
                run=run,
                modal_profile=profile,
                modal_environment=environment,
                sdk=modal,
                clock=UTCClock(),
                **(
                    {"cancellation_grants": _CurrentCancellationGrant(run)}
                    if mutation
                    else {}
                ),
            ) as retained:
                if retained.run != run:
                    raise ValueError
                if action == "show":
                    data = {
                        "run": run.to_dict(),
                        "outcome": retained.runs.show(run).to_dict(),
                    }
                    succeeded = True
                elif action == "observe":
                    data = {
                        "run": run.to_dict(),
                        "outcome": retained.runs.outcome(run).to_dict(),
                    }
                    succeeded = True
                elif action in {"cancel", "reconcile"}:
                    outcome = (
                        retained.runs.cancel(run, reason)
                        if action == "cancel"
                        else retained.runs.reconcile(run)
                    )
                    data = {"run": run.to_dict(), "outcome": outcome.to_dict()}
                    succeeded = True
                elif action == "verify":
                    verification = retained.runs.verify(run)
                    data = {
                        "run": run.to_dict(),
                        "verification": {
                            "run": verification.run.to_dict(),
                            "verified": verification.verified,
                            "checked_at": verification.checked_at,
                        },
                    }
                    succeeded = verification.verified is True
                else:
                    from tuner.training.modal_host_download import (
                        download_verified_modal_artifact,
                    )
                    from tuner.execution.providers.modal.coordinator_producer import (
                        MODAL_TRAINING_ARTIFACT_BOUNDS_V1,
                    )

                    path = download_verified_modal_artifact(
                        retained.runs,
                        run,
                        role=role,
                        output_root=Path(destination).absolute(),
                        maximum_bytes=MODAL_TRAINING_ARTIFACT_BOUNDS_V1.max_artifact_bytes,
                    )
                    data = {
                        "run": run.to_dict(),
                        "artifact_role": role,
                        "path": str(path),
                    }
                    succeeded = True
            self.output(data, success=succeeded)
            return 0 if succeeded else 2
        except Exception:
            self.output_error(
                "Retained run operation unavailable", code="RUN_READ_UNAVAILABLE"
            )
            return 2
