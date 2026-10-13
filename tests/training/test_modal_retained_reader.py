"""New terminal retained runs reconstruct existing public reads without sources."""
import sys

from synaptic_tuner.api.v1.runs_facade import RunListRequest, RunArtifactRequest
from tuner.training import modal_standalone_runner as runner
from tuner.training import modal_retained_reader as reader
from tests.training.test_modal_standalone_runner import _setup


def test_terminal_reader_replays_new_stores_and_exposes_public_reads(tmp_path, monkeypatch):
    plan, context, events = _setup(tmp_path, monkeypatch)
    report = runner.run_modal_standalone_job(
        plan=plan, context=context, modal_profile="fixture", modal_environment="main",
    )
    run = report.run
    root = tmp_path / "state" / "synaptic-training" / run.run_id
    retained = reader.load_retained_modal_run(root, expected_run=run)
    reader._typed_inputs(retained)  # pure reconstruction failures remain visible in this provider-free test
    monkeypatch.setattr(reader, "validate_local_launcher", lambda: None)
    binding = reader.ModalClientBinding(**retained.inputs["client_binding"])
    monkeypatch.setattr(reader, "open_modal_host_scope", lambda **kwargs: (object(), binding))
    with reader.open_retained_modal_runs(
        root=root, run=run, modal_profile="fixture", modal_environment="main", sdk=sys.modules["modal"],
    ) as view:
        assert view.run == run and not hasattr(view, "training")
        assert view.runs.show(run).state.value == "succeeded"
        assert view.runs.list(RunListRequest(run.project_ref)).outcomes[0].run == run
        assert view.runs.outcome(run).state.value == "succeeded"
        assert b"".join(view.runs.artifacts(RunArtifactRequest(run, "training_metrics", 1024)).iter_bytes()) == b"data"
    assert events == ["bootstrap", "cpu", "stage", "spawn"]
