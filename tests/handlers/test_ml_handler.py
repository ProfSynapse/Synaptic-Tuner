from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tuner.handlers.ml_handler import MLHandler
from tuner.project import ProjectContext
from tuner.project.errors import WriteAccessError


class _LightweightPipeline:
    """Serializable model leaf used to avoid optional LightGBM in acceptance."""

    named_steps: dict = {}

    def fit(self, _features, _labels):
        return self


class _NoopTracker:
    def set_experiment(self, _name):
        return None

    @contextmanager
    def start_run(self, _name):
        yield

    def log_params(self, _params):
        return None

    def log_metrics(self, _metrics):
        return None


def _snapshot(root: Path, *, exclude_synaptic: bool = False) -> dict[Path, bytes]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
        and not (
            exclude_synaptic
            and path.relative_to(root).parts
            and path.relative_to(root).parts[0] == ".synaptic"
        )
    }


def _config(path: Path, *, output: str | None = None) -> None:
    payload = {
        "task": {
            "type": "classification",
            "name": "host-smoke",
            "target_column": "label",
        },
        "data": {
            "train_path": "../data/train.csv",
            "test_path": "../data/test.csv",
        },
        "features": {"numeric": {"columns": ["value"]}},
    }
    if output is not None:
        payload["output"] = {"dir": output}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _handler(tmp_path: Path) -> tuple[MLHandler, Path]:
    engine = tmp_path / "engine"
    project = tmp_path / "host"
    config_path = project / "experiments" / "ml.yaml"
    config_path.parent.mkdir(parents=True)
    data = project / "data"
    data.mkdir()
    (data / "train.csv").write_text("value,label\n1,yes\n", encoding="utf-8")
    (data / "test.csv").write_text("value,label\n2,no\n", encoding="utf-8")
    context = ProjectContext.host(engine_root=engine, project_root=project)
    return MLHandler().bind_context(context), config_path


def test_host_ml_config_resolves_declaring_document_inputs_and_default_output(tmp_path):
    handler, config_path = _handler(tmp_path)
    _config(config_path)

    config = handler._validated_host_config(config_path)

    assert Path(config.data.train_path) == config_path.parent.parent / "data" / "train.csv"
    assert Path(config.data.test_path) == config_path.parent.parent / "data" / "test.csv"
    assert Path(config.output.dir) == handler.artifact_root / "ml"


def test_host_ml_config_rejects_explicit_source_tree_output(tmp_path):
    handler, config_path = _handler(tmp_path)
    _config(config_path, output="../results")

    with pytest.raises(WriteAccessError):
        handler._validated_host_config(config_path)


def test_host_ml_config_accepts_declared_artifact_output(tmp_path):
    handler, config_path = _handler(tmp_path)
    _config(config_path, output="artifact://ml/custom")

    config = handler._validated_host_config(config_path)

    assert Path(config.output.dir) == handler.artifact_root / "ml" / "custom"


@pytest.mark.parametrize(
    ("declared_output", "expected_relative"),
    [
        (None, Path("ml")),
        ("artifact://ml/custom", Path("ml/custom")),
    ],
)
def test_real_host_handler_train_action_is_root_safe_from_unrelated_cwd(
    tmp_path,
    monkeypatch,
    declared_output,
    expected_relative,
):
    handler, config_path = _handler(tmp_path)
    _config(config_path, output=declared_output)
    config_before = config_path.read_bytes()
    unrelated = tmp_path / "unrelated cwd"
    unrelated.mkdir()
    (unrelated / "sentinel.txt").write_text("unchanged", encoding="utf-8")
    (handler.engine_root / "engine-sentinel.txt").parent.mkdir(parents=True, exist_ok=True)
    (handler.engine_root / "engine-sentinel.txt").write_text("unchanged", encoding="utf-8")
    host_before = _snapshot(handler.project_root, exclude_synaptic=True)
    engine_before = _snapshot(handler.engine_root)
    unrelated_before = _snapshot(unrelated)
    temp_before = _snapshot(tmp_path)
    original_cwd = Path.cwd()

    import Trainers.ml.train as train_module
    import shared.experiment_tracking as tracking_module

    def fake_split(config):
        # This assertion is inside the real trainer call and proves it received
        # the handler's validated, declaring-document-resolved configuration.
        assert Path(config.data.train_path) == handler.project_root / "data" / "train.csv"
        assert Path(config.data.test_path) == handler.project_root / "data" / "test.csv"
        assert Path(config.output.dir) == handler.artifact_root / expected_relative
        features = pd.DataFrame({"value": [1, 2]})
        labels = pd.Series(["yes", "no"])
        return features, features.copy(), labels, labels.copy()

    monkeypatch.setattr(train_module, "load_and_split", fake_split)
    monkeypatch.setattr(
        train_module,
        "build_pipeline",
        lambda _config, n_classes=None: _LightweightPipeline(),
    )
    monkeypatch.setattr(
        train_module,
        "evaluate_model",
        lambda *_args, **_kwargs: {"accuracy": 1.0},
    )
    monkeypatch.setattr(
        tracking_module,
        "create_tracker",
        lambda *_args, **_kwargs: _NoopTracker(),
    )
    handler._args = Namespace(
        ml_config=str(config_path),
        ml_subcommand="train",
        json=True,
    )
    handler._json_mode = True

    monkeypatch.chdir(unrelated)
    cwd_before = Path.cwd()
    assert handler.handle() == 0

    output_base = handler.artifact_root / expected_relative
    run_dirs = [path for path in output_base.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    representative = run_dir / "model.joblib"
    assert representative.is_file()
    assert representative.resolve().is_relative_to(handler.artifact_root.resolve())
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "schema.json").is_file()
    assert Path.cwd() == cwd_before
    assert config_path.read_bytes() == config_before
    assert _snapshot(handler.project_root, exclude_synaptic=True) == host_before
    assert _snapshot(handler.engine_root) == engine_before
    assert _snapshot(unrelated) == unrelated_before
    temp_after = _snapshot(tmp_path)
    changed_temp_paths = {
        path
        for path in temp_before.keys() | temp_after.keys()
        if temp_before.get(path) != temp_after.get(path)
    }
    artifact_relative = handler.artifact_root.relative_to(tmp_path)
    assert changed_temp_paths
    assert all(path.is_relative_to(artifact_relative) for path in changed_temp_paths)

    monkeypatch.chdir(original_cwd)


def test_standalone_training_keeps_legacy_single_argument_call(tmp_path, monkeypatch):
    config_path = tmp_path / "ml.yaml"
    _config(config_path)
    handler = MLHandler().bind_context(
        ProjectContext.standalone(engine_root=tmp_path)
    )
    calls = []

    def fake_main(*args, **kwargs):
        calls.append((args, kwargs))
        return tmp_path / "run"

    monkeypatch.setattr("Trainers.ml.train.main", fake_main)

    assert handler._run_training(str(config_path)) == 0
    assert calls == [((str(config_path),), {})]
