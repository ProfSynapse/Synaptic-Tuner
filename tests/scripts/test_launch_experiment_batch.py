from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".skills" / "fine-tuning" / "scripts" / "launch_experiment_batch.py"
SPEC = spec_from_file_location("launch_experiment_batch", SCRIPT_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_build_command_includes_requested_flags() -> None:
    cmd = MODULE.build_command(
        "Trainers/cloud/experiments/demo.yaml",
        auto_hardware=True,
        optimize_for="cost",
        price_cap=2.5,
        yes=True,
    )

    assert cmd == [
        MODULE.sys.executable,
        "tuner.py",
        "run-experiment",
        "--experiment-spec",
        "Trainers/cloud/experiments/demo.yaml",
        "--auto-hardware",
        "--optimize-for",
        "cost",
        "--price-cap",
        "2.5",
        "--yes",
    ]


def test_launch_batch_staggers_between_runs(monkeypatch) -> None:
    launched = []
    sleeps = []

    def fake_run(cmd, cwd, check):
        launched.append((cmd, cwd, check))

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(MODULE.time, "sleep", lambda seconds: sleeps.append(seconds))

    exit_code = MODULE.launch_batch(
        [
            "Trainers/cloud/experiments/one.yaml",
            "Trainers/cloud/experiments/two.yaml",
            "Trainers/cloud/experiments/three.yaml",
        ],
        stagger_seconds=5.0,
        auto_hardware=True,
        optimize_for="balanced",
        price_cap=None,
        yes=True,
        dry_run=False,
    )

    assert exit_code == 0
    assert len(launched) == 3
    assert sleeps == [5.0, 5.0]


def test_launch_batch_dry_run_skips_sleep_and_subprocess(monkeypatch) -> None:
    launched = []
    sleeps = []

    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: launched.append((args, kwargs)))
    monkeypatch.setattr(MODULE.time, "sleep", lambda seconds: sleeps.append(seconds))

    exit_code = MODULE.launch_batch(
        ["Trainers/cloud/experiments/one.yaml", "Trainers/cloud/experiments/two.yaml"],
        stagger_seconds=5.0,
        auto_hardware=False,
        optimize_for=None,
        price_cap=None,
        yes=False,
        dry_run=True,
    )

    assert exit_code == 0
    assert launched == []
    assert sleeps == []
