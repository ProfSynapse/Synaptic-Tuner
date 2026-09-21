import hashlib
import json
import posixpath
import subprocess
from argparse import Namespace
from pathlib import Path

import yaml
import pytest

from tuner.cloud.derived_training_image import (
    DerivedTrainingImageError,
    VERIFICATION_SCHEMA,
    load_profile as load_derived_image_profile,
    render_dockerfile,
)
from tuner.handlers.local_run_handler import LocalRunError, LocalRunHandler
from tuner.discovery import (
    BaseModelDiscovery,
    DatasetDiscovery,
    PromptSetDiscovery,
    RubricDiscovery,
    TrainingRunDiscovery,
    list_recipes,
)
from tuner.project import ProjectContext
from tuner.batch.runner import _runtime_dirs


def test_checked_in_qwen35_32k_recipe_binds_immutable_local_smoke_contract():
    recipe_path = (
        Path(__file__).resolve().parents[2]
        / "Trainers"
        / "recipes"
        / "qwen35_4b_32k_prompt_completion.yaml"
    )
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))

    assert recipe["run"]["dry_run"] is True
    assert recipe["job"]["runtime_profile"] == "qwen35-sft-v1"
    assert "image" not in recipe["job"]
    assert "setup" not in recipe
    assert "image_qualification" not in recipe["job"]
    assert recipe["model"] == {
        "name": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "max_seq_length": 32768,
        "load_in_4bit": False,
    }
    assert recipe["dataset"]["schema_version"] == "syntunia-sft-row/v2"
    assert recipe["dataset"]["format"] == "messages"
    assert recipe["dataset"]["use_preassigned_splits"] is True
    assert recipe["dataset"]["split_dataset"] is False
    assert recipe["training"]["packing"] is False
    assert recipe["training"]["completion_only_loss"] is True
    assert recipe["training"]["assistant_only_loss"] is False
    assert recipe["training"]["prompt_render"] == "prompt_completion"
    assert recipe["training"]["require_memory_efficient_loss"] is True


def test_named_runtime_profile_resolves_into_plan_and_lineage_inputs(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"messages":[]}\n', encoding="utf-8")
    config = tmp_path / "job.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "profiled-local-sft",
                "provider": "local_docker",
                "job": {"runtime_profile": "qwen35-sft-v1", "transfer": "copy"},
                "run": {"method": "sft", "dry_run": True},
                "model": {
                    "name": "Qwen/Qwen3.5-4B",
                    "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                    "load_in_4bit": False,
                },
                "dataset": {"local_file": str(dataset)},
                "training": {"max_steps": 1},
                "artifacts": {
                    "output_root": "toolset-training-artifacts/runs/local_docker/sft/profiled",
                    "run_timestamp": "unit",
                },
            }
        ),
        encoding="utf-8",
    )

    handler = LocalRunHandler(args=Namespace(json=True, job_config=str(config)))
    plan = handler._compile(config, handler._load_yaml(config))

    expected_image = (
        "unsloth/unsloth@sha256:"
        "1644d635bc7c5b57ed64cabbab1ae00647dfb583c2135fdfb6a461aeeba52739"
    )
    assert plan["image"] == expected_image
    assert plan["runtime_profile"]["name"] == "qwen35-sft-v1"
    assert plan["runtime_profile"]["image"] == expected_image
    assert plan["runtime_profile"]["inventory_sha256"] == (
        "sha256:f59c5a85101a9d289c0f680dd5fec9f73a5bd388c984aa60dff6d7ef08aef889"
    )
    assert plan["lineage_inputs"]["runtime_profile"] == {
        "name": "qwen35-sft-v1",
        "profile_sha256": plan["runtime_profile"]["profile_sha256"],
        "image": expected_image,
        "inventory_sha256": plan["runtime_profile"]["inventory_sha256"],
        "model": "Qwen/Qwen3.5-4B",
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    }
    assert plan["command"][plan["command"].index("--runtime-profile-name") + 1] == (
        "qwen35-sft-v1"
    )
    assert plan["command"][plan["command"].index("--runtime-image") + 1] == expected_image


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"job_image": True}, "raw job.image override"),
        ({"setup_pip": True}, "forbid setup.pip"),
    ],
)
def test_named_runtime_profile_rejects_runtime_mutation(
    tmp_path, mutation, message
):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"messages":[]}\n', encoding="utf-8")
    job = {"runtime_profile": "qwen35-sft-v1", "transfer": "copy"}
    config_doc = {
        "name": "profiled-local-sft",
        "provider": "local_docker",
        "job": job,
        "run": {"method": "sft", "dry_run": True},
        "model": {
            "name": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "load_in_4bit": False,
        },
        "dataset": {"local_file": str(dataset)},
        "training": {"max_steps": 1},
        "artifacts": {"output_root": "toolset-training-artifacts/test", "run_timestamp": "unit"},
    }
    if mutation.get("job_image"):
        job["image"] = "example.invalid/image@sha256:" + "1" * 64
    if mutation.get("setup_pip"):
        config_doc["setup"] = {"pip": ["transformers==0.0.0"]}
    config = tmp_path / "job.yaml"
    config.write_text(yaml.safe_dump(config_doc), encoding="utf-8")
    handler = LocalRunHandler(args=Namespace(json=True, job_config=str(config)))

    with pytest.raises(LocalRunError, match=message):
        handler._compile(config, handler._load_yaml(config))


@pytest.mark.parametrize(
    ("model", "revision", "method", "message"),
    [
        ("Qwen/Qwen3.5-2B", "revision", "sft", "does not support model"),
        ("Qwen/Qwen3.5-4B", None, "sft", "does not support model revision"),
        ("Qwen/Qwen3.5-4B", "arbitrary", "sft", "does not support model revision"),
        (
            "Qwen/Qwen3.5-4B",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "kto",
            "supports only run.method=sft",
        ),
    ],
)
def test_named_runtime_profile_rejects_unsupported_compatibility(
    tmp_path, model, revision, method, message
):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{}\n', encoding="utf-8")
    config = tmp_path / "job.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "profiled-incompatible",
                "provider": "local_docker",
                "job": {"runtime_profile": "qwen35-sft-v1", "transfer": "copy"},
                "run": {"method": method, "dry_run": True},
                "model": {
                    "name": model,
                    "revision": revision,
                    "load_in_4bit": False,
                },
                "dataset": {"local_file": str(dataset)},
                "training": {"max_steps": 1},
            }
        ),
        encoding="utf-8",
    )
    handler = LocalRunHandler(args=Namespace(json=True, job_config=str(config)))

    with pytest.raises(LocalRunError, match=message):
        handler._compile(config, handler._load_yaml(config))


@pytest.mark.parametrize(
    "reserved",
    [
        "--model-name",
        "--model-name=Other/Model",
        "--model-revision",
        "--model-revision=other",
        "--model-rev",
        "--model-rev=other",
        "--runtime-profile-name",
        "--runtime-profile-name=other",
        "--runtime-profile-digest",
        "--runtime-profile-digest=sha256:" + "1" * 64,
        "--runtime-inventory-digest",
        "--runtime-inventory-digest=sha256:" + "1" * 64,
        "--runtime-image",
        "--runtime-image=example/image@sha256:" + "1" * 64,
        "--runtime-im",
    ],
)
def test_named_runtime_profile_rejects_reserved_extra_args(tmp_path, reserved):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{}\n', encoding="utf-8")
    config = tmp_path / "job.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "profiled-extra-args",
                "provider": "local_docker",
                "job": {"runtime_profile": "qwen35-sft-v1", "transfer": "copy"},
                "run": {"method": "sft", "extra_args": [reserved]},
                "model": {
                    "name": "Qwen/Qwen3.5-4B",
                    "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                    "load_in_4bit": False,
                },
                "dataset": {"local_file": str(dataset)},
                "training": {"max_steps": 1},
            }
        ),
        encoding="utf-8",
    )
    handler = LocalRunHandler(args=Namespace(json=True, job_config=str(config)))

    with pytest.raises(LocalRunError, match="reserved runtime identity flag"):
        handler._compile(config, handler._load_yaml(config))


def test_named_runtime_profile_rejects_run_command(tmp_path):
    config = tmp_path / "job.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "profiled-command",
                "provider": "local_docker",
                "job": {"runtime_profile": "qwen35-sft-v1", "transfer": "copy"},
                "run": {"method": "sft", "command": ["python", "custom.py"]},
                "model": {
                    "name": "Qwen/Qwen3.5-4B",
                    "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                    "load_in_4bit": False,
                },
            }
        ),
        encoding="utf-8",
    )
    handler = LocalRunHandler(args=Namespace(json=True, job_config=str(config)))

    with pytest.raises(LocalRunError, match="does not support run.command"):
        handler._compile(config, handler._load_yaml(config))


@pytest.mark.parametrize(
    "trainer",
    [
        "../outside.py",
        "Trainers/sft/../kto/train_kto.py",
        "C:/outside.py",
        "/outside.py",
        r"Trainers\sft\train_sft.py",
    ],
)
def test_local_run_rejects_non_normalized_or_escaping_trainer_path(
    tmp_path, trainer
):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{}\n', encoding="utf-8")
    config = tmp_path / "job.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "trainer-path-rejection",
                "provider": "local_docker",
                "job": {"transfer": "copy"},
                "run": {"method": "sft", "trainer": trainer, "dry_run": True},
                "model": {"name": "Qwen/Qwen3.5-2B", "load_in_4bit": False},
                "dataset": {"local_file": str(dataset)},
                "training": {"max_steps": 1},
            }
        ),
        encoding="utf-8",
    )
    handler = LocalRunHandler(args=Namespace(json=True, job_config=str(config)))

    with pytest.raises(LocalRunError, match="run.trainer"):
        handler._compile(config, handler._load_yaml(config))


def test_local_run_qualified_image_template_compiles_but_effectful_run_fails_closed(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"messages":[]}\n', encoding="utf-8")
    profile = (
        Path(__file__).resolve().parents[2]
        / "Trainers"
        / "image_profiles"
        / "qwen35_4b_32k_prompt_completion.yaml"
    )
    missing_report = tmp_path / "missing-verification.json"
    config = tmp_path / "job.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "qualified-template",
                "provider": "local_docker",
                "job": {
                    "transfer": "copy",
                    "image_qualification": {
                        "required": True,
                        "profile": str(profile),
                        "verification_report": str(missing_report),
                    },
                },
                "run": {"method": "sft", "dry_run": True},
                "model": {"name": "Qwen/Qwen3.5-4B", "load_in_4bit": False},
                "dataset": {"local_file": str(dataset)},
                "training": {"max_steps": 1},
                "artifacts": {
                    "output_root": "toolset-training-artifacts/runs/local_docker/sft/unit",
                    "run_timestamp": "unit",
                },
            }
        ),
        encoding="utf-8",
    )

    planning = LocalRunHandler(args=Namespace(json=True, job_config=str(config)))
    plan = planning._compile(config, planning._load_yaml(config))
    assert plan["image"] == ""
    assert plan["image_qualification"]["verification_report"] == str(missing_report)

    effectful = LocalRunHandler(
        args=Namespace(json=False, job_config=str(config), auto_confirm=True)
    )
    monkeypatch.setattr(
        effectful,
        "_pull_image",
        lambda *_args, **_kwargs: pytest.fail("Docker must not be reached"),
    )
    assert effectful.handle() == 1


def test_forged_diagnostic_report_needs_live_docker_before_filesystem_prep(
    tmp_path, monkeypatch
):
    engine_root = tmp_path / "engine"
    engine_root.mkdir()
    context = ProjectContext.standalone(engine_root=engine_root)
    dataset = engine_root / "data.jsonl"
    dataset.write_text('{"messages":[]}\n', encoding="utf-8")
    base = "docker.io/unsloth/unsloth@sha256:" + "1" * 64
    config_digest = "sha256:" + "3" * 64
    profile_path = engine_root / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "synaptic-derived-training-image-profile/v1",
                "name": "handler-test",
                "platform": "linux/amd64",
                "base_image": base,
                "python_executable": "/opt/conda/bin/python3",
                "packages": [
                    {
                        "distribution": "transformers",
                        "source": "index",
                        "version": "5.5.0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    profile = load_derived_image_profile(profile_path)
    report_path = engine_root / "verification.json"
    build_metadata = {
        "containerimage.digest": "sha256:" + "2" * 64,
        "containerimage.config.digest": config_digest,
    }
    build_metadata_bytes = (
        json.dumps(build_metadata, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    report_path.write_text(
        json.dumps(
            {
                "schema_version": VERIFICATION_SCHEMA,
                "status": "DIAGNOSTIC_PASS",
                "profile_sha256": profile.canonical_sha256,
                "candidate_sha256": "sha256:" + "4" * 64,
                "build_receipt_sha256": "sha256:" + "5" * 64,
                "dockerfile_sha256": "sha256:"
                + hashlib.sha256(
                    render_dockerfile(profile).encode("utf-8")
                ).hexdigest(),
                "build_metadata": build_metadata,
                "build_metadata_sha256": "sha256:"
                + hashlib.sha256(build_metadata_bytes).hexdigest(),
                "base_image": base,
                "final_oci_reference": (
                    "registry.example/syntunia/trainer@sha256:" + "2" * 64
                ),
                "final_oci_digest": "sha256:" + "2" * 64,
                "image_config_digest": config_digest,
                "packages": {
                    "transformers": {"version": "5.5.0", "direct_url": None}
                },
            }
        ),
        encoding="utf-8",
    )
    artifact_root = engine_root / "must-not-exist"
    config = engine_root / "job.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "forged-report",
                "provider": "local_docker",
                "job": {
                    "image": config_digest,
                    "pull_policy": "never",
                    "transfer": "copy",
                    "image_qualification": {
                        "required": True,
                        "profile": str(profile_path),
                        "verification_report": str(report_path),
                    },
                },
                "run": {"method": "sft", "dry_run": True},
                "model": {"name": "Qwen/Qwen3.5-4B", "load_in_4bit": False},
                "dataset": {"local_file": str(dataset)},
                "training": {"max_steps": 1},
                "artifacts": {
                    "output_root": str(artifact_root),
                    "run_timestamp": "unit",
                },
            }
        ),
        encoding="utf-8",
    )
    docker = engine_root / "docker"
    docker.write_bytes(b"docker")
    monkeypatch.setattr(
        "tuner.handlers.local_run_handler.shutil.which", lambda _name: str(docker)
    )
    live_calls = 0

    def reject_forged_report(**_kwargs):
        nonlocal live_calls
        live_calls += 1
        raise DerivedTrainingImageError("LIVE_IMAGE_MISMATCH")

    monkeypatch.setattr(
        "tuner.handlers.local_run_handler.verify_effectful_launch",
        reject_forged_report,
    )

    planning = LocalRunHandler(
        args=Namespace(json=True, job_config=str(config)), context=context
    )
    assert planning.handle() == 0
    assert live_calls == 0

    effectful = LocalRunHandler(
        args=Namespace(json=False, job_config=str(config), auto_confirm=True),
        context=context,
    )
    monkeypatch.setattr(
        effectful,
        "_pull_image",
        lambda *_args, **_kwargs: pytest.fail("pull must not be reached"),
    )
    assert effectful.handle() == 1
    assert live_calls == 1
    assert not artifact_root.exists()

    mutated = yaml.safe_load(config.read_text(encoding="utf-8"))
    mutated["setup"] = {"pip": ["transformers==0.0.0"]}
    config.write_text(yaml.safe_dump(mutated), encoding="utf-8")
    rejecting = LocalRunHandler(
        args=Namespace(json=True, job_config=str(config)), context=context
    )
    with pytest.raises(LocalRunError, match="forbid setup.pip"):
        rejecting._compile(config, rejecting._load_yaml(config))


def test_qualified_run_reuses_exact_verified_docker_authority(monkeypatch):
    handler = LocalRunHandler(args=Namespace(json=True))
    checks = 0

    class Authority:
        environment = {"DOCKER_CONTENT_TRUST": "1", "PATH": "closed"}

        def assert_current(self):
            nonlocal checks
            checks += 1

        def command(self, arguments):
            self.assert_current()
            return ["C:/trusted/docker.exe", "--config", "C:/empty", *arguments]

    handler._qualified_docker = Authority()  # type: ignore[assignment]
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("tuner.handlers.local_run_handler.subprocess.run", fake_run)
    handler._run(["docker", "pull", "repo/image@sha256:" + "1" * 64])

    assert captured["args"][:3] == [
        "C:/trusted/docker.exe",
        "--config",
        "C:/empty",
    ]
    assert captured["kwargs"]["env"] == Authority.environment
    assert checks == 2


def test_local_run_sft_config_compiles_repo_relative_dataset(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"conversations":[]}\n', encoding="utf-8")
    config = tmp_path / "job.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "unit-local-sft",
                "provider": "local_docker",
                "job": {"transfer": "copy"},
                "run": {"method": "sft"},
                "model": {"name": "Qwen/Qwen3.5-2B", "load_in_4bit": False},
                "dataset": {"local_file": str(dataset)},
                "training": {"max_steps": 1},
                "artifacts": {
                    "output_root": "toolset-training-artifacts/runs/local_docker/sft/unit-local-sft",
                    "run_timestamp": "unit",
                },
            }
        ),
        encoding="utf-8",
    )

    handler = LocalRunHandler(args=Namespace(json=True, job_config=str(config)))
    plan = handler._compile(config, handler._load_yaml(config))

    assert plan["transfer"] == "copy"
    assert plan["command"][:3] == ["python", "train_sft.py", "--model-name"]
    local_file_index = plan["command"].index("--local-file") + 1
    dataset_entry = next(
        entry for entry in plan["copy_entries"] if entry.source == dataset.resolve()
    )
    expected_dataset_arg = posixpath.relpath(
        dataset_entry.destination, plan["workdir"]
    )
    assert plan["command"][local_file_index] == expected_dataset_arg
    assert ":" not in dataset_entry.destination.split("/workspace/repo/", 1)[-1]
    assert plan["host_artifact_path"].name == "unit"


def test_embedded_catalogs_are_host_first_with_declaring_roots(tmp_path):
    engine = tmp_path / "engine"
    project = tmp_path / "host"
    config_root = project / "experiments"
    for root in (engine, project, config_root):
        root.mkdir(parents=True, exist_ok=True)
    context = ProjectContext.host(
        engine_root=engine, project_root=project, config_root=config_root
    )

    for root, marker in ((engine, "engine"), (project, "host")):
        datasets = root / "Datasets"
        datasets.mkdir()
        (datasets / "shared.jsonl").write_text(
            json.dumps({"prompt": marker}) + "\n", encoding="utf-8"
        )
        rubrics = root / "SynthChat" / "rubrics"
        rubrics.mkdir(parents=True)
        (rubrics / "shared.yaml").write_text(
            yaml.safe_dump({"name": marker, "description": marker}), encoding="utf-8"
        )
        scenarios = root / "Evaluator" / "config" / "scenarios"
        scenarios.mkdir(parents=True)
        (scenarios / "shared.yaml").write_text(
            yaml.safe_dump({"description": marker, "tests": [{"id": marker}]}),
            encoding="utf-8",
        )
        recipes = root / "Trainers" / "recipes"
        recipes.mkdir(parents=True)
        (recipes / "shared.yaml").write_text(
            yaml.safe_dump({"name": marker, "target": "local", "method": "sft"}),
            encoding="utf-8",
        )

    dataset = DatasetDiscovery(context=context).discover_all()[0]
    rubric = RubricDiscovery(context=context).discover_all()[0]
    prompt_set = PromptSetDiscovery(context=context).discover_all()[0]
    recipe = list_recipes(engine, context=context)[0]
    assert dataset.path.is_relative_to(project)
    assert dataset.declaring_root == project / "Datasets"
    assert rubric.description == "host"
    assert rubric.declaring_root == project / "SynthChat" / "rubrics"
    assert prompt_set.description == "host"
    assert prompt_set.declaring_root == project / "Evaluator" / "config" / "scenarios"
    assert recipe.name == "host"
    assert recipe.declaring_root == project / "Trainers" / "recipes"


def test_embedded_run_and_model_discovery_use_artifact_and_host_config_roots(tmp_path):
    engine = tmp_path / "engine"
    project = tmp_path / "host"
    config_root = project / "experiments"
    config_root.mkdir(parents=True)
    context = ProjectContext.host(
        engine_root=engine, project_root=project, config_root=config_root
    )
    run = context.artifact_root / "runs" / "local_docker" / "sft" / "demo" / "001"
    (run / "final_model").mkdir(parents=True)
    engine_run = engine / "Trainers" / "sft" / "sft_output" / "002"
    (engine_run / "final_model").mkdir(parents=True)
    (config_root / "train.yaml").write_text(
        yaml.safe_dump({"model": {"model_name": "host/model"}}), encoding="utf-8"
    )

    discovered = TrainingRunDiscovery(context=context).discover("sft")
    base_models, finetuned = BaseModelDiscovery(context=context).discover_all()
    assert discovered == [run, engine_run]
    assert base_models[0].name == "host/model"
    assert base_models[0].declaring_root == config_root
    assert finetuned[0].path == str(run.relative_to(project) / "final_model")

    output, state = _runtime_dirs(context.artifact_root / "batch" / "demo", context)
    assert output == (context.artifact_root / "batch" / "demo").resolve()
    assert state is not None and state.parent == context.state_root / "batch"


def _compile_local_command(
    tmp_path, *, method, trainer, training, lora=None, aux_head=None,
    dataset_config=None, model_config=None,
):
    """Compile a local-docker recipe and return the built trainer command list."""
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"conversations":[]}\n', encoding="utf-8")
    config = tmp_path / "job.yaml"
    recipe = {
        "name": "unit-local",
        "provider": "local_docker",
        "job": {"transfer": "copy"},
        "run": {"method": method, "trainer": trainer},
        "model": {
            "name": "Qwen/Qwen3.5-2B",
            "load_in_4bit": False,
            **(model_config or {}),
        },
        "dataset": {"local_file": str(dataset), **(dataset_config or {})},
        "training": training,
        "artifacts": {
            "output_root": "toolset-training-artifacts/runs/local_docker/unit",
            "run_timestamp": "unit",
        },
    }
    if lora is not None:
        recipe["lora"] = lora
    if aux_head is not None:
        recipe["aux_head"] = aux_head
    config.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    handler = LocalRunHandler(args=Namespace(json=True, job_config=str(config)))
    plan = handler._compile(config, handler._load_yaml(config))
    return plan["command"]


def test_local_run_forwards_seed(tmp_path):
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1, "seed": 1234},
    )
    assert "--seed" in command
    assert command[command.index("--seed") + 1] == "1234"


def test_local_run_forwards_seed_zero(tmp_path):
    # seed=0 must survive forwarding; _append_flag skips only None, not falsy 0.
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1, "seed": 0},
    )
    assert "--seed" in command
    assert command[command.index("--seed") + 1] == "0"


def test_local_run_omits_seed_when_absent(tmp_path):
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1},
    )
    assert "--seed" not in command


def test_local_run_omits_beta_for_sft(tmp_path):
    # beta is gated to dpo/kto; the SFT trainer has no --beta flag, so a stray beta
    # in an sft recipe must not be forwarded as a command argument.
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1, "beta": 0.5},
    )
    assert "--beta" not in command


# Run-control flags with no experimental meaning that dpo/kto do not expose as CLI
# args. The handler gates these to sft so dpo/kto commands omit them (scope v2 (A)).
RUN_CONTROL_SFT_ONLY_FLAGS = (
    "--save-steps",
    "--save-total-limit",
    "--load-in-4bit",
    "--no-load-in-4bit",
    "--no-dashboard",
    "--quiet",
)

# LoRA budget flags that now have parity on all three trainers (scope v2 (B)). The
# recipe's LoRA budget is load-bearing (the identical-budget confound control), so
# these MUST flow to dpo/kto, not be gated.
LORA_PARITY_FLAGS = (
    "--lora-r",
    "--lora-alpha",
    "--lora-dropout",
    "--lora-target-modules",
)

_LORA_BLOCK = {"r": 64, "alpha": 128, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]}


def test_local_run_dispatches_dpo_method(tmp_path):
    # local-run dispatches dpo through the generic builder (no longer SFT-only).
    command = _compile_local_command(
        tmp_path, method="dpo", trainer="Trainers/dpo/train_dpo.py",
        training={"max_steps": 1, "seed": 7, "beta": 0.05},
    )
    assert command[:2] == ["python", "train_dpo.py"]
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--beta") + 1] == "0.05"


def test_local_run_dispatches_kto_method(tmp_path):
    command = _compile_local_command(
        tmp_path, method="kto", trainer="Trainers/kto/train_kto.py",
        training={"max_steps": 1, "seed": 7, "beta": 0.5},
    )
    assert command[:2] == ["python", "train_kto.py"]
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--beta") + 1] == "0.5"


def test_local_run_dpo_omits_run_control_flags(tmp_path):
    # Run-control flags (dashboard/quiet/save/4bit) have no experimental meaning and
    # the dpo trainer's argparse rejects them, so a dpo command must omit them.
    command = _compile_local_command(
        tmp_path, method="dpo", trainer="Trainers/dpo/train_dpo.py",
        training={"max_steps": 1, "seed": 7, "beta": 0.05, "save_steps": 50},
        lora=_LORA_BLOCK,
    )
    for flag in RUN_CONTROL_SFT_ONLY_FLAGS:
        assert flag not in command, f"dpo command leaked run-control flag {flag}"


def test_local_run_dpo_carries_lora_budget(tmp_path):
    # The LoRA budget is load-bearing (identical-budget confound control): a dpo
    # recipe's lora block MUST flow to the trainer, not silently fall back to the
    # trainer config default. Verified end-to-end via the parity CLI flags.
    command = _compile_local_command(
        tmp_path, method="dpo", trainer="Trainers/dpo/train_dpo.py",
        training={"max_steps": 1, "seed": 7, "beta": 0.05},
        lora=_LORA_BLOCK,
    )
    for flag in LORA_PARITY_FLAGS:
        assert flag in command, f"dpo command dropped load-bearing LoRA flag {flag}"
    assert command[command.index("--lora-r") + 1] == "64"
    assert command[command.index("--lora-alpha") + 1] == "128"
    assert command[command.index("--lora-target-modules") + 1] == "q_proj,v_proj"


def test_local_run_kto_carries_lora_budget(tmp_path):
    command = _compile_local_command(
        tmp_path, method="kto", trainer="Trainers/kto/train_kto.py",
        training={"max_steps": 1, "seed": 7, "beta": 0.5},
        lora=_LORA_BLOCK,
    )
    assert command[command.index("--lora-r") + 1] == "64"
    assert command[command.index("--lora-alpha") + 1] == "128"


def test_local_run_sft_still_emits_its_flags(tmp_path):
    # The sft path is byte-unchanged: it still receives the gated run-control flags.
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1, "save_steps": 50},
    )
    assert "--quiet" in command
    assert "--no-dashboard" in command
    assert "--save-steps" in command


def test_local_run_forwards_beta_zero_for_dpo(tmp_path):
    # beta uses is-not-None semantics (mirroring --seed): an explicit beta: 0.0 is
    # honored, not silently dropped to the trainer default. (Provenance: matrix
    # never uses 0, but a silent swap is the cardinal sin this guards against.)
    command = _compile_local_command(
        tmp_path, method="dpo", trainer="Trainers/dpo/train_dpo.py",
        training={"max_steps": 1, "beta": 0.0},
    )
    assert "--beta" in command
    assert command[command.index("--beta") + 1] == "0.0"


def test_local_run_omits_beta_when_absent_for_dpo(tmp_path):
    command = _compile_local_command(
        tmp_path, method="dpo", trainer="Trainers/dpo/train_dpo.py",
        training={"max_steps": 1, "seed": 7},
    )
    assert "--beta" not in command


def test_local_run_rejects_unregistered_method(tmp_path):
    # The dispatch guard still rejects methods outside {sft, dpo, kto} (absent an
    # explicit run.command), rather than silently building a command.
    import pytest

    from tuner.handlers.local_run_handler import LocalRunError

    with pytest.raises(LocalRunError):
        _compile_local_command(
            tmp_path, method="grpo", trainer="Trainers/grpo/train_grpo.py",
            training={"max_steps": 1},
        )


def test_local_run_sft_serializes_chat_template_kwargs_as_json(tmp_path):
    # chat_template_kwargs is a nested mapping, so it cannot ride the scalar
    # _append_flag path; the handler JSON-encodes it onto --chat-template-kwargs.
    # The trainer parses the JSON back into config.training.chat_template_kwargs.
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1, "chat_template_kwargs": {"enable_thinking": False}},
    )
    assert "--chat-template-kwargs" in command
    payload = command[command.index("--chat-template-kwargs") + 1]
    assert json.loads(payload) == {"enable_thinking": False}


def test_local_run_sft_omits_chat_template_kwargs_when_absent(tmp_path):
    # Byte-identical for recipes that do not set the key: no flag emitted.
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1},
    )
    assert "--chat-template-kwargs" not in command


def test_local_run_dpo_omits_chat_template_kwargs(tmp_path):
    # --chat-template-kwargs is sft-only: the dpo/kto trainers template internally
    # via TRL and expose no such flag, so a stray key in a dpo recipe must not be
    # forwarded (argparse would reject it).
    command = _compile_local_command(
        tmp_path, method="dpo", trainer="Trainers/dpo/train_dpo.py",
        training={"max_steps": 1, "beta": 0.05, "chat_template_kwargs": {"enable_thinking": False}},
    )
    assert "--chat-template-kwargs" not in command


def test_local_run_sft_omits_aux_head_flags_when_absent(tmp_path):
    # Byte-identical for recipes with no aux_head block and no training.prompt_render:
    # zero --aux-head-* flags emitted (the whole point of the gated forwarding).
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1},
    )
    assert not any(arg.startswith("--aux-head-") for arg in command)
    assert "--no-aux-head-enabled" not in command


def test_local_run_sft_forwards_aux_head_block(tmp_path):
    # An aux_head block forwards field-by-field; enabled/freeze_base use the
    # tri-state --flag/--no-flag form; prompt_render rides training, not aux_head.
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1, "prompt_render": "prompt_completion"},
        aux_head={
            "enabled": True,
            "layer": 35,
            "token_position": "end_of_prompt",
            "target_field": "score",
            "loss": "bce",
            "head_type": "linear",
            "out_activation": "sigmoid",
            "input_norm": "layernorm",
            "freeze_base": False,
            "lm_loss_weight": 1.0,
            "head_lr": 0.005,
        },
    )
    assert "--aux-head-enabled" in command
    assert "--no-aux-head-freeze-base" in command
    assert command[command.index("--aux-head-layer") + 1] == "35"
    assert command[command.index("--aux-head-token-position") + 1] == "end_of_prompt"
    assert command[command.index("--aux-head-target-field") + 1] == "score"
    assert command[command.index("--aux-head-loss") + 1] == "bce"
    assert command[command.index("--aux-head-head-type") + 1] == "linear"
    assert command[command.index("--aux-head-out-activation") + 1] == "sigmoid"
    assert command[command.index("--aux-head-input-norm") + 1] == "layernorm"
    assert command[command.index("--aux-head-lm-loss-weight") + 1] == "1.0"
    assert command[command.index("--aux-head-head-lr") + 1] == "0.005"
    assert command[command.index("--aux-head-prompt-render") + 1] == "prompt_completion"


def test_local_run_sft_forwards_falsy_aux_head_values(tmp_path):
    # A set-but-falsy lm_loss_weight (0.0) and an explicit enabled: false must
    # still forward (provenance: the trainer must run the recipe's exact values).
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1},
        aux_head={"enabled": False, "lm_loss_weight": 0.0},
    )
    assert "--no-aux-head-enabled" in command
    assert command[command.index("--aux-head-lm-loss-weight") + 1] == "0.0"


def test_local_run_sft_forwards_prompt_render_without_aux_head_block(tmp_path):
    # prompt_render is a training-config knob, forwarded independently of the
    # aux_head block (a plain SFT recipe may want the faithful render).
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1, "prompt_render": "prompt_completion"},
    )
    assert command[command.index("--aux-head-prompt-render") + 1] == "prompt_completion"


def test_local_run_dpo_omits_aux_head_flags(tmp_path):
    # aux_head forwarding is sft-only: a stray aux_head block in a dpo recipe must
    # not be forwarded (the dpo trainer exposes no --aux-head-* flags).
    command = _compile_local_command(
        tmp_path, method="dpo", trainer="Trainers/dpo/train_dpo.py",
        training={"max_steps": 1, "beta": 0.05, "prompt_render": "prompt_completion"},
        aux_head={"enabled": True, "layer": 35},
    )
    assert not any(arg.startswith("--aux-head-") for arg in command)


def test_local_run_sft_propagates_complete_raw_text_semantics(tmp_path):
    command = _compile_local_command(
        tmp_path,
        method="sft",
        trainer="Trainers/sft/train_sft.py",
        training={
            "max_steps": 1,
            "completion_only_loss": False,
            "assistant_only_loss": False,
        },
        dataset_config={
            "schema_version": "syntunia-sft-row/v1",
            "format": "raw_text",
            "use_preassigned_splits": True,
        },
    )

    assert "--no-completion-only-loss" in command
    assert "--no-assistant-only-loss" in command
    assert "--use-preassigned-splits" in command
    assert "--split-dataset" not in command


@pytest.mark.parametrize(
    "training,dataset_config,aux_head,message",
    [
        (
            {"max_steps": 1, "assistant_only_loss": False},
            {"schema_version": "syntunia-sft-row/v1", "format": "raw_text", "use_preassigned_splits": True},
            None,
            "completion_only_loss=false",
        ),
        (
            {"max_steps": 1, "completion_only_loss": False},
            {"schema_version": "syntunia-sft-row/v1", "format": "raw_text", "use_preassigned_splits": True},
            None,
            "assistant_only_loss=false",
        ),
        (
            {"max_steps": 1, "completion_only_loss": False, "assistant_only_loss": False},
            {"schema_version": "syntunia-sft-row/v1", "format": "raw_text"},
            None,
            "use_preassigned_splits=true",
        ),
        (
            {"max_steps": 1, "completion_only_loss": False, "assistant_only_loss": False},
            {
                "schema_version": "syntunia-sft-row/v1",
                "format": "raw_text",
                "use_preassigned_splits": True,
                "split_dataset": True,
            },
            None,
            "cannot use random",
        ),
        (
            {"max_steps": 1, "completion_only_loss": False, "assistant_only_loss": False},
            {"schema_version": "syntunia-sft-row/v1", "format": "raw_text", "use_preassigned_splits": True},
            {"token_position": "end_of_prompt"},
            "end_of_prompt",
        ),
    ],
)
def test_local_run_sft_rejects_incomplete_or_incompatible_raw_text_semantics(
    tmp_path, training, dataset_config, aux_head, message
):
    with pytest.raises(LocalRunError, match=message):
        _compile_local_command(
            tmp_path,
            method="sft",
            trainer="Trainers/sft/train_sft.py",
            training=training,
            dataset_config=dataset_config,
            aux_head=aux_head,
        )


def test_local_run_legacy_command_omits_raw_text_flags(tmp_path):
    command = _compile_local_command(
        tmp_path,
        method="sft",
        trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1},
    )
    assert "--completion-only-loss" not in command
    assert "--no-completion-only-loss" not in command
    assert "--assistant-only-loss" not in command
    assert "--no-assistant-only-loss" not in command
    assert "--use-preassigned-splits" not in command


def test_local_run_sft_forwards_authoritative_message_splits(tmp_path):
    command = _compile_local_command(
        tmp_path,
        method="sft",
        trainer="Trainers/sft/train_sft.py",
        training={
            "max_steps": 1,
            "packing": False,
            "completion_only_loss": True,
            "assistant_only_loss": False,
            "prompt_render": "prompt_completion",
        },
        dataset_config={
            "schema_version": "syntunia-sft-row/v2",
            "format": "messages",
            "use_preassigned_splits": True,
        },
    )
    assert "--use-preassigned-splits" in command
    assert "--split-dataset" not in command
    assert "--completion-only-loss" in command
    assert "--no-assistant-only-loss" in command


def test_local_run_sft_transports_32k_loss_guard_and_exact_revision(tmp_path):
    revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    command = _compile_local_command(
        tmp_path,
        method="sft",
        trainer="Trainers/sft/train_sft.py",
        model_config={"revision": revision, "max_seq_length": 32768},
        training={
            "max_steps": 1,
            "packing": False,
            "completion_only_loss": True,
            "assistant_only_loss": False,
            "prompt_render": "prompt_completion",
            "require_memory_efficient_loss": True,
        },
        dataset_config={
            "schema_version": "syntunia-sft-row/v2",
            "format": "messages",
            "use_preassigned_splits": True,
            "split_dataset": False,
        },
    )
    assert command[command.index("--model-revision") + 1] == revision
    assert command[command.index("--max-seq-length") + 1] == "32768"
    assert "--require-memory-efficient-loss" in command


def test_local_run_sft_32k_fails_closed_without_loss_guard(tmp_path):
    with pytest.raises(LocalRunError, match="SFT_LONG_CONTEXT_LOSS_GUARD_REQUIRED"):
        _compile_local_command(
            tmp_path,
            method="sft",
            trainer="Trainers/sft/train_sft.py",
            model_config={"max_seq_length": 32768},
            training={
                "max_steps": 1,
                "packing": False,
                "completion_only_loss": True,
                "assistant_only_loss": False,
                "prompt_render": "prompt_completion",
            },
            dataset_config={
                "schema_version": "syntunia-sft-row/v2",
                "format": "messages",
                "use_preassigned_splits": True,
            },
        )


@pytest.mark.parametrize(
    "dataset_config,message",
    [
        (
            {"format": "messages", "use_preassigned_splits": True},
            "schema_version",
        ),
        (
            {"schema_version": "syntunia-sft-row/v2", "use_preassigned_splits": True},
            "format='messages'",
        ),
        (
            {"schema_version": "syntunia-sft-row/v2", "format": "messages"},
            "use_preassigned_splits=true",
        ),
        (
            {
                "schema_version": "syntunia-sft-row/v2",
                "format": "messages",
                "use_preassigned_splits": True,
                "split_dataset": True,
            },
            "cannot use random",
        ),
    ],
)
def test_local_run_sft_rejects_incomplete_or_random_authoritative_message_splits(
    tmp_path, dataset_config, message
):
    with pytest.raises(LocalRunError, match=message):
        _compile_local_command(
            tmp_path,
            method="sft",
            trainer="Trainers/sft/train_sft.py",
            training={"max_steps": 1},
            dataset_config=dataset_config,
        )
