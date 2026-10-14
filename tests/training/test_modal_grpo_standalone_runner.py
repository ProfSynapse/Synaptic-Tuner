"""Ordinary GRPO host join with explicit CPU image/provider/ML observations.

The CPU protocol, signed receipt, retained preparation, public lifecycle, isolated
owned trainer, native verifier and signed artifact readback remain real. These
controls do not qualify an image, GPU, billing, or physical provider isolation.
"""

from contextlib import contextmanager, ExitStack
from argparse import Namespace
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import pytest
import yaml

from tests.training.test_modal_grpo_recipe import setup
from tests.training.test_modal_standalone_runner import _facts
from tests.runtime import cpu_qualification_image as image
from tests.runtime.test_packaged_cpu_methods import BOOTSTRAP, HELPERS
from tests.runtime.test_packaged_grpo_child import _BOOTSTRAP, _prepare
from tuner.project.context import ProjectContext
from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV2
from tuner.training.modal_recipe import plan_modal_recipe
from tuner.training import modal_local_preparation as local
from tuner.training import modal_standalone_runner as runner
from tuner.execution.providers.modal import runtime_release_deployment as deployment
from tuner.execution.providers.modal.runtime_release_qualification import (
    ModalRuntimeReleaseQualificationWorker,
    ModalRuntimeReleaseQualificationRoots,
    ModalRuntimeQualificationHmacAuthenticator,
)
from tuner.execution.providers.modal.packaged_worker import (
    ModalPackagedWorker,
    ModalPackagedWorkerRoots,
    InstalledPackagedSFTTrainerExecutor,
)
from tuner.training.modal_host_authority import HMACAuthenticator
from tuner.training.modal_host_runtime import ModalHostRuntimeV1
from tests.runtime.test_packaged_runtime_releases import _release_values

SOURCE = Path(runner.__file__).resolve().parents[2]


def _runtime(release, plan):
    facts = _facts(release)
    specs = (
        deployment.ModalRuntimeReleaseFunctionSpecV1(
            "training",
            "packaged-training",
            deployment.EXACT_PACKAGED_TRAINING_MODULE,
            deployment.EXACT_PACKAGED_TRAINING_QUALNAME,
            ("control", "artifacts", "model_cache"),
            ("qualification", "hf-token"),
            1000,
            4096,
            plan.recipe.timeout_seconds,
            plan.recipe.accelerator,
            False,
            restrict_modal_access=False,
        ),
        deployment.ModalRuntimeReleaseFunctionSpecV1(
            "self_check",
            "packaged-self-check",
            deployment.EXACT_SELF_CHECK_MODULE,
            deployment.EXACT_SELF_CHECK_QUALNAME,
            ("control", "artifacts"),
            ("qualification",),
            1000,
            512,
            120,
            None,
            True,
            restrict_modal_access=False,
        ),
    )
    actual = deployment.ModalRuntimeReleaseDeploymentFactsV1(
        deployment_spec_digest=facts.deployment_spec_digest,
        runtime_release_digest=release.manifest_digest,
        client_binding=facts.client_binding,
        app_name=facts.app_name,
        app_id=facts.app_id,
        generation=facts.deployment_generation,
        image_reference=None,
        image_digest=None,
        image_id=facts.image_id,
        functions=tuple(
            deployment.ModalRuntimeReleaseFunctionFactV1(spec, identifier)
            for spec, identifier in zip(
                specs, (facts.function_id, facts.self_check_function_id), strict=True
            )
        ),
        volumes=tuple(
            deployment.ModalRuntimeReleaseVolumeFactV1(
                deployment.ModalRuntimeReleaseVolumeSpecV1(role, name, mount),
                identifier,
            )
            for role, name, mount, identifier in (
                ("control", "control", "/mnt/control", facts.control_volume_id),
                ("artifacts", "artifacts", "/mnt/artifacts", facts.artifact_volume_id),
                (
                    "model_cache",
                    "cache",
                    "/mnt/model-cache",
                    facts.model_cache_volume_id,
                ),
            )
        ),
        secrets=(
            deployment.ModalRuntimeReleaseSecretFactV1(
                deployment.ModalRuntimeReleaseSecretSpecV1(
                    "qualification", deployment.EXACT_QUALIFICATION_SECRET_REQUIRED_KEYS
                ),
                "st-qualification",
            ),
            deployment.ModalRuntimeReleaseSecretFactV1(
                deployment.ModalRuntimeReleaseSecretSpecV1("hf-token", ("HF_TOKEN",)),
                "st-hftoken",
            ),
        ),
        python_implementation=release.python_implementation,
        python_version=release.python_version,
        python_executable=release.python_executable,
        python_executable_digest=release.python_executable_digest,
        package_digest=release.package_digest,
        installed_distributions_digest=release.installed_distributions_digest,
        worker_entrypoint=release.worker_entrypoint,
        worker_closure_digest=release.worker_closure_digest,
        material_digest=release.material_digest,
        schema_version=deployment.MODAL_RUNTIME_RELEASE_DEPLOYMENT_FACTS_V2_SCHEMA,
    )
    return ModalHostRuntimeV1(
        release,
        actual,
        facts,
        SimpleNamespace(gpu_only_timeout_estimate_minor_units=1, quote_digest="e" * 64),
        (
            ("vo-control", "control"),
            ("vo-artifacts", "artifacts"),
            ("vo-cache", "cache"),
        ),
        b"k" * 32,
        SimpleNamespace(observe=lambda **kwargs: facts),
    )


def _joined(
    tmp_path,
    monkeypatch,
    request,
    *,
    remove_source=False,
    cpu_bad_mac=False,
    lose_submit_reply=False,
):
    project, engine, profiles, publication, recipe, declaration, _ = setup(tmp_path)
    # Existing explicit ML fixture asserts this auxiliary retained column. It
    # remains source data admitted through the real generic normalizer.
    (tmp_path / "source/rows.jsonl").write_bytes(
        b'{"prompt":"Say OK","answer":"OK","declared":"retained"}\n'
    )
    from tuner.training.grpo_preparation import (
        GRPOTrainingInputNormalizerV1,
        GRPONormalizerConfigV1,
    )
    from synaptic_tuner.api.v1.training_sources import LocalTrainingInputPathV1

    publication = (
        GRPOTrainingInputNormalizerV1(allowed_source_roots=(tmp_path / "source",))
        .prepare(
            LocalTrainingInputPathV1(tmp_path / "source"),
            config=GRPONormalizerConfigV1("rows.jsonl", "prompt"),
            output_root=project / ".tracking/datasets",
        )
        .path
    )
    manifest = json.loads((publication / "manifest.json").read_bytes())
    declaration["dataset"]["local_file"] = str(
        (publication / "dataset.jsonl").relative_to(project)
    )
    declaration["dataset"]["expected_digest"] = manifest["semantic_digest"]
    declaration["training_input"]["dataset"]["ref"] = (
        "prepared://sha256/" + manifest["semantic_digest"]
    )
    recipe.write_text(yaml.safe_dump(declaration))
    buildpath = engine / "Trainers/image_profiles/fixture_grpo/profile.yaml"
    build = yaml.safe_load(buildpath.read_bytes())
    # Explicit CPU image identity; no locked launcher/image is changed.
    build["python_executable"] = sys.executable
    build["packaged_runtime"]["python"]["executable"] = sys.executable
    buildpath.write_text(yaml.safe_dump(build))
    subprocess.run(
        ["git", "-C", str(engine), "add", "."],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(engine),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "CPU interpreter observation",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    plan = plan_modal_recipe(
        recipe, profiles_root=profiles, project_root=project, source_root=engine
    )
    context = ProjectContext.host(
        project_root=project, engine_root=engine, invocation_cwd=project
    )
    image_root = tmp_path / "image"
    image_root.mkdir(mode=0o700)
    config_path, config = image.capture(SOURCE, image_root)
    # Local custody also checks the real selected inspector; include it in the
    # synthetic wheel instead of substituting that source/wheel consistency guard.
    wheel = Path(config["wheel"])
    with ZipFile(wheel, "a", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "tuner/runtime/modal_build_inspector.py",
            (engine / "tuner/runtime/modal_build_inspector.py").read_bytes(),
        )
    config["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    config["wheel_names"].append("tuner/runtime/modal_build_inspector.py")
    config_path.write_bytes(image.canonical(config))
    image.install(SOURCE, config, monkeypatch.setattr)
    monkeypatch.setattr(local, "validate_local_launcher", lambda: None)
    monkeypatch.setattr(runner, "validate_local_launcher", lambda: None)
    monkeypatch.setattr(runner.platform, "python_version", lambda: "3.11.14")
    monkeypatch.setenv("HF_TOKEN", "fixture-token")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    bootstrap = tmp_path / "fixture_bootstrap-1.0.0-py3-none-any.whl"
    bootstrap.write_bytes(b"fixture-bootstrap")
    bootstrap.chmod(0o600)
    monkeypatch.setattr(
        local, "bootstrap_paths", lambda *args, **kwargs: {bootstrap.name: bootstrap}
    )

    def builder(root, staged, **kwargs):
        assert root == engine
        target = staged / "synaptic_tuner-1.1.0-py3-none-any.whl"
        target.write_bytes(wheel.read_bytes())
        return target, config["wheel_sha256"]

    monkeypatch.setattr(local, "prepare_current_source_wheel", builder)
    events, holders = [], []
    original_local = runner.prepare_modal_standalone_local
    runtime_slot = {}

    def prepare(**kwargs):
        holder = original_local(**kwargs)
        holders.append(holder)
        values = _release_values(
            worker_entrypoint="tuner.runtime.packaged_training_worker:main",
            worker_closure_digest=config["grpo"]["closure_digest"],
            package_digest=config["wheel_sha256"],
            python_version="3.11.14",
            python_executable=sys.executable,
            compatible_methods=("grpo",),
            compatible_models=(("model/example", "a" * 40),),
            compatible_dataset_formats=("synaptic-grpo-prompts/v1",),
            workload_schema="synaptic-packaged-grpo-workload/v1",
            artifact_contract_schema="synaptic-grpo-artifacts/v1",
        )
        values.pop("image_ref")
        values.pop("image_digest")
        runtime_slot["runtime"] = _runtime(
            PackagedTrainingRuntimeReleaseV2.build(**values, material=holder.layout[5]),
            plan,
        )
        events.append("local")
        return holder

    monkeypatch.setattr(runner, "prepare_modal_standalone_local", prepare)
    client = SimpleNamespace(stub=object())
    owned = TemporaryDirectory(prefix="modal-grpo-join-", dir=Path.home())
    request.addfinalizer(owned.cleanup)
    base = Path(owned.name)
    volume_root, mounts = base / "volumes", base / "mounts"
    volume_root.mkdir(mode=0o700)
    mounts.mkdir(mode=0o700)
    ids = {"control": "vo-control", "artifacts": "vo-artifacts", "cache": "vo-cache"}
    roots = {name: volume_root / identifier for name, identifier in ids.items()}
    for name, root in roots.items():
        root.mkdir(mode=0o700)
        (mounts / name).symlink_to(root, target_is_directory=True)
    from tuner.execution.providers.modal import volume_root_binding as root_binding

    monkeypatch.setattr(root_binding, "_PROVIDER_VOLUME_ROOT", str(volume_root))
    results = {}

    class NotFoundError(Exception):
        pass

    class Volume:
        ids = {
            "control": "vo-control",
            "artifacts": "vo-artifacts",
            "cache": "vo-cache",
        }

        def __init__(self, name):
            self.name, self.object_id, self.is_hydrated = name, self.ids[name], False

        @classmethod
        def from_name(cls, name, **kwargs):
            assert kwargs == dict(
                environment_name="main",
                create_if_missing=False,
                version=1,
                client=client,
            )
            return cls(name)

        def hydrate(self, supplied):
            assert supplied is client
            self.is_hydrated = True
            return self

        @contextmanager
        def batch_upload(self, *, force):
            assert force is False
            pending = []
            batch = SimpleNamespace(
                put_file=lambda stream, path: pending.append((path, stream.read()))
            )
            yield batch
            for path, raw in pending:
                leaf = roots[self.name] / path
                if leaf.exists():
                    raise ValueError("fixture collision")
                leaf.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
                leaf.write_bytes(raw)

        def read_file(self, path):
            yield (roots[self.name] / path).read_bytes()

        def iterdir(self, prefix, recursive=True):
            root = roots[self.name] / prefix.rstrip("/")
            if not root.exists():
                raise NotFoundError
            paths = (
                [root]
                if root.is_file()
                else sorted(root.rglob("*") if recursive else root.iterdir())
            )
            for path in paths:
                if path.is_file():
                    yield SimpleNamespace(
                        path=path.relative_to(roots[self.name]).as_posix(),
                        type=1,
                        size=path.stat().st_size,
                        mtime=1,
                    )

    def current():
        facts = runtime_slot["runtime"].deployment_facts
        return deployment.ModalRuntimeReleaseDeploymentObservationV1(
            facts.app_name,
            "main",
            True,
            facts.app_id,
            "",
            facts.generation,
            tuple(sorted((row.spec.name, row.function_id) for row in facts.functions)),
        )

    class Function:
        def __init__(self, name):
            self.name, self.object_id, self.is_hydrated = (
                name,
                ("fu-check" if name == "packaged-self-check" else "fu-training"),
                False,
            )

        @classmethod
        def from_name(cls, app, name, **kwargs):
            assert app == "owned-app" and kwargs == dict(
                environment_name="main", client=client
            )
            return cls(name)

        def hydrate(self, supplied):
            assert supplied is client
            self.is_hydrated = True
            return self

        def spawn(self, payload):
            runtime = runtime_slot["runtime"]
            if self.name == "packaged-self-check":
                events.append("cpu")
                auth = ModalRuntimeQualificationHmacAuthenticator(
                    runtime.qualification_key
                )
                worker = ModalRuntimeReleaseQualificationWorker(
                    expected_facts=runtime.deployment_facts,
                    verifier=auth,
                    signer=auth,
                    observer=SimpleNamespace(observe=lambda facts: current()),
                    call_id_provider=lambda: "fc-cpu",
                    roots=ModalRuntimeReleaseQualificationRoots(
                        roots["control"], roots["artifacts"]
                    ),
                )
                result = worker(
                    payload, commit_control=lambda: None, commit_artifacts=lambda: None
                )
                identifier = "fc-cpu"
                if cpu_bad_mac:
                    tag = next(roots["control"].rglob("receipt.mac"))
                    tag.write_bytes(b"x" * 32)
                if remove_source:
                    (publication / "dataset.jsonl").unlink()
                    (tmp_path / "source/rows.jsonl").unlink()
            else:
                events.append("training")
                auth = HMACAuthenticator(
                    {
                        "modal-runtime-release-qualification-hmac-v1": runtime.qualification_key
                    },
                    allowed_purposes=frozenset(
                        {
                            "modal-packaged-dispatch/v1",
                            "modal-packaged-dispatch/v2",
                            "modal-packaged-completion/v1",
                        }
                    ),
                )
                from tuner.execution.providers.modal.packaged_dispatch import (
                    parse_modal_packaged_dispatch,
                )

                dispatch = parse_modal_packaged_dispatch(payload, auth)
                scratch = base / "training-scratch"
                scratch.mkdir(mode=0o700)
                with ExitStack() as stack:
                    bindings = {}
                    for marker in dispatch.volume_markers:
                        name = "cache" if marker.role == "model_cache" else marker.role
                        bindings[marker.role] = stack.enter_context(
                            root_binding.VolumeRootBinding.bind(
                                root_path=str(mounts / name),
                                volume_id=marker.volume_id,
                                marker_name=marker.marker_name,
                                marker_sha256=marker.value_sha256,
                            )
                        )
                    worker = ModalPackagedWorker(
                        expected_facts=runtime.facts,
                        dispatch_verifier=auth,
                        trainer_executor=InstalledPackagedSFTTrainerExecutor(
                            model_preparer=_prepare
                        ),
                        evidence_signer=auth,
                        roots=ModalPackagedWorkerRoots(
                            roots["control"], roots["artifacts"], roots["cache"]
                        ),
                        volume_bindings=bindings,
                        private_root=scratch,
                    )
                    identifier = "fc-training"
                    result = worker(
                        payload,
                        provider_job_ref=identifier,
                        commit_control=lambda: None,
                        commit_artifacts=lambda: None,
                    )
            results[identifier] = result
            if identifier == "fc-training" and lose_submit_reply:
                raise TimeoutError("explicit lost reply fixture")
            return SimpleNamespace(object_id=identifier)

    class FunctionCall:
        def __init__(self, identifier):
            self.object_id, self.is_hydrated, self.client = identifier, False, client

        @classmethod
        def from_id(cls, identifier, *, client):
            return cls(identifier)

        def hydrate(self, supplied):
            assert supplied is client
            self.is_hydrated = True
            return self

        def get(self, *, timeout):
            assert timeout == 0 and self.object_id in results
            return results[self.object_id]

    sdk = SimpleNamespace(
        __version__="1.5.4",
        Volume=Volume,
        Function=Function,
        FunctionCall=FunctionCall,
        exception=SimpleNamespace(NotFoundError=NotFoundError),
    )
    monkeypatch.setitem(sys.modules, "modal", sdk)
    from tuner.execution.providers.modal.facade import (
        ExplicitModal154ReadFacade, ModalFunctionCallState,
    )

    def observe_status(self, identifier, *, deadline):
        # This completed-worker SDK double has output only for its captured
        # call IDs. Actual raw protobuf status validation has pinned-SDK tests.
        assert self.client is client
        return (ModalFunctionCallState.RETURNED if identifier in results
                else ModalFunctionCallState.UNKNOWN)

    monkeypatch.setattr(
        ExplicitModal154ReadFacade, "observe_known_call_status", observe_status
    )
    async_utils = ModuleType("modal._utils.async_utils")
    import asyncio

    async_utils.synchronizer = SimpleNamespace(
        create_blocking=lambda operation: (
            lambda *args, **kwargs: asyncio.run(operation(*args, **kwargs))
        )
    )
    monkeypatch.setitem(sys.modules, "modal._utils", ModuleType("modal._utils"))
    monkeypatch.setitem(sys.modules, "modal._utils.async_utils", async_utils)
    scope = lambda **kwargs: ("workspace", "workspace", "main", "train-config-host")
    monkeypatch.setattr(
        runner,
        "open_modal_host_scope",
        lambda **kwargs: (client, runtime_slot["runtime"].facts.client_binding),
    )
    monkeypatch.setattr(runner, "observe_modal_host_scope", scope)
    monkeypatch.setattr(
        "tuner.training.modal_host_qualification.observe_modal_host_scope", scope
    )
    monkeypatch.setattr(
        deployment.ExplicitModal154ReleaseDeploymentReader,
        "observe",
        lambda *args, **kwargs: current(),
    )

    def bootstrap_runtime(**kwargs):
        assert kwargs["local_inputs"] is holders[0]
        events.append("bootstrap")
        return runtime_slot["runtime"]

    monkeypatch.setattr(runner, "prepare_modal_runtime_for_host", bootstrap_runtime)
    original_run = subprocess.run
    cpu_calls, trainer_calls = [], []

    def run(command, **kwargs):
        if command[:4] == [
            sys.executable,
            "-I",
            "-m",
            "tuner.runtime.packaged_sft_child",
        ]:
            cpu_calls.append(command)
            command = [
                sys.executable,
                "-I",
                "-B",
                "-c",
                BOOTSTRAP,
                str(SOURCE),
                str(HELPERS),
                str(config_path),
                *command[4:],
            ]
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    from tuner.runtime import packaged_sft_execution as shared

    monkeypatch.setattr(
        shared, "_inspect_parent_release", lambda release, **kwargs: None
    )  # Explicit image observation only.
    original_popen = subprocess.Popen

    def spawn(command, **kwargs):
        assert tuple(command[:5]) == (
            sys.executable,
            "-I",
            "-B",
            "-m",
            "tuner.runtime.packaged_grpo_child",
        )
        trainer_calls.append(command)
        return original_popen(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _BOOTSTRAP,
                str(SOURCE),
                str(HELPERS),
                *command[5:],
            ],
            **kwargs,
        )

    monkeypatch.setattr(
        shared,
        "subprocess",
        SimpleNamespace(
            Popen=spawn,
            DEVNULL=subprocess.DEVNULL,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    from tuner.execution.providers.modal.bounded_volume_read import (
        BoundedModalVolumeReader,
    )

    monkeypatch.setattr(
        BoundedModalVolumeReader, "__init__", lambda *args, **kwargs: None
    )

    # The mounted provider files are shared with the real worker; only SDK I/O is substituted.
    async def marker(
        reader, *, volume_id, path, expected_size, expected_sha256, max_bytes
    ):
        name = next(name for name in roots if Volume.ids[name] == volume_id)
        raw = (roots[name] / path).read_bytes()
        assert (
            len(raw) == expected_size
            and hashlib.sha256(raw).hexdigest() == expected_sha256
        )
        return raw

    monkeypatch.setattr(BoundedModalVolumeReader, "read_exact", marker)
    return plan, context, events, holders, cpu_calls, trainer_calls, publication, recipe


def test_ordinary_grpo_runner_joins_real_cpu_native_public_path(
    tmp_path, monkeypatch, request
):
    plan, context, events, holders, cpu, trainer, _, _ = _joined(
        tmp_path, monkeypatch, request, remove_source=True
    )
    result = runner.run_modal_standalone_job(
        plan=plan, context=context, modal_profile="fixture", modal_environment="main"
    )
    assert len(result.artifact_paths) == 5
    assert all(path.is_file() for path in result.artifact_paths)
    assert events == ["local", "bootstrap", "cpu", "training"]
    assert len(cpu) == len(trainer) == 1
    with pytest.raises(Exception):
        holders[0].revalidate(
            holders[0].profile_path, plan.runtime_material_intent_digest
        )


def test_cpu_report_method_tags_preserve_sft_and_reject_unknown():
    report = runner.ModalStandaloneCPUQualificationV1("a" * 64, "b" * 64, "c" * 64)
    assert (
        report.to_dict()["schema_version"] == "synaptic-modal-sft-cpu-qualification/v1"
    )
    assert (
        runner.ModalStandaloneCPUQualificationV1(
            "a" * 64, "b" * 64, "c" * 64, method="grpo"
        ).to_dict()["schema_version"]
        == "synaptic-modal-grpo-cpu-qualification/v1"
    )
    with pytest.raises(ValueError):
        runner.ModalStandaloneCPUQualificationV1(
            "a" * 64, "b" * 64, "c" * 64, method="other"
        ).to_dict()


@pytest.mark.parametrize("mode", ("execute", "qualify"))
def test_actual_cli_grpo_execute_and_cpu_qualify_routes(
    tmp_path, monkeypatch, request, mode
):
    from tuner.handlers.modal_job_config_handler import ModalJobConfigHandler

    plan, context, events, holders, cpu, trainer, _, recipe = _joined(
        tmp_path, monkeypatch, request
    )
    args = Namespace(
        job_config=str(recipe),
        json=True,
        modal_profile="fixture",
        modal_environment="main",
        **({"qualify": True} if mode == "qualify" else {}),
    )
    handler = ModalJobConfigHandler(args, context=context)
    outputs, errors = [], []
    monkeypatch.setattr(
        handler, "output", lambda value, **kwargs: outputs.append(value)
    )
    monkeypatch.setattr(
        handler, "output_error", lambda message, **kwargs: errors.append(kwargs)
    )
    assert handler.handle() == 0, errors
    assert not errors and len(outputs) == 1
    if mode == "execute":
        assert outputs[0]["schema_version"] == "synaptic-modal-grpo-run-result/v1"
        assert len(outputs[0]["verified_artifacts"]) == 5
        assert len(trainer) == 1 and events == ["local", "bootstrap", "cpu", "training"]
    else:
        assert (
            outputs[0]["schema_version"] == "synaptic-modal-grpo-cpu-qualification/v1"
        )
        assert (
            outputs[0]["training_executed"] is False
            and outputs[0]["gpu_qualified"] is False
        )
        assert not trainer and events == ["local", "bootstrap", "cpu"]
    assert len(cpu) == 1
    with pytest.raises(Exception):
        holders[0].revalidate(
            holders[0].profile_path, plan.runtime_material_intent_digest
        )


def test_unsupported_tokenizer_snapshot_rejects_before_local_or_provider(
    tmp_path, monkeypatch
):
    project, engine, profiles, _, recipe, _, _ = setup(tmp_path)
    plan = plan_modal_recipe(
        recipe, profiles_root=profiles, project_root=project, source_root=engine
    )
    from synaptic_tuner.api.v1.training_input import TrainingInputV1
    from tuner.training.grpo_compilation import compile_grpo_input

    declaration = plan.recipe.declaration.to_dict()
    declaration["model"]["tokenizer_revision"] = "b" * 40
    changed_recipe = replace(
        plan.recipe, declaration=TrainingInputV1.from_dict(declaration)
    )
    changed = replace(
        plan,
        recipe=changed_recipe,
        core_workload_digest=compile_grpo_input(
            changed_recipe.declaration, plan.prepared_identity
        ).fingerprint,
    )
    monkeypatch.setattr(runner.platform, "python_version", lambda: "3.11.14")
    monkeypatch.setattr(
        runner,
        "prepare_modal_standalone_local",
        lambda **kwargs: pytest.fail("local effect"),
    )
    context = ProjectContext.host(
        project_root=project, engine_root=engine, invocation_cwd=project
    )
    with pytest.raises(
        runner.ModalStandaloneRunUnavailable, match="modal_host_configuration_invalid"
    ):
        runner.run_modal_standalone_job(
            plan=changed,
            context=context,
            modal_profile="fixture",
            modal_environment="main",
        )


@pytest.mark.parametrize("mode", ("quote", "metrics"))
def test_grpo_quote_and_metrics_still_reject_before_provider(
    tmp_path, monkeypatch, mode
):
    from tuner.handlers.modal_job_config_handler import ModalJobConfigHandler

    project, engine, _, _, recipe, _, _ = setup(tmp_path)
    context = ProjectContext.host(
        project_root=project, engine_root=engine, invocation_cwd=project
    )
    args = Namespace(
        job_config=str(recipe),
        json=True,
        **(
            {"quote": True}
            if mode == "quote"
            else {"plan": True, "metrics_qualification_attempt": "attempt-fresh"}
        ),
    )
    handler = ModalJobConfigHandler(args, context=context)
    errors = []
    monkeypatch.setattr(
        handler, "output_error", lambda message, **kwargs: errors.append(kwargs)
    )
    monkeypatch.setattr(handler, "_quote", lambda *args: pytest.fail("provider quote"))
    monkeypatch.setattr(handler, "_execute", lambda *args: pytest.fail("execute"))
    assert handler.handle() == 2
    assert errors[-1]["code"] == "MODAL_GRPO_DISPATCH_UNSUPPORTED"


@pytest.mark.parametrize("fault", ("cpu_mac", "submit_reply"))
def test_consumed_failure_never_replays_or_signs_fake_success(
    tmp_path, monkeypatch, request, fault
):
    from tuner.training.modal_host_qualification import (
        ModalHostQualificationUnavailable,
    )

    plan, context, events, holders, cpu, trainer, _, _ = _joined(
        tmp_path,
        monkeypatch,
        request,
        cpu_bad_mac=fault == "cpu_mac",
        lose_submit_reply=fault == "submit_reply",
    )
    expected = (
        ModalHostQualificationUnavailable
        if fault == "cpu_mac"
        else runner.ModalStandalonePhaseUnavailable
    )
    with pytest.raises(expected) as error:
        runner.run_modal_standalone_job(
            plan=plan,
            context=context,
            modal_profile="fixture",
            modal_environment="main",
        )
    assert len(cpu) == 1
    if fault == "cpu_mac":
        assert error.value.phase == "RECEIPT_VERIFY"
        assert events == ["local", "bootstrap", "cpu"] and not trainer
    else:
        assert error.value.phase == "RUN_SUBMIT_RECONCILE_REQUIRED"
        assert error.value.retained_run is not None
        assert events == ["local", "bootstrap", "cpu", "training"] and len(trainer) == 1
        # Absence of a captured call ID is not evidence that the completed
        # provider fixture did not execute. The claimed submit stays uncertain.
        run = error.value.retained_run
        root = (
            Path(__import__("os").environ["XDG_STATE_HOME"])
            / "synaptic-training"
            / run.run_id
        )
        assert (root / "workflow.sqlite3").is_file()
    with pytest.raises(Exception):
        holders[0].revalidate(
            holders[0].profile_path, plan.runtime_material_intent_digest
        )
