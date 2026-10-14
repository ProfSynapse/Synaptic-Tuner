"""Real Git topology, publisher and compiler; source transport/key are CPU fixtures."""

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import tuner

from tuner.dataset_prep import DatasetPrepConfigV1, prepare_dataset_v1
from tuner.ingestion import NormalizedItemInputV1, write_normalized_bundle_v1
from tuner.handlers.ingestion_handler import _structures
from tuner.project.git_verification import GitLsRemotePushedCommitVerifier
from tuner.project.errors import SourceLockError
from tuner.training.contracts import CanonicalDocument
from tuner.training.coordinator_material import CoordinatorResolvedMaterial
from tuner.training import default_recipe_registry
from tuner.runtime_profiles import load_runtime_profile


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = Path(
    os.environ.get(
        "SYNAPTIC_TEST_HF_PREPARATION_EXAMPLE",
        ROOT / "examples/training_qualification/prepare_hf.py",
    )
)
spec = importlib.util.spec_from_file_location("hf_prepare_example", EXAMPLE)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
ASSETS = EXAMPLE.parent / "hf_raw_text_fixture"
PACKAGE = Path(tuner.__file__).resolve().parent.parent


def canonical(value):
    return CanonicalDocument.from_mapping(value).canonical_json.encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(root, *args):
    return (
        subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, timeout=15
        )
        .stdout.decode()
        .strip()
    )


def repo(root):
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Synthetic test owner")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(
        root,
        "remote",
        "add",
        "origin",
        "https://github.com/ProfSynapse/Synaptic-Tuner.git",
    )


def publication(tmp):
    config = json.loads((ASSETS / "ingestion.json").read_bytes())
    structure = _structures(config)
    items = []
    for p in sorted((ASSETS / "sources").glob("*.md")):
        raw = p.read_bytes()
        logical = "fixture/" + p.name
        items.append(
            NormalizedItemInputV1(
                logical_path=logical,
                source_sha256=sha(raw),
                source_size_bytes=len(raw),
                structure_ref=structure.structures[0].ref.to_dict(),
                fields={"body": raw.decode(), "path": logical},
            )
        )
    bundle = write_normalized_bundle_v1(
        tmp / "bundles", structure.to_dict(), tuple(items)
    )
    value = json.loads((ASSETS / "preparation.json").read_bytes())
    value["source"]["bundle_path"] = str(bundle.path)
    assert (
        bundle.semantic_identity.bundle_digest
        == value["source"]["expected_bundle_digest"]
    )
    return prepare_dataset_v1(DatasetPrepConfigV1.from_dict(value), tmp / "datasets")


@pytest.fixture
def configured(tmp_path):
    published = publication(tmp_path)
    profile = load_runtime_profile(
        "qwen35-sft-v1", PACKAGE / "Trainers/runtime_profiles"
    )
    host = tmp_path / "host"
    repo(host)
    engine = host / "engine"
    repo(engine)
    image_profile = (
        engine
        / "Trainers/image_profiles"
        / profile.packaged_build_profile
        / "profile.yaml"
    )
    image_profile.parent.mkdir(parents=True)
    original = Path(
        os.environ.get(
            "SYNAPTIC_TEST_HF_IMAGE_PROFILE",
            ROOT
            / "Trainers/image_profiles/qwen35_4b_packaged_sft_3360351c/profile.yaml",
        )
    )
    shutil.copyfile(original, image_profile)
    git(engine, "add", ".")
    git(engine, "commit", "-qm", "Synthetic declared engine profile")
    (host / ".gitmodules").write_text(
        '[submodule "engine"]\n\tpath = engine\n\turl = https://github.com/ProfSynapse/Synaptic-Tuner.git\n\tbranch = main\n'
    )
    # The caller manifest is a real, committed valid project document.
    manifest = {
        "schema_version": "synaptic-project/v1",
        "project": {"id": "hf-prepare-test", "name": "HF preparation fixture"},
        "engine": {"requires": ">=1.0,<2.0", "api": "v1"},
        "paths": {
            name: "project://.synaptic/" + name
            for name in ("configs", "artifacts", "state", "tracking", "cache", "tmp")
        },
    }
    import yaml

    (host / "synaptic.yaml").write_text(yaml.safe_dump(manifest))
    git(host, "add", ".gitmodules", "synaptic.yaml")
    git(
        host,
        "update-index",
        "--add",
        "--cacheinfo",
        "160000",
        git(engine, "rev-parse", "HEAD"),
        "engine",
    )
    git(host, "commit", "-qm", "Synthetic host with real gitlink")
    normalized = tmp_path / "sft.json"
    normalized.write_bytes((ASSETS / "sft-config.json").read_bytes())
    selection = {
        "scope": {"account_ref": "fixture-owner", "namespace_ref": "fixture-owner"},
        "bucket_id": "fixture-owner/test-bucket",
        "source_prefix": "source/test",
        "run_prefix": "runs/test",
        "flavor": "l40sx1",
        "timeout_seconds": 300,
        "host_token": {"provider": "env", "name": "HF_TOKEN"},
        "job_token": {"provider": "env", "name": "HF_TOKEN"},
        "quote_digest": sha(b"Explicit synthetic CPU quote, not provider pricing"),
    }
    consumer = {
        "provider": "hf_jobs",
        "state_root": str(tmp_path / "attempt"),
        "host": {
            "authority_secret": {"provider": "env", "name": "AUTHORITY"},
            "evidence_secret": {"provider": "env", "name": "READER"},
            "grant_ref": "fixture-grant",
            "allowed_requirements": [
                {
                    "operation": "training.start",
                    "paid_effect": True,
                    "maximum_cost_minor_units": 100,
                    "currency": "USD",
                }
            ],
        },
        "hf_jobs": {
            "provider": {"provider_id": "hf_jobs", "profile_ref": "fixture-profile"},
            "runtime_profile": {
                "name": profile.name,
                "root": str(profile.path.parent),
                "sha256": profile.profile_sha256.removeprefix("sha256:"),
            },
            "selection": selection,
            "reader_authority_ref": "reader",
            "reader_key_ref": "reader-key",
            "maximum_cost_minor_units": 100,
            "currency": "USD",
            "read_seconds": 30,
            "resources": {
                "accelerator": "gpu",
                "accelerator_count": 1,
                "timeout_seconds": 300,
            },
        },
    }
    pins = yaml.safe_load(image_profile.read_text())["packaged_runtime"]["python"]
    declaration = {
        "schema_version": "hf-runtime-declaration/v1",
        "runtime_profile": {"name": profile.name, "sha256": profile.profile_sha256},
        "inventory_sha256": sha(profile.inventory_path.read_bytes()),
        "image": profile.image,
        "interpreter": {
            k: pins[k]
            for k in ("implementation", "version", "executable", "executable_digest")
        },
        "roots": {
            n: "/workspace/private/"
            + ("" if n in {"engine", "project"} else "run/")
            + n
            for n in (
                "engine",
                "project",
                "artifacts",
                "state",
                "tracking",
                "cache",
                "tmp",
            )
        },
        "writable_capability_root": "/workspace/private/run",
        "environment": {
            "PATH": "/opt/unsloth-venv/bin:/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
        },
        "secret_requirements_digest": sha(
            canonical({k: selection[k] for k in ("host_token", "job_token")})
        ),
    }
    declaration["provider_runtime_requirements_digest"] = sha(
        canonical(
            {
                k: declaration[k]
                for k in ("runtime_profile", "inventory_sha256", "image", "interpreter")
            }
        )
    )
    roots = declaration["roots"]
    declaration["environment"].update(
        {
            "PYTHONSAFEPATH": "1",
            "PYTHONPATH": roots["engine"],
            **{
                "SYNAPTIC_" + name.upper() + "_ROOT": roots[key]
                for name, key in (
                    ("engine", "engine"),
                    ("project", "project"),
                    ("artifact", "artifacts"),
                    ("state", "state"),
                    ("tracking", "tracking"),
                    ("cache", "cache"),
                    ("tmp", "tmp"),
                )
            },
            "HF_HOME": roots["cache"] + "/huggingface",
            "TRANSFORMERS_CACHE": roots["cache"] + "/transformers",
            "WANDB_DISABLED": "true",
        }
    )
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_bytes(canonical(declaration))
    config = {
        "project": {
            "root": str(host),
            "engine_root": str(engine),
            "manifest": str(host / "synaptic.yaml"),
        },
        "run_id": "hf-prepare-fixture",
        "request_id": "hf-prepare-request",
        "output_root": str(tmp_path / "prepared"),
        "prepared_publication": str(published.path),
        "resolved_config": str(normalized),
        "runtime_declaration": str(runtime_path),
        "runtime_environment": declaration["environment"],
        "image_profile": str(image_profile),
        "source_authority": {
            "secret": {"provider": "env", "name": "SOURCE_KEY"},
            "issuer_ref": "fixture-source",
            "key_ref": "source-key",
            "audience_ref": "hf-prepare",
        },
        "consumer": consumer,
    }
    clock = helper._example_module("hf_test_clock", "authority.py").UTCClock()
    auth = helper._example_module("hf_test_auth", "authority.py").HMACAuthenticator(
        {"source-key": b"synthetic-host-source-signing-key!"},
        allowed_purposes=frozenset({helper.SOURCE_EVIDENCE_PURPOSE}),
    )

    class Remote:
        calls = []

        def read_ref(self, *, canonical_url, exact_ref):
            # Same URL intentionally models two fixture repos; real verifier
            # consumes exact host then engine SHA, never claims real pushed refs.
            target = host if not self.calls else engine
            self.calls.append((canonical_url, exact_ref))
            return f"{git(target, 'rev-parse', 'HEAD')}\t{exact_ref}\n".encode()

    remote = Remote()
    verifier = GitLsRemotePushedCommitVerifier(
        remote,
        auth,
        clock=clock.now,
        audience_ref="hf-prepare",
        issuer_ref="fixture-source",
        key_ref="source-key",
        challenge_factory=lambda: "fixture-challenge",
        evidence_ref_factory=lambda: "fixture-evidence",
    )
    return config, verifier, auth, clock, remote


def test_real_publication_source_material_and_consumer_preview(configured):
    config, verifier, auth, clock, remote = configured
    result = helper.prepare(
        config, pushed_sources=verifier, authenticator=auth, clock=clock
    )
    assert len(remote.calls) == 2
    output = Path(config["output_root"])
    consumer = json.loads((output / "consumer.json").read_bytes())
    assert helper.hf_host.preview(consumer) == result["preview"]
    material = CoordinatorResolvedMaterial.parse(
        (output / "material.json").read_bytes(), default_recipe_registry()
    )
    context = (
        material.to_dict()["execution_context"]
        if hasattr(material, "to_dict")
        else json.loads(material.canonical_bytes)["execution_context"]
    )
    source = json.loads(material.canonical_bytes)["execution_source"]
    assert sha(canonical(context)) == source["deployment_member_sha256"]
    assert canonical(context) == (output / "hf-runtime-declaration.json").read_bytes()
    assert result["public_preflight"] == "deferred_until_host_composition"
    assert not Path(consumer["state_root"]).exists()
    with pytest.raises(FileExistsError):
        output.mkdir()
    with pytest.raises(ValueError, match="used"):
        helper.prepare(config, pushed_sources=verifier, authenticator=auth, clock=clock)


@pytest.mark.parametrize(
    "field",
    ["image", "inventory_sha256", "roots", "interpreter", "secret_requirements_digest"],
)
def test_runtime_field_tamper_refuses_without_ready_output(configured, field):
    config, verifier, auth, clock, remote = configured
    p = Path(config["runtime_declaration"])
    d = json.loads(p.read_bytes())
    if field == "roots":
        d[field]["engine"] = "/workspace/engine"
    elif field == "interpreter":
        d[field]["executable_digest"] = "0" * 64
    else:
        d[field] = "0" * 64
    p.write_bytes(canonical(d))
    with pytest.raises(ValueError):
        helper.prepare(config, pushed_sources=verifier, authenticator=auth, clock=clock)
    assert remote.calls == [] and not Path(config["output_root"]).exists()


def test_input_tamper_and_real_gitlink_mismatch(configured):
    config, verifier, auth, clock, remote = configured
    raw = Path(config["prepared_publication"]) / "dataset.jsonl"
    original = raw.read_bytes()
    raw.chmod(0o600)
    raw.write_bytes(original + b" ")
    raw.chmod(0o400)
    with pytest.raises(ValueError):
        helper.prepare(config, pushed_sources=verifier, authenticator=auth, clock=clock)
    assert not remote.calls
    raw.chmod(0o600)
    raw.write_bytes(original)
    raw.chmod(0o400)
    engine = Path(config["project"]["engine_root"])
    (engine / "changed.txt").write_text("actual new engine commit")
    git(engine, "add", ".")
    git(engine, "commit", "-qm", "Changed gitlink target")
    with pytest.raises(SourceLockError, match="committed submodule identity"):
        helper.prepare(config, pushed_sources=verifier, authenticator=auth, clock=clock)
    assert not remote.calls and not Path(config["output_root"]).exists()


def test_wrong_signature_refuses_and_no_operational_output(configured):
    config, verifier, auth, clock, _ = configured
    original = verifier.verify
    verifier.verify = lambda lock: replace(original(lock), tag_base64="YmFk")
    with pytest.raises(ValueError, match="source_evidence_invalid"):
        helper.prepare(config, pushed_sources=verifier, authenticator=auth, clock=clock)
    assert not Path(config["output_root"]).exists()


def test_actual_preview_refusal_does_not_advertise_consumer(configured):
    config, verifier, auth, clock, _ = configured
    config["consumer"]["hf_jobs"]["reader_authority_ref"] = ""
    with pytest.raises(ValueError, match="hf_configuration_invalid"):
        helper.prepare(config, pushed_sources=verifier, authenticator=auth, clock=clock)
    root = Path(config["output_root"])
    assert (root / "material.json").is_file()  # retain failed local capture
    assert (
        not (root / "consumer.json").exists() and not (root / "preview.json").exists()
    )


@pytest.mark.parametrize("token", ["host_token", "job_token"])
def test_cli_provider_token_cannot_be_source_signing_key(
    configured, token, monkeypatch, capsys
):
    config, _, _, _, _ = configured
    config["source_authority"]["secret"] = config["consumer"]["hf_jobs"]["selection"][
        token
    ]
    p = Path(config["output_root"]).parent / "config.json"
    p.write_bytes(canonical(config))
    calls = []
    monkeypatch.setattr(
        helper.launch.NamedSecrets, "resolve", lambda *a: calls.append(a)
    )
    assert helper.main(["prepare", str(p)]) == 2
    assert not calls
    output = capsys.readouterr()
    assert (
        not output.err
        and json.loads(output.out)["diagnostic_code"]
        == "hf_material_preparation_failed"
    )
    assert not Path(config["output_root"]).exists()


def test_cli_suppresses_secret_bearing_failure(configured, monkeypatch, capsys):
    config, _, _, _, _ = configured
    p = Path(config["output_root"]).parent / "config.json"
    p.write_bytes(canonical(config))
    secret = "synthetic-secret-that-must-never-appear"
    monkeypatch.setattr(helper.launch.NamedSecrets, "resolve", lambda *a: secret)
    monkeypatch.setattr(
        helper.GitCliLocalSourceInspector,
        "inspect",
        lambda *a, **k: (_ for _ in ()).throw(ValueError(secret)),
    )
    assert helper.main(["prepare", str(p)]) == 2
    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "diagnostic_code": "hf_material_preparation_failed"
    }
    assert secret not in output.out + output.err and not output.err
    assert not Path(config["output_root"]).exists()


def test_declare_is_credential_free_and_create_only(configured, monkeypatch, capsys):
    config, _, _, _, _ = configured
    declaration = helper.runtime_declaration(config)
    assert declaration.to_dict() == json.loads(
        Path(config["runtime_declaration"]).read_bytes()
    )
    Path(config["runtime_declaration"]).unlink()
    path = Path(config["output_root"]).parent / "declare.json"
    path.write_bytes(canonical(config))
    monkeypatch.setattr(
        helper.launch.NamedSecrets,
        "resolve",
        lambda *a: pytest.fail("secret resolution"),
    )
    assert helper.main(["declare", str(path)]) == 0
    assert (
        Path(config["runtime_declaration"]).read_bytes()
        == declaration.canonical_json.encode()
    )
    assert json.loads(capsys.readouterr().out)["operational_preflight"] is False
    assert helper.main(["declare", str(path)]) == 2
    assert not Path(config["output_root"]).exists()


def test_declared_environment_is_configured_and_bound(configured):
    config, verifier, auth, clock, remote = configured
    config["runtime_environment"]["OMP_NUM_THREADS"] = "1"
    declaration = helper.runtime_declaration(config)
    Path(config["runtime_declaration"]).write_bytes(declaration.canonical_json.encode())
    result = helper.prepare(
        config, pushed_sources=verifier, authenticator=auth, clock=clock
    )
    material = CoordinatorResolvedMaterial.parse(
        Path(result["consumer_config"]).parent.joinpath("material.json").read_bytes(),
        default_recipe_registry(),
    )
    assert (
        json.loads(material.execution_context_bytes)["environment"]["OMP_NUM_THREADS"]
        == "1"
    )
