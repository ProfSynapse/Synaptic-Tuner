"""Host-owned HF material preparation, without Jobs or model calls.

Preparation reads exact public Git refs and authenticates local source evidence.
The resulting consumer preview is credential-free; it is not operational preflight.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import secrets
import sys

# Only fixed, checkout-owned siblings; engine imports must come from installation.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_host
import launch

from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.secrets import SecretRef
from tuner.dataset_prep.publication import snapshot_prepared_dataset_v1
from tuner.execution.evidence import (
    SOURCE_EVIDENCE_POLICY,
    SOURCE_EVIDENCE_PURPOSE,
    validate_evidence_window,
)
from tuner.project.context import ProjectContext
from tuner.project.execution_source import (
    ExecutionSourceV1,
    AuthenticatedSourceEvidenceV1,
)
from tuner.project.git_verification import (
    GitCliLocalSourceInspector,
    GitLsRemotePushedCommitVerifier,
)
from tuner.project.manifest import load_project_manifest
from tuner.runtime.packaged_worker_closure import stable_read
from tuner.runtime_profiles import load_runtime_profile
from tuner.training import TrainingService, default_recipe_registry
from tuner.training.contracts import (
    ArtifactPolicy,
    CanonicalDocument,
    ResolvedTrainingComponents,
    RuntimeSpec,
    ResourceSpec,
)
from tuner.training.coordinator_material import (
    derive_coordinator_material,
    CoordinatorResolvedMaterial,
)
from SynthChat.utils.yaml_loader import parse_workflow_yaml


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _exact(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError("hf_preparation_fields_invalid")
    return value


def _document(path, maximum=1 << 20):
    return CanonicalDocument(stable_read(Path(path), maximum).decode("utf-8"))


def _same_checkout(left, right):
    return (
        left.project_source == right.project_source
        and left.engine_source == right.engine_source
    )


def _source_secret(config):
    authority = _exact(
        config["source_authority"], {"secret", "issuer_ref", "key_ref", "audience_ref"}
    )
    source = SecretRef.from_dict(authority["secret"])
    selection = config["consumer"]["hf_jobs"]["selection"]
    if any(
        source == SecretRef.from_dict(selection[key])
        for key in ("host_token", "job_token")
    ):
        raise ValueError("hf_source_key_must_be_distinct")
    return source


def _environment(config):
    environment = config["runtime_environment"]
    if (
        type(environment) is not dict
        or not environment
        or any(
            type(key) is not str or type(value) is not str
            for key, value in environment.items()
        )
    ):
        raise ValueError("hf_runtime_environment_invalid")
    return dict(environment)


def runtime_declaration(config):
    """Pure selected-profile declaration; no source proof or operational readiness."""
    c = config["consumer"]["hf_jobs"]
    p = c["runtime_profile"]
    profile = load_runtime_profile(p["name"], Path(p["root"]))
    if p["sha256"] != profile.profile_sha256.removeprefix(
        "sha256:"
    ) or profile.methods != ("sft",):
        raise ValueError("hf_profile_changed_or_unsupported")
    image_profile = profile.modal_build_profile_path(
        Path(config["project"]["engine_root"]) / "Trainers/image_profiles"
    )
    if Path(config["image_profile"]).resolve(strict=True) != image_profile.resolve(
        strict=True
    ):
        raise ValueError("hf_interpreter_profile_path_mismatch")
    pins = parse_workflow_yaml(stable_read(image_profile, 1 << 20))
    if pins.get("base_image", "").removeprefix("docker.io/") != profile.image:
        raise ValueError("hf_interpreter_profile_mismatch")
    py = pins["packaged_runtime"]["python"]
    interpreter = {
        key: py[key]
        for key in ("implementation", "version", "executable", "executable_digest")
    }
    if interpreter["version"] != profile.runtime_facts["python_version"]:
        raise ValueError("hf_interpreter_mismatch")
    roots = {
        name: "/workspace/private/"
        + ("" if name in {"engine", "project"} else "run/")
        + name
        for name in (
            "engine",
            "project",
            "artifacts",
            "state",
            "tracking",
            "cache",
            "tmp",
        )
    }
    d = {
        "schema_version": "hf-runtime-declaration/v1",
        "runtime_profile": {"name": profile.name, "sha256": profile.profile_sha256},
        "inventory_sha256": _sha(stable_read(profile.inventory_path, 1 << 20)),
        "image": profile.image,
        "interpreter": interpreter,
        "roots": roots,
        "writable_capability_root": "/workspace/private/run",
        "environment": _environment(config),
        "secret_requirements_digest": _sha(
            CanonicalDocument.from_mapping(
                {key: c["selection"][key] for key in ("host_token", "job_token")}
            ).canonical_json.encode()
        ),
    }
    d["provider_runtime_requirements_digest"] = _sha(
        CanonicalDocument.from_mapping(
            {
                key: d[key]
                for key in (
                    "runtime_profile",
                    "inventory_sha256",
                    "image",
                    "interpreter",
                )
            }
        ).canonical_json.encode()
    )
    return CanonicalDocument.from_mapping(d)


def _runtime(config, profile):
    declared = _document(config["runtime_declaration"], 64 << 10)
    d = _exact(
        declared.to_dict(),
        {
            "schema_version",
            "runtime_profile",
            "inventory_sha256",
            "image",
            "interpreter",
            "roots",
            "writable_capability_root",
            "environment",
            "secret_requirements_digest",
            "provider_runtime_requirements_digest",
        },
    )
    if d["schema_version"] != "hf-runtime-declaration/v1":
        raise ValueError("hf_runtime_declaration_invalid")
    if d["runtime_profile"] != {"name": profile.name, "sha256": profile.profile_sha256}:
        raise ValueError("hf_runtime_profile_mismatch")
    if d["image"] != profile.image or d["inventory_sha256"] != _sha(
        stable_read(profile.inventory_path, 1 << 20)
    ):
        raise ValueError("hf_runtime_inventory_mismatch")
    # Reuse the selected configuration's recorded Python pins, not a deployment.
    image_profile = profile.modal_build_profile_path(
        Path(config["project"]["engine_root"]) / "Trainers/image_profiles"
    )
    if Path(config["image_profile"]).resolve(strict=True) != image_profile.resolve(
        strict=True
    ):
        raise ValueError("hf_interpreter_profile_path_mismatch")
    pins = parse_workflow_yaml(stable_read(image_profile, 1 << 20))
    expected_name = profile.packaged_build_profile
    if (
        not expected_name
        or pins.get("name", "").replace("-", "_") != expected_name
        or pins.get("base_image", "").removeprefix("docker.io/") != profile.image
    ):
        raise ValueError("hf_interpreter_profile_mismatch")
    python = pins["packaged_runtime"]["python"]
    interpreter = {
        key: python[key]
        for key in ("implementation", "version", "executable", "executable_digest")
    }
    if (
        d["interpreter"] != interpreter
        or interpreter["version"] != profile.runtime_facts["python_version"]
    ):
        raise ValueError("hf_interpreter_mismatch")
    roots = {
        name: "/workspace/private/"
        + ("" if name in {"engine", "project"} else "run/")
        + name
        for name in (
            "engine",
            "project",
            "artifacts",
            "state",
            "tracking",
            "cache",
            "tmp",
        )
    }
    if d["roots"] != roots or d["writable_capability_root"] != "/workspace/private/run":
        raise ValueError("hf_runtime_roots_invalid")
    env = d["environment"]
    if env != _environment(config):
        raise ValueError("hf_runtime_environment_invalid")
    selection = config["consumer"]["hf_jobs"]["selection"]
    secret_bytes = CanonicalDocument.from_mapping(
        {key: selection[key] for key in ("host_token", "job_token")}
    ).canonical_json.encode()
    requirement_bytes = CanonicalDocument.from_mapping(
        {
            key: d[key]
            for key in ("runtime_profile", "inventory_sha256", "image", "interpreter")
        }
    ).canonical_json.encode()
    if d["secret_requirements_digest"] != _sha(secret_bytes) or d[
        "provider_runtime_requirements_digest"
    ] != _sha(requirement_bytes):
        raise ValueError("hf_runtime_requirements_mismatch")
    return declared, d


def validate_runtime_binding(declaration, source, runtime):
    """A retained host declaration, never a staged/provider deployment claim."""
    d = declaration.to_dict()
    py = d["interpreter"]
    if (
        source.deployment_member_sha256 != _sha(declaration.canonical_json.encode())
        or dict(source.roots) != d["roots"]
        or source.writable_capability_root != d["writable_capability_root"]
        or dict(source.environment) != d["environment"]
        or source.python_implementation != py["implementation"]
        or source.python_version != py["version"]
        or source.python_executable != py["executable"]
        or source.python_executable_digest != py["executable_digest"]
        or source.secret_requirements_digest != d["secret_requirements_digest"]
        or source.provider_runtime_requirements_digest
        != d["provider_runtime_requirements_digest"]
        or runtime.image != d["image"]
        or runtime.python_version != py["version"]
        or runtime.dependency_lock_digest != d["inventory_sha256"]
    ):
        raise ValueError("hf_runtime_source_mismatch")


def authenticated_source(
    config,
    context,
    normalized,
    identity,
    declaration,
    runtime_fields,
    manifest,
    *,
    pushed_sources,
    authenticator,
    clock,
):
    """Existing source admission shared by checkout-owned provider consumers."""
    inspected = GitCliLocalSourceInspector().inspect(context=context)
    if inspected.project_source.dirty or inspected.engine_source.dirty:
        raise ValueError("hf_sources_dirty")
    lock = replace(
        inspected,
        run_id=config["run_id"],
        created_at=clock.now(),
        project=manifest.to_dict(),
        configuration={"sha256": _sha(normalized.canonical_json.encode())},
        inputs=(identity.to_dict(),),
        runtime={"declaration_sha256": _sha(declaration.canonical_json.encode())},
    )
    original_binding = lock.binding
    evidence = pushed_sources.verify(lock)
    authority = _exact(
        config["source_authority"], {"secret", "issuer_ref", "key_ref", "audience_ref"}
    )
    if (
        type(evidence) is not AuthenticatedSourceEvidenceV1
        or lock.binding != original_binding
        or not evidence.binds(lock)
        or evidence.issuer_ref != authority["issuer_ref"]
        or evidence.key_ref != authority["key_ref"]
        or evidence.audience_ref != authority["audience_ref"]
    ):
        raise ValueError("hf_source_evidence_mismatch")
    payload = evidence.authenticated_payload
    if (
        _sha(payload) != evidence.attestation_digest
        or authenticator.verify(
            SOURCE_EVIDENCE_PURPOSE,
            payload,
            base64.b64decode(evidence.tag_base64),
            evidence.key_ref,
        )
        is not True
    ):
        raise ValueError("hf_source_evidence_invalid")
    validate_evidence_window(
        verified_at=evidence.verified_at,
        expires_at=evidence.expires_at,
        now=clock.now(),
        policy=SOURCE_EVIDENCE_POLICY,
    )
    py = runtime_fields["interpreter"]
    source = ExecutionSourceV1(
        run_id=lock.run_id,
        created_at=lock.created_at,
        project_source=lock.project_source,
        engine_source=lock.engine_source,
        engine_submodule_path=lock.engine_source.submodule_path,
        source_evidence=evidence,
        deployment_member_sha256=_sha(declaration.canonical_json.encode()),
        roots=runtime_fields["roots"],
        writable_capability_root=runtime_fields["writable_capability_root"],
        python_implementation=py["implementation"],
        python_version=py["version"],
        python_executable=py["executable"],
        python_executable_digest=py["executable_digest"],
        environment=runtime_fields["environment"],
        secret_requirements_digest=runtime_fields["secret_requirements_digest"],
        provider_runtime_requirements_digest=runtime_fields[
            "provider_runtime_requirements_digest"
        ],
    )
    return source, lock, inspected


def prepare(config, *, pushed_sources, authenticator, clock):
    """Finalize actual caller material through existing source/compiler ports.

    No host/provider credential is resolved here; the caller supplies the source
    verifier/authenticator. The CLI below explicitly selects a local signing key.
    """
    _exact(
        config,
        {
            "project",
            "run_id",
            "request_id",
            "output_root",
            "prepared_publication",
            "resolved_config",
            "runtime_declaration",
            "runtime_environment",
            "image_profile",
            "source_authority",
            "consumer",
        },
    )
    _source_secret(config)
    root = Path(config["output_root"])
    if not root.is_absolute() or root.exists() or root.is_symlink():
        raise ValueError("hf_output_root_invalid_or_used")
    project = _exact(config["project"], {"root", "engine_root", "manifest"})
    context = ProjectContext.host(
        project_root=Path(project["root"]),
        engine_root=Path(project["engine_root"]),
        manifest_path=Path(project["manifest"]),
        invocation_cwd=Path(project["root"]),
    )
    manifest = load_project_manifest(context.manifest_path)
    consumer = json.loads(json.dumps(config["consumer"], allow_nan=False))
    c = consumer["hf_jobs"]
    profile = load_runtime_profile(
        c["runtime_profile"]["name"], Path(c["runtime_profile"]["root"])
    )
    if c["runtime_profile"]["sha256"] != profile.profile_sha256.removeprefix("sha256:"):
        raise ValueError("hf_profile_changed")
    resources = ResourceSpec(**c["resources"])
    s = c["selection"]
    hf_host.HFJobsSelection(
        hf_host.ProviderRef.from_dict(c["provider"]),
        hf_host.ExecutionScopeV1(**s["scope"]),
        profile,
        stable_read(profile.inventory_path, 1 << 20),
        s["bucket_id"],
        s["source_prefix"],
        s["run_prefix"],
        s["flavor"],
        s["timeout_seconds"],
        SecretRef.from_dict(s["host_token"]),
        SecretRef.from_dict(s["job_token"]),
        s["quote_digest"],
        resources,
    )
    declaration, runtime_fields = _runtime(config, profile)
    verified, raw = snapshot_prepared_dataset_v1(Path(config["prepared_publication"]))
    semantic = verified.semantic_identity
    if len(raw) > 2 << 20:
        raise ValueError("hf_prepared_input_too_large")
    identity = PreparedTrainingInputIdentity(
        "prepared://sha256/" + semantic.dataset_digest,
        semantic.dataset_digest,
        _sha(raw),
        len(raw),
        "syntunia-sft-row/v1",
    )
    normalized = _document(config["resolved_config"])
    normalized_fields = normalized.to_dict()
    if normalized_fields.get("dataset") != identity.to_dict():
        raise ValueError("hf_prepared_configuration_mismatch")
    model = normalized_fields["model"]
    profile.resolve(model=model["ref"], model_revision=model["revision"], method="sft")
    if model["tokenizer_revision"] != model["revision"]:
        raise ValueError("hf_tokenizer_revision_unsupported")
    source, lock, inspected = authenticated_source(
        config,
        context,
        normalized,
        identity,
        declaration,
        runtime_fields,
        manifest,
        pushed_sources=pushed_sources,
        authenticator=authenticator,
        clock=clock,
    )
    py = runtime_fields["interpreter"]
    runtime = RuntimeSpec(
        profile.image, profile.inventory_sha256.removeprefix("sha256:"), py["version"]
    )
    validate_runtime_binding(declaration, source, runtime)
    # resources are preparation-only; the output adapter derives them from material.
    del c["resources"]
    artifacts = ArtifactPolicy(
        (
            "final_model",
            "tokenizer",
            "training_lineage",
            "training_metrics",
            "workload_record",
        ),
        False,
    )
    components = ResolvedTrainingComponents(
        source, declaration, normalized, runtime, resources, artifacts
    )

    class Resolver:
        def resolve(self, request, *, context):
            return components

    recipes = default_recipe_registry()
    service = TrainingService(context=context, resolver=Resolver(), recipes=recipes)
    request = service.load(normalized)
    resolved = service.resolve(request)
    material = derive_coordinator_material(
        resolved,
        recipes,
        request_id=config["request_id"],
        project_ref=manifest.project_id,
        run_id=source.run_id,
    )
    # Exercise the same canonical reconstruction consumed by the HF adapter.
    retained = CoordinatorResolvedMaterial.parse(material.canonical_bytes, recipes)
    retained_document = json.loads(retained.canonical_bytes)
    retained_context = CanonicalDocument.from_mapping(
        retained_document["execution_context"]
    )
    if retained_context != declaration:
        raise ValueError("hf_runtime_context_mismatch")
    validate_runtime_binding(
        retained_context,
        ExecutionSourceV1.from_dict(retained_document["execution_source"]),
        RuntimeSpec(**retained_document["runtime"]),
    )
    # Recheck the actual local sources after any caller verifier work.
    if not _same_checkout(
        inspected, GitCliLocalSourceInspector().inspect(context=context)
    ):
        raise ValueError("hf_sources_changed")
    root.mkdir(mode=0o700)
    launch._private(root)
    for name, data in (
        ("request.json", material.request_bytes),
        ("material.json", material.canonical_bytes),
        ("prepared-input.jsonl", raw),
        ("hf-runtime-declaration.json", declaration.canonical_json.encode()),
        (
            "source-lock.json",
            json.dumps(lock.to_dict(), sort_keys=True, separators=(",", ":")).encode(),
        ),
    ):
        launch._write_bytes(root / name, data)

    def spec(name):
        p = root / name
        data = stable_read(p, 2 << 20)
        return {"path": str(p), "sha256": _sha(data), "size_bytes": len(data)}

    consumer["request_file"] = spec("request.json")
    c["material_file"] = spec("material.json")
    c["prepared_file"] = spec("prepared-input.jsonl")
    launch._write(root / "consumer-candidate.json", consumer)
    consumer = launch.load_config(root / "consumer-candidate.json")
    preview = hf_host.preview(consumer)
    launch._write(root / "preview.json", preview)
    launch._write(root / "consumer.json", consumer)
    return {
        "consumer_config": str(root / "consumer.json"),
        "preview": preview,
        "public_preflight": "deferred_until_host_composition",
    }


def _example_module(name, filename):
    path = Path(__file__).resolve().parent.parent / "modal_chat" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "prepare"))
    parser.add_argument("configuration", type=Path)
    args = parser.parse_args(argv)
    try:
        config = _document(args.configuration).to_dict()
        if args.action == "declare":
            declaration = runtime_declaration(config)
            path = Path(config["runtime_declaration"])
            launch._private(path.parent)
            launch._write_bytes(path, declaration.canonical_json.encode())
            print(
                json.dumps(
                    {
                        "runtime_declaration": str(path),
                        "sha256": _sha(declaration.canonical_json.encode()),
                        "operational_preflight": False,
                    }
                )
            )
            return 0
        authority = config["source_authority"]
        key = launch.NamedSecrets().resolve(_source_secret(config))
        if type(key) is not str or len(key.encode()) < 32:
            raise ValueError("hf_source_key_unavailable")
        hmac = _example_module("hf_source_auth", "authority.py").HMACAuthenticator
        auth = hmac(
            {authority["key_ref"]: key.encode()},
            allowed_purposes=frozenset({SOURCE_EVIDENCE_PURPOSE}),
        )
        context = ProjectContext.host(
            project_root=Path(config["project"]["root"]),
            engine_root=Path(config["project"]["engine_root"]),
        )
        lock = GitCliLocalSourceInspector().inspect(context=context)
        remote = _example_module("hf_source_remote", "remote.py").ModalChatGitRemote
        refs = frozenset(
            (s.location.canonical_url, "refs/heads/" + s.branch)
            for s in (lock.project_source, lock.engine_source)
        )
        clock = _example_module("hf_source_clock", "authority.py").UTCClock()
        verifier = GitLsRemotePushedCommitVerifier(
            remote(refs),
            auth,
            clock=clock.now,
            audience_ref=authority["audience_ref"],
            issuer_ref=authority["issuer_ref"],
            key_ref=authority["key_ref"],
            challenge_factory=lambda: secrets.token_hex(24),
            evidence_ref_factory=lambda: secrets.token_hex(24),
        )
        result = prepare(
            config, pushed_sources=verifier, authenticator=auth, clock=clock
        )
    except Exception:
        print(json.dumps({"diagnostic_code": "hf_material_preparation_failed"}))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
