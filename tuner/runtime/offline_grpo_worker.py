"""Separate fixed GRPO source inventory; the SFT inventory is never extended."""
from __future__ import annotations

import hashlib
import importlib.resources
from pathlib import Path

from .offline_sft_worker import (
    OfflineSFTWorkerError, OfflineSFTWorkerManifestV1, _canonical_json,
    _parse_manifest, _strict_document, OWNED_MODULE_PREFIXES, closure_digest,
)
from .packaged_worker_closure import stable_read

GRPO_CLOSURE_SCHEMA = "synaptic-offline-grpo-worker-closure/v1"
GRPO_CLOSURE_REF = "synaptic/offline-grpo-worker/v1"
GRPO_ENTRYPOINT = "tuner/runtime/packaged_grpo_child.py"
GRPO_TRAINER_ENTRYPOINT = "Trainers/grpo/runtime_v1.py"
GRPO_BOOTSTRAP = "tuner/runtime/offline_grpo_worker.py"
GRPO_MANIFEST_NAME = "offline-grpo-worker-v1.json"
GRPO_PACKAGED_SOURCE = "package:tuner.runtime/manifests/" + GRPO_MANIFEST_NAME
# Membership is reviewed explicitly, not discovered from ambient imports.
GRPO_OFFLINE_MEMBERS = (
    'SynthChat/__init__.py',
    'SynthChat/config/__init__.py',
    'SynthChat/config/config_loader.py',
    'SynthChat/config/format_resolver.py',
    'SynthChat/utils/__init__.py',
    'SynthChat/utils/prompt_renderer.py',
    'SynthChat/utils/yaml_loader.py',
    'Trainers/grpo/configs/config.yaml',
    'Trainers/grpo/runtime_v1.py',
    'Trainers/grpo/src/data_loader.py',
    'Trainers/grpo/src/model_loader.py',
    'Trainers/grpo/src/rewards.py',
    'Trainers/sft/runtime_v1.py',
    'shared/judge/__init__.py',
    'shared/judge/interaction_logger.py',
    'shared/judge/judge_service.py',
    'shared/judge/models.py',
    'shared/judge/rubric_loader.py',
    'shared/judge/schema_builder.py',
    'shared/llm/__init__.py',
    'shared/llm/base.py',
    'shared/llm/config.py',
    'shared/llm/exceptions.py',
    'shared/llm/factory.py',
    'shared/llm/providers/__init__.py',
    'shared/llm/providers/lmstudio.py',
    'shared/llm/providers/ollama.py',
    'shared/llm/providers/openai_responses.py',
    'shared/llm/providers/openrouter.py',
    'shared/llm/providers/unsloth.py',
    'shared/llm/usage.py',
    'shared/ml/__init__.py',
    'shared/ml/retrieval_metrics.py',
    'shared/model_loading/__init__.py',
    'shared/model_loading/merge.py',
    'shared/training_utils.py',
    'shared/utilities/__init__.py',
    'shared/utilities/env.py',
    'shared/utilities/paths.py',
    'shared/utilities/yaml_loader.py',
    'shared/validation/__init__.py',
    'shared/validation/fitness.py',
    'shared/validation/parsing/__init__.py',
    'shared/validation/parsing/configured_formats.py',
    'shared/validation/parsing/enums.py',
    'shared/validation/parsing/response_parser.py',
    'shared/validation/parsing/tool_call_parser.py',
    'shared/validation/parsing/utilities.py',
    'shared/validation/rollout_filters.py',
    'shared/validation/rubric/__init__.py',
    'shared/validation/rubric/rubric_cache.py',
    'shared/validation/rubric/rubric_loader.py',
    'shared/validation/rubric/rubric_repository.py',
    'shared/validation/validators/__init__.py',
    'shared/validation/validators/base.py',
    'shared/validation/validators/content/__init__.py',
    'shared/validation/validators/content/code_validator.py',
    'shared/validation/validators/content/json_validator.py',
    'shared/validation/validators/content/regex_validator.py',
    'shared/validation/validators/content/xml_validator.py',
    'shared/validation/validators/content/yaml_validator.py',
    'shared/validation/validators/cross_scope_validator.py',
    'shared/validation/validators/structure_validator.py',
    'shared/verifiers/__init__.py',
    'shared/verifiers/builtins/__init__.py',
    'shared/verifiers/builtins/args_match.py',
    'shared/verifiers/builtins/assertion_verifier.py',
    'shared/verifiers/builtins/audio_verifier.py',
    'shared/verifiers/builtins/llm_judge.py',
    'shared/verifiers/builtins/retrieval_verifier.py',
    'shared/verifiers/builtins/structure.py',
    'shared/verifiers/builtins/substring.py',
    'shared/verifiers/builtins/tool_sequence.py',
    'shared/verifiers/composite.py',
    'shared/verifiers/contract.py',
    'shared/verifiers/extraction.py',
    'shared/verifiers/registry.py',
    'synaptic_tuner/__init__.py',
    'synaptic_tuner/_version.py',
    'synaptic_tuner/api/__init__.py',
    'synaptic_tuner/api/v1/__init__.py',
    'synaptic_tuner/api/v1/_contract.py',
    'synaptic_tuner/api/v1/_timestamps.py',
    'synaptic_tuner/api/v1/observations.py',
    'synaptic_tuner/api/v1/planning.py',
    'synaptic_tuner/api/v1/providers.py',
    'synaptic_tuner/api/v1/results.py',
    'synaptic_tuner/api/v1/training_facade.py',
    'synaptic_tuner/api/v1/training_input.py',
    'synaptic_tuner/api/v1/training_sources.py',
    'synaptic_tuner/api/v1/usage.py',
    'tuner/__init__.py',
    'tuner/cloud/__init__.py',
    'tuner/cloud/runtime_layout.py',
    'tuner/execution/__init__.py',
    'tuner/execution/contracts.py',
    'tuner/execution/evidence.py',
    'tuner/execution/lifecycle.py',
    'tuner/execution/registry.py',
    'tuner/execution/service.py',
    'tuner/project/__init__.py',
    'tuner/project/config_layers.py',
    'tuner/project/context.py',
    'tuner/project/errors.py',
    'tuner/project/execution_source.py',
    'tuner/project/git_verification.py',
    'tuner/project/manifest.py',
    'tuner/project/path_refs.py',
    'tuner/project/secrets.py',
    'tuner/project/source_bundle.py',
    'tuner/runtime/__init__.py',
    'tuner/runtime/artifacts.py',
    'tuner/runtime/dispatch.py',
    'tuner/runtime/manifests/grpo-constructor-library-sources-v1.json',
    'tuner/runtime/offline_grpo_worker.py',
    'tuner/runtime/offline_sft_worker.py',
    'tuner/runtime/packaged_grpo_child.py',
    'tuner/runtime/packaged_grpo_constructor_child.py',
    'tuner/runtime/packaged_grpo_execution.py',
    'tuner/runtime/packaged_grpo_verification.py',
    'tuner/runtime/packaged_grpo_worker_closure.py',
    'tuner/runtime/packaged_sft_child.py',
    'tuner/runtime/packaged_sft_execution.py',
    'tuner/runtime/packaged_training_worker.py',
    'tuner/runtime/packaged_worker_closure.py',
    'tuner/runtime/releases.py',
    'tuner/runtime/verification.py',
    'tuner/training/__init__.py',
    'tuner/training/contracts.py',
    'tuner/training/grpo_compilation.py',
    'tuner/training/packaged_compilation.py',
    'tuner/training/packaged_grpo_compilation.py',
    'tuner/training/recipes.py',
)


def parse_offline_grpo_worker_manifest(payload, *, source_ref=GRPO_PACKAGED_SOURCE,
                                       manifest_path=None, expected_digest=None):
    if type(payload) is not bytes or not 0 < len(payload) <= 1024 * 1024:
        raise OfflineSFTWorkerError("GRPO worker manifest is bounded")
    document = _strict_document(payload)
    if _canonical_json(document) + b"\n" != payload:
        raise OfflineSFTWorkerError("GRPO worker manifest is canonical")
    path = Path(GRPO_PACKAGED_SOURCE) if manifest_path is None else manifest_path
    closure = _parse_manifest(document, manifest_path=path,
        expected_digest=document.get("closure_digest") if expected_digest is None else expected_digest,
        method="grpo",
        _method_identity=(GRPO_CLOSURE_SCHEMA, GRPO_CLOSURE_REF, GRPO_ENTRYPOINT,
                          GRPO_TRAINER_ENTRYPOINT, GRPO_BOOTSTRAP),
        _method_members=GRPO_OFFLINE_MEMBERS)
    if any(member.git_mode != "100644" for member in closure.members):
        raise OfflineSFTWorkerError("GRPO source modes are unsupported")
    return OfflineSFTWorkerManifestV1(source_ref=source_ref, canonical_bytes=payload,
        byte_count=len(payload), sha256=hashlib.sha256(payload).hexdigest(), closure=closure)


def load_packaged_offline_grpo_worker_manifest():
    resource = importlib.resources.files("tuner.runtime").joinpath("manifests", GRPO_MANIFEST_NAME)
    if not isinstance(resource, Path):
        raise OfflineSFTWorkerError("GRPO worker resource is unavailable")
    return parse_offline_grpo_worker_manifest(stable_read(resource, 1024 * 1024))


def verify_installed_grpo_members():
    manifest = load_packaged_offline_grpo_worker_manifest()
    root = Path(__file__).absolute().parents[2]
    for member in manifest.closure.members:
        raw = stable_read(root / member.path, member.size_bytes)
        if len(raw) != member.size_bytes or hashlib.sha256(raw).hexdigest() != member.sha256:
            raise OfflineSFTWorkerError("GRPO worker source differs")
    return manifest


def captured_grpo_manifest_document(root):
    """Hash only reviewed fixed members; no discovery, SDK or credential access."""
    root = Path(root)
    members = []
    for name in GRPO_OFFLINE_MEMBERS:
        raw = stable_read(root / name, 64 * 1024 * 1024)
        members.append({"path": name, "git_mode": "100644", "size_bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()})
    document = {"schema_version": GRPO_CLOSURE_SCHEMA, "closure_ref": GRPO_CLOSURE_REF,
        "entrypoint": GRPO_ENTRYPOINT, "trainer_entrypoint": GRPO_TRAINER_ENTRYPOINT,
        "owned_module_prefixes": list(OWNED_MODULE_PREFIXES), "optional_features": [],
        "member_count": len(members), "payload_bytes": sum(member["size_bytes"] for member in members),
        "members": members}
    document["closure_digest"] = closure_digest(document)
    parse_offline_grpo_worker_manifest(_canonical_json(document) + b"\n")
    return document
