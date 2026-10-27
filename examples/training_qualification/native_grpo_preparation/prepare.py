"""GRPO preparation and offline tokenizer-only diagnosis; no execution authority.

Wrap the existing public preparation service. Keep normalization, publication
identity, row validation and retained-source verification in the engine.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import os
from pathlib import Path
import re
import sys

ENGINE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ENGINE_ROOT))

from synaptic_tuner.api.v1.training_input import TrainingInputV1, TrainingMethodV1
from synaptic_tuner.api.v1.training_sources import LocalTrainingInputPathV1, TrainingPreparationConfigV1
from tuner.runtime.native_files import NativeTree, file_identity
from tuner.runtime.native_mlx_material import canonical, strict_json
from tuner.training.grpo_preparation import (
    GRPONormalizerConfigV1, GRPOPreparedInputFormatVerifierV1,
    GRPOTrainingInputNormalizerV1, snapshot_grpo_publication,
)
from tuner.training.input_preparation import (
    PosixPrivatePreparedRootAuthorityV1, TrainingInputPreparationServiceV1,
)

SCHEMA = "synaptic-native-grpo-preparation-example/v1"
RESULT_SCHEMA = "synaptic-native-grpo-preparation-result/v1"
TOKENIZER_SCHEMA = "synaptic-native-grpo-tokenizer-profile-example/v1"
TOKENIZER_RESULT_SCHEMA = "synaptic-native-grpo-tokenizer-profile-result/v1"


def _path(value, base):
    if (type(value) is not str or not 1 <= len(value.encode("utf-8")) <= 1024
            or any(ord(c) < 32 for c in value) or ".." in Path(value).parts):
        raise ValueError("invalid example path")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _read(path, maximum):
    with NativeTree(path.parent) as tree:
        before = os.stat(path.name, dir_fd=tree.fd, follow_symlinks=False)
        if before.st_nlink != 1:
            raise ValueError("example input must be single-link")
        raw = tree.read(path.name, maximum=maximum)
        after = os.stat(path.name, dir_fd=tree.fd, follow_symlinks=False)
        if file_identity(before) != file_identity(after) or after.st_nlink != 1:
            raise ValueError("example input changed")
        return raw


def _mkdir(parent, name):
    parent.check()
    os.mkdir(name, mode=0o700, dir_fd=parent.fd)
    os.fsync(parent.fd)
    parent.check()


def run(action, configuration):
    """Return only a bounded summary; retain every owned failure stage."""
    if action == "profile-tokenizer":
        return profile_tokenizer(configuration)
    phase, owned = "configuration", None
    with ExitStack() as stack:
        try:
            if sys.platform != "linux" or action not in ("inspect", "prepare"):
                raise ValueError("Linux preparation only")
            path = Path(configuration).absolute()
            config = strict_json(_read(path, 128 * 1024), maximum=128 * 1024)
            if (type(config) is not dict or set(config) != {
                    "schema", "source", "output_root", "request_id", "project_ref", "training_input"}
                    or config["schema"] != SCHEMA):
                raise ValueError("example fields differ")
            source = config["source"]
            if (type(source) is not dict or set(source) != {"file", "sha256", "size_bytes", "prompt_column"}
                    or type(source["sha256"]) is not str or not re.fullmatch("[0-9a-f]{64}", source["sha256"])
                    or type(source["size_bytes"]) is not int or not 1 <= source["size_bytes"] <= 65536):
                raise ValueError("example source commitment differs")
            training = TrainingInputV1.from_dict(config["training_input"])
            if training.method is not TrainingMethodV1.GRPO or training.hyperparameters.prompt_column != source["prompt_column"]:
                raise ValueError("example GRPO prompt selection differs")
            normalizer_config = GRPONormalizerConfigV1("seed.jsonl", source["prompt_column"])
            preparation = TrainingPreparationConfigV1(config["request_id"], config["project_ref"],
                training.canonical_json(), normalizer_config)
            seed = _path(source["file"], path.parent)
            output = _path(config["output_root"], path.parent)
            phase = "source"
            raw = _read(seed, 65536)
            if len(raw) != source["size_bytes"] or hashlib.sha256(raw).hexdigest() != source["sha256"]:
                raise ValueError("example source bytes differ")
            phase = "output"
            parent = stack.enter_context(NativeTree(output.parent, private=True))
            if action == "inspect":
                try:
                    os.stat(output.name, dir_fd=parent.fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise ValueError("example output already exists")
                parent.check()
                return 0, {"schema": RESULT_SCHEMA, "status": "INPUT_BYTES_VERIFIED_ONLY",
                    "source_sha256": source["sha256"], "source_bytes": len(raw), "automatic_retry": False}
            # The new private root is the create-only claim. Existing roots,
            # including failed attempts, are never adopted, repaired or deleted.
            _mkdir(parent, output.name)
            owned = stack.enter_context(NativeTree(output, private=True))
            owned.write_exclusive("configuration.json", canonical(config))
            for name in ("source", "publications"):
                _mkdir(owned, name)
            source_root, publications = output / "source", output / "publications"
            with NativeTree(source_root, private=True) as staged:
                staged.write_exclusive(normalizer_config.member, raw)
            phase = "preparation"
            verifier = GRPOPreparedInputFormatVerifierV1()
            service = TrainingInputPreparationServiceV1(
                prepared_root=publications,
                normalizers={normalizer_config.normalizer_ref:
                    GRPOTrainingInputNormalizerV1(allowed_source_roots=(source_root,))},
                format_verifiers={verifier.format: verifier},
                root_authority=PosixPrivatePreparedRootAuthorityV1(),
            )
            result = service.prepare(LocalTrainingInputPathV1(source_root), preparation)
            phase = "verification"
            identity = result.prepared.identity
            publication = publications / ("dataset-" + identity.revision)
            manifest, published = snapshot_grpo_publication(publication)
            if (published != raw or manifest["semantic_digest"] != identity.revision
                    or manifest["content_digest"] != identity.content_digest
                    or manifest["size_bytes"] != identity.size_bytes):
                raise ValueError("example publication bytes differ")
            phase = "publication"
            owned.write_exclusive("identity.json", canonical(identity.to_dict()))
            owned.write_exclusive("training-input.json", result.prepared.request.canonical_json.encode("utf-8"))
            owned.write_exclusive("prepared-request.json", canonical(result.prepared.to_dict()))
            summary = {"schema": RESULT_SCHEMA, "status": "PREPARED_INPUT_ONLY",
                "output_root": str(output), "publication": str(publication),
                "identity_ref": identity.ref, "content_sha256": identity.content_digest,
                "source_bytes": identity.size_bytes, "rows": manifest["row_count"],
                "identity_file": "identity.json", "training_input_file": "training-input.json",
                "automatic_retry": False}
            if len(canonical(summary)) > 8192:
                raise ValueError("example summary exceeds bound")
            owned.write_exclusive("result.json", canonical(summary))
            owned.check()
            parent.check()
            return 0, summary
        except BaseException:
            summary = {"schema": RESULT_SCHEMA, "status": "REFUSED", "phase": phase,
                "automatic_retry": False}
            if owned is not None:
                try:
                    owned.write_exclusive("failure.json", canonical(summary))
                except BaseException:
                    pass  # Preserve uncertain custody and all partial bytes.
            return 2, summary


def profile_tokenizer(configuration):
    """Tokenize the verified publication exactly as the worker, without MLX.

    Linux/Darwin host diagnostic only: no model, runtime capture, admission,
    download, generation or training. Existing output roots always reject.
    """
    phase, owned = "configuration", None
    with ExitStack() as stack:
        try:
            if sys.platform not in ("linux", "darwin"):
                raise ValueError("POSIX tokenizer diagnosis only")
            path = Path(configuration).absolute()
            config = strict_json(_read(path, 128 * 1024), maximum=128 * 1024)
            if (type(config) is not dict or set(config) != {
                    "schema", "workload_file", "prepared_publication", "tokenizer_snapshot_file",
                    "transformers_version", "tokenizer_budget", "maximum_token_evidence_bytes", "output_root"}
                    or config["schema"] != TOKENIZER_SCHEMA
                    or type(config["transformers_version"]) is not str
                    or not re.fullmatch(r"[A-Za-z0-9.+-]{1,64}", config["transformers_version"])
                    or type(config["maximum_token_evidence_bytes"]) is not int
                    or not 1 <= config["maximum_token_evidence_bytes"] <= 1024 * 1024):
                raise ValueError("tokenizer profile fields differ")
            from tuner.runtime.native_mlx_child import _snapshot
            from tuner.runtime.native_mlx_grpo import prepared_prompts, tokenize_prompts
            from tuner.runtime.native_mlx_publication import NativeArtifactBudget
            from tuner.training.native_mlx_compilation import parse_native_workload

            phase = "input_admission"
            projection = parse_native_workload(_read(_path(config["workload_file"], path.parent), 2 * 1024 * 1024))
            if projection.workload.method != "grpo":
                raise ValueError("tokenizer profile requires a compiled GRPO workload")
            publication = _path(config["prepared_publication"], path.parent)
            publication_tree = stack.enter_context(NativeTree(publication, private=True))
            verifier = GRPOPreparedInputFormatVerifierV1()
            training_json = canonical(projection.workload.document["training_input"]).decode("utf-8")
            verifier.validate_training_input(publication, projection.source, training_json)
            observed = verifier.verify(publication)
            prompts = prepared_prompts(observed.snapshot, projection)
            snapshot = _snapshot(strict_json(_read(_path(config["tokenizer_snapshot_file"], path.parent), 1024 * 1024)))
            if (snapshot.repository, snapshot.revision) != (projection.profile.model_ref, projection.profile.tokenizer_revision):
                raise ValueError("tokenizer profile source pin differs")
            budget = NativeArtifactBudget(**config["tokenizer_budget"])
            if (len(snapshot.files) > 64 or any(item.size > budget.per_file for item in snapshot.files)
                    or sum(item.size for item in snapshot.files) > budget.aggregate):
                raise ValueError("tokenizer snapshot exceeds configured budget")
            token_tree = stack.enter_context(NativeTree(Path(snapshot.root), private=True))
            if token_tree.inventory(maximum_total=budget.aggregate) != snapshot.files:
                raise ValueError("tokenizer snapshot bytes differ")
            phase = "output"
            output = _path(config["output_root"], path.parent)
            parent = stack.enter_context(NativeTree(output.parent, private=True))
            _mkdir(parent, output.name)
            owned = stack.enter_context(NativeTree(output, private=True))
            owned.write_exclusive("configuration.json", canonical(config))
            phase = "tokenizer_load"
            version = importlib.metadata.version("transformers")
            if version != config["transformers_version"]:
                raise ValueError("tokenizer package version differs")
            # Standalone CLI imports/loads offline under closed output. Never
            # forward third-party warnings, paths or exception text to stdout.
            with open(os.devnull, "w") as silent, redirect_stdout(silent), redirect_stderr(silent):
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(snapshot.root, local_files_only=True,
                    trust_remote_code=False, use_fast=projection.tokenization.use_fast)
                phase = "input_tokenization"
                encoded, eos = tokenize_prompts(prompts, tokenizer=tokenizer, projection=projection)
            phase = "custody"
            if token_tree.inventory(maximum_total=budget.aggregate) != snapshot.files:
                raise ValueError("tokenizer changed during profiling")
            verifier.validate_training_input(publication, projection.source, training_json)
            publication_tree.check()
            phase = "publication"
            evidence = canonical({"schema": "synaptic-native-grpo-token-evidence/v1",
                "workload_digest": projection.workload.fingerprint, "source": projection.source.to_dict(),
                "tokenizer_snapshot_digest": snapshot.digest, "transformers_version": version,
                "tokenization": projection.tokenization.to_dict(), "return_dict": False,
                "add_generation_prompt": True, "prompt_tokens": encoded, "eos_token_id": eos})
            owned.write_exclusive("token-evidence.json", evidence, maximum=config["maximum_token_evidence_bytes"])
            lengths = [len(row) for row in encoded]
            summary = {"schema": TOKENIZER_RESULT_SCHEMA, "status": "TOKENIZER_ONLY_COMPLETE",
                "output_root": str(output), "workload_digest": projection.workload.fingerprint,
                "tokenizer_snapshot_digest": snapshot.digest, "transformers_version": version,
                "rows": len(encoded), "minimum_prompt_tokens": min(lengths), "maximum_prompt_tokens": max(lengths),
                "total_prompt_tokens": sum(lengths), "token_evidence_sha256": hashlib.sha256(evidence).hexdigest(),
                "automatic_retry": False}
            if len(canonical(summary)) > 8192:
                raise ValueError("tokenizer summary exceeds bound")
            owned.write_exclusive("result.json", canonical(summary))
            owned.check()
            parent.check()
            return 0, summary
        except BaseException:
            summary = {"schema": TOKENIZER_RESULT_SCHEMA, "status": "REFUSED", "phase": phase,
                "automatic_retry": False}
            if owned is not None:
                try:
                    owned.write_exclusive("failure.json", canonical(summary))
                except BaseException:
                    pass
            return 2, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "prepare", "profile-tokenizer"))
    parser.add_argument("configuration", help="reviewed example JSON configuration")
    args = parser.parse_args()
    try:
        status, summary = run(args.action, args.configuration)
    except BaseException:
        schema = TOKENIZER_RESULT_SCHEMA if args.action == "profile-tokenizer" else RESULT_SCHEMA
        status, summary = 2, {"schema": schema, "status": "REFUSED", "phase": "cleanup",
            "automatic_retry": False}
    print(canonical(summary).decode("utf-8"))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
