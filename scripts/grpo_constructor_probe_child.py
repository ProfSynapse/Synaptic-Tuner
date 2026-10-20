"""Closed, constructor-only diagnostic inside an admitted GRPO image.

The caller supplies a verified offline snapshot and exact input bytes. This file
is transported as code by the derived-image qualifier, not installed or fetched.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import time
from pathlib import Path


SCHEMA = "synaptic-grpo-constructor-probe-result/v1"
STAGES = ("INPUT", "DEPENDENCIES", "CONFIG", "MODEL_LOAD", "LORA", "DATA", "REWARD", "CALLBACKS", "CONSTRUCTOR")
CATEGORIES = ("TYPE", "VALUE", "ATTRIBUTE", "KEY", "IMPORT", "MEMORY", "OS", "RUNTIME", "OTHER")


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _category(error: BaseException) -> str:
    for kind, label in ((TypeError, "TYPE"), (ValueError, "VALUE"), (AttributeError, "ATTRIBUTE"),
                        (KeyError, "KEY"), (ImportError, "IMPORT"), (MemoryError, "MEMORY"),
                        (OSError, "OS"), (RuntimeError, "RUNTIME")):
        if isinstance(error, kind):
            return label
    return "OTHER"


def diagnose(input_document: dict, *, model_root: Path, output_root: Path, dependencies=None) -> dict:
    """Use actual runtime preparation through construction; never train or generate."""
    stage = "INPUT"
    class_digest = None
    signature_digest = None
    try:
        if (type(input_document) is not dict or set(input_document) !=
                {"schema_version", "config", "rows", "reward_documents", "reward_schema", "qualification_policy"}
                or input_document["schema_version"] != "synaptic-grpo-constructor-probe-input/v1"
                or type(input_document["rows"]) is not list or len(input_document["rows"]) != 4
                or any(type(row) is not dict for row in input_document["rows"])):
            raise ValueError
        from Trainers.grpo import runtime_v1 as runtime
        config = input_document["config"]
        stage = "DEPENDENCIES"
        deps = dependencies() if dependencies is not None else runtime._dependencies()
        trainer_class = deps["trainer_class"]
        identity = (trainer_class.__module__, trainer_class.__qualname__)
        class_digest = hashlib.sha256(_canonical(identity)).hexdigest()
        signature_digest = hashlib.sha256(str(inspect.signature(trainer_class.__init__)).encode("utf-8")).hexdigest()
        stage = "CONFIG"
        args = runtime.build_grpo_config(config, output_root / "checkpoints",
                                         config_class=deps["config_class"], bf16_supported=deps["bf16_supported"])
        stage = "MODEL_LOAD"
        model, tokenizer, is_vl = deps["model_loader"].load_model_and_tokenizer(
            str(model_root), max_seq_length=config["model"]["max_seq_length"],
            dtype=config["model"].get("dtype"), load_in_4bit=config["model"].get("load_in_4bit", True), offline=True)
        if is_vl:
            raise ValueError
        stage = "LORA"
        model = deps["model_loader"].apply_lora_adapters(model=model, is_vision_model=False, **config["lora"])
        stage = "DATA"
        from Trainers.grpo.src.data_loader import dataset_from_rows, format_dataset_for_grpo
        data = dataset_from_rows(input_document["rows"], dataset_class=deps["dataset_class"])
        data = format_dataset_for_grpo(data, tokenizer,
            prompt_column=config["dataset"].get("prompt_column", "prompt"), num_proc=1,
            chat_template_kwargs=config["training"].get("chat_template_kwargs"))
        stage = "REWARD"
        from Trainers.grpo.src.rewards import build_combined_reward_function
        reward, _ = build_combined_reward_function(config["rewards"], Path("."),
            documents=input_document["reward_documents"], schema_config=input_document["reward_schema"], raw_only=True)
        measurements = runtime._QualificationMeasurements(input_document["qualification_policy"], time.monotonic() + 110)
        reward = measurements.reward_wrapper(reward)
        stage = "CALLBACKS"
        from transformers import TrainerCallback
        class Metrics(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                return control
        class Evidence(TrainerCallback):
            def on_train_begin(self, args, state, control, model=None, **kwargs):
                measurements.begin(model, args)
                return control
            def on_train_end(self, args, state, control, model=None, **kwargs):
                measurements.end(model)
                return control
            def on_log(self, args, state, control, logs=None, **kwargs):
                measurements.log(state, logs)
                return control
        stage = "CONSTRUCTOR"
        trainer = runtime.create_grpo_trainer(trainer_class=trainer_class, model=model,
            processing_class=tokenizer, reward_funcs=reward, args=args,
            train_dataset=data, callbacks=[Metrics(), Evidence()])
        del trainer
        status, category = "CONSTRUCTED", "NONE"
    except BaseException as error:
        status, category = "REJECTED", _category(error)
    return {"schema_version": SCHEMA, "status": status, "stage": stage,
            "category": category, "class_digest": class_digest,
            "signature_digest": signature_digest, "training_executed": False,
            "generation_executed": False}


def main() -> int:
    # Keep a private copy of stdout for one closed final record. Native libraries
    # may write to file descriptors, so redirect both descriptors before import.
    result_fd = os.dup(1)
    sink_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(sink_fd, 1)
    os.dup2(sink_fd, 2)
    os.close(sink_fd)
    result = {"schema_version": SCHEMA, "status": "REJECTED", "stage": "INPUT",
              "category": "OTHER", "class_digest": None, "signature_digest": None,
              "training_executed": False, "generation_executed": False}
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            raw = Path("/probe-input/input.json").read_bytes()
            if not 0 < len(raw) <= 65536:
                raise ValueError
            document = json.loads(raw)
            result = diagnose(document, model_root=Path("/probe-model"), output_root=Path("/probe"))
    except BaseException:
        pass
    payload = _canonical(result)
    if len(payload) > 8192:
        return 125
    os.write(result_fd, payload)
    os.close(result_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
