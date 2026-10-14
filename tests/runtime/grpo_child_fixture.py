"""CPU-only third-party doubles for the real isolated owned GRPO control path.

This substitutes image/ML-library observations explicitly. It never qualifies
Torch, Unsloth, a provider image, model training or source publication.
"""
from __future__ import annotations

from dataclasses import make_dataclass
import hashlib
import importlib.machinery
import json
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace


def install(source_root):
    from tuner.runtime import packaged_sft_execution as shared
    from tuner.runtime import packaged_sft_child
    from tuner.runtime import packaged_training_worker
    from tuner.runtime.offline_grpo_worker import GRPO_OFFLINE_MEMBERS
    if Path(shared.__file__).absolute().parents[2] != source_root:
        raise AssertionError("fixture imported wrong production origin")
    from Trainers.grpo import runtime_v1 as owned_core
    assert Path(owned_core.__file__).absolute().parents[2] == source_root
    # The source CPU lane can share an environment with an older installed
    # engine wheel. Select the already asserted source root for namespace
    # parents explicitly; production's installed-wheel guard is tested separately.
    for name in ("Trainers", "Trainers.grpo"):
        module = sys.modules.get(name)
        if module is not None and module.__spec__.origin is None:
            locations = [str(source_root.joinpath(*name.split(".")))]
            module.__path__ = locations
            module.__spec__.submodule_search_locations = locations
    members = {source_root / name: hashlib.sha256(shared.stable_read(source_root / name)).hexdigest()
               for name in GRPO_OFFLINE_MEMBERS if name.endswith(".py")}
    # Actual owned-source byte guard, with image inspection explicitly stubbed.
    guard = lambda release, trainer, **kwargs: packaged_sft_child._OwnedImportGuard(members)
    packaged_sft_child._installed_import_guard = guard
    sys.modules["tuner.runtime.packaged_grpo_child"]._installed_import_guard = guard
    shared._inspect_release = lambda release, **kwargs: source_root / "Trainers/grpo/runtime_v1.py"
    packaged_training_worker.admit_packaged_training_release = lambda raw, **kwargs: __import__(
        "tuner.runtime.releases", fromlist=["parse_packaged_runtime_release"]).parse_packaged_runtime_release(json.loads(raw))
    for name in ("torch", "unsloth", "trl", "datasets", "transformers"):
        module = ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = module
    sys.modules["torch"].cuda = SimpleNamespace(is_available=lambda: False)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tokenize"] is False
            return "\n".join(item["content"] for item in messages)
        def save_pretrained(self, destination):
            root = Path(destination)
            (root / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "ExampleTokenizer"}))
            (root / "tokenizer.json").write_text(json.dumps({"version": "1.0", "model": {"type": "BPE", "vocab": {"a": 0}}}))

    class Model:
        value = 0
        config = SimpleNamespace(_name_or_path="controlled-offline-fixture")
        def parameters(self):
            return ()

    class Fast:
        @staticmethod
        def from_pretrained(**kwargs):
            assert kwargs["token"] is False and kwargs["local_files_only"] is True
            assert (Path(kwargs["model_name"]) / "config.json").read_bytes() == b"{}"
            return Model(), Tokenizer()
        @staticmethod
        def get_peft_model(model, **kwargs):
            return model
    sys.modules["unsloth"].FastLanguageModel = Fast
    sys.modules["unsloth"].FastVisionModel = Fast
    sys.modules["unsloth"].is_bfloat16_supported = lambda: False

    class Dataset(list):
        @classmethod
        def from_list(cls, rows):
            return cls(rows)
        @property
        def column_names(self):
            return list(self[0])
        def map(self, function, **kwargs):
            return Dataset([function(row) for row in self])

    names = ("output_dir per_device_train_batch_size gradient_accumulation_steps num_generations "
             "max_completion_length temperature learning_rate weight_decay warmup_ratio lr_scheduler_type "
             "optim logging_steps save_steps save_total_limit num_train_epochs fp16 bf16 seed report_to beta "
             "max_prompt_length max_steps max_grad_norm").split()
    config_class = make_dataclass("CPUConfig", [(name, object) for name in names])
    class Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.state = SimpleNamespace(global_step=0, log_history=[])
        def train(self, **kwargs):
            assert kwargs == {"resume_from_checkpoint": None}
            assert self.kwargs["train_dataset"][0]["declared"] == "retained"
            for step in range(self.kwargs["args"].max_steps):
                scores = self.kwargs["reward_funcs"](["OK", "bad"])
                assert scores == [1.0, 0.0]
                self.kwargs["model"].value += 1
                self.state.global_step = step + 1
                record = {"loss": 1 / (step + 1), "reward": sum(scores) / len(scores)}
                self.state.log_history.append(record)
                for callback in self.kwargs["callbacks"]:
                    callback.on_log(None, self.state, None, logs=record)
            return SimpleNamespace(metrics={"train_loss": self.state.log_history[-1]["loss"]})
        def save_model(self, destination):
            root = Path(destination)
            (root / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": "/controlled/private"}))
            header = json.dumps({"lora_A": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
            (root / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", self.kwargs["model"].value))
    sys.modules["trl"].GRPOConfig = config_class
    sys.modules["trl"].GRPOTrainer = Trainer
    sys.modules["datasets"].Dataset = Dataset
    sys.modules["transformers"].TrainerCallback = object


def check_installed_guard(source_root, mutation):
    """Exercise the actual selected installed guard over a complete fixture wheel."""
    from io import BytesIO
    from zipfile import ZipFile, ZIP_DEFLATED
    from pathlib import PurePosixPath
    from tuner.runtime import packaged_sft_child as child
    from tuner.runtime import offline_grpo_worker as offline
    from tuner.runtime import packaged_sft_execution as shared
    from Trainers.grpo import runtime_v1 as core
    assert Path(core.__file__).absolute().parents[2] == source_root
    for name in ("Trainers", "Trainers.grpo"):
        module = sys.modules[name]
        if module.__spec__.origin is None:
            module.__path__ = [str(source_root.joinpath(*name.split(".")))]
            module.__spec__.submodule_search_locations = module.__path__
    names = set(offline.GRPO_OFFLINE_MEMBERS) | {"Trainers/sft/train_sft.py"}
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name in sorted(names):
            content = shared.stable_read(source_root / name)
            if mutation == "unselected_wheel_python" and name == "Trainers/sft/train_sft.py":
                content += b"\n# inconsistent wheel fixture\n"
            archive.writestr(name, content)
    wheel = buffer.getvalue()
    metadata = {"wheel": {"filename": "cpu-fixture.whl"}}
    original_read = child.stable_read
    def read(path, maximum=1024 * 1024):
        if path == Path("/opt/synaptic-runtime/build-inputs.json"):
            return json.dumps(metadata).encode()
        if path == Path("/opt/synaptic-runtime/cpu-fixture.whl"):
            return wheel
        return original_read(path, maximum)
    child.stable_read = read
    class Distribution:
        files = tuple(PurePosixPath(name) for name in names)
        def locate_file(self, name):
            return source_root / name
    child.importlib.metadata.distribution = lambda name: Distribution()
    document = offline.captured_grpo_manifest_document(source_root)
    if mutation == "declared_hash":
        next(item for item in document["members"] if item["path"] == "Trainers/grpo/src/model_loader.py")["sha256"] = "0" * 64
        from tuner.runtime.offline_sft_worker import closure_digest
        document["closure_digest"] = closure_digest(document)
    manifest = offline.parse_offline_grpo_worker_manifest(offline._canonical_json(document) + b"\n")
    offline.load_packaged_offline_grpo_worker_manifest = lambda: manifest
    release = SimpleNamespace(package_name="synaptic-tuner", package_digest=hashlib.sha256(wheel).hexdigest())
    try:
        guard = child._installed_import_guard(release, source_root / "Trainers/grpo/runtime_v1.py", method="grpo")
    except (ValueError, ImportError, RuntimeError):
        assert mutation in {"declared_hash", "unselected_wheel_python"}
        return
    assert mutation == "valid"
    assert source_root / "Trainers/grpo/src/model_loader.py" in guard.members
    assert source_root / "Trainers/sft/train_sft.py" not in guard.members
    # Python present in the exact wheel is still not executable if omitted from
    # the separately declared GRPO closure. No ambient owned-source fallback.
    try:
        guard.find_spec("Trainers.sft.train_sft", [str(source_root / "Trainers/sft")])
    except ImportError:
        pass
    else:
        raise AssertionError("undeclared owned source was accepted")
