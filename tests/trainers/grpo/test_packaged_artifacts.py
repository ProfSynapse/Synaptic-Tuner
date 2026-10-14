"""Selected packaged save semantics; legacy core remains unchanged."""
import pytest
from pathlib import Path
import importlib


def _helpers():
    return importlib.import_module(".test_runtime_v1", __package__) if __package__ else importlib.import_module("test_runtime_v1")

def test_selected_adapter_stamp_binds_logical_model_and_refuses_links(tmp_path):
    from Trainers.grpo.runtime_v1 import _stamp_packaged_adapter
    import json
    path = tmp_path / "adapter_config.json"
    path.write_text('{"peft_type":"LORA","base_model_name_or_path":"/private/snapshot"}')
    _stamp_packaged_adapter(tmp_path, "org/model")
    assert json.loads(path.read_bytes())["base_model_name_or_path"] == "org/model"
    original = path.read_bytes()
    path.rename(tmp_path / "real.json")
    path.symlink_to(tmp_path / "real.json")
    with pytest.raises(OSError):
        _stamp_packaged_adapter(tmp_path, "other/model")
    assert (tmp_path / "real.json").read_bytes() == original


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'[]'])
def test_selected_adapter_stamp_rejects_malformed_without_rewrite(tmp_path, raw):
    from Trainers.grpo.runtime_v1 import _stamp_packaged_adapter
    path = tmp_path / "adapter_config.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        _stamp_packaged_adapter(tmp_path, "org/model")
    assert path.read_bytes() == raw


def test_selected_core_saves_tokenizer_in_native_model_layout(tmp_path):
    import json
    from Trainers.grpo import runtime_v1 as runtime
    helpers = _helpers()
    values = helpers.inputs(tmp_path)
    values["model_identity"]["tokenizer_revision"] = values["model_identity"]["revision"]
    values["workload"]["model"] = dict(values["model_identity"])
    events = []
    result = runtime.run_grpo(**values, _packaged_artifacts=True,
                              _dependency_factory=helpers.dependencies(events))
    assert (result["final_model"] / "tokenizer.json").read_bytes() == (result["tokenizer"] / "tokenizer.json").read_bytes()
    assert json.loads((result["final_model"] / "adapter_config.json").read_bytes())["base_model_name_or_path"] == "declared/model"


def test_selected_core_native_model_tokenizer_save_failure_propagates(tmp_path, monkeypatch):
    from Trainers.grpo import runtime_v1 as runtime
    helpers = _helpers()
    values = helpers.inputs(tmp_path)
    values["model_identity"]["tokenizer_revision"] = values["model_identity"]["revision"]
    values["workload"]["model"] = dict(values["model_identity"])
    original = helpers.Tokenizer.save_pretrained
    def fail_final(self, destination):
        if Path(destination).name == "final_model":
            raise OSError("private tokenizer save failed")
        return original(self, destination)
    monkeypatch.setattr(helpers.Tokenizer, "save_pretrained", fail_final)
    with pytest.raises(OSError):
        runtime.run_grpo(**values, _packaged_artifacts=True,
                         _dependency_factory=helpers.dependencies([]))
    assert not (values["output_root"] / "training_lineage.json").exists()


@pytest.mark.parametrize("replacement", ("leaf", "parent"))
def test_selected_stamp_substitution_before_write_never_modifies_replacement(tmp_path, monkeypatch, replacement):
    from Trainers.grpo import runtime_v1 as runtime
    root = tmp_path / "final"
    root.mkdir()
    path = root / "adapter_config.json"
    original = b'{"peft_type":"LORA","base_model_name_or_path":"original"}'
    path.write_bytes(original)
    original_parse = runtime.json.loads
    def substitute(*args, **kwargs):
        parsed = original_parse(*args, **kwargs)
        if replacement == "leaf":
            path.rename(root / "old.json")
            path.write_bytes(b'{"replacement":true}')
        else:
            root.rename(tmp_path / "old-root")
            root.mkdir()
            path.write_bytes(b'{"replacement":true}')
        return parsed
    monkeypatch.setattr(runtime.json, "loads", substitute)
    with pytest.raises(ValueError):
        runtime._stamp_packaged_adapter(root, "org/model")
    assert path.read_bytes() == b'{"replacement":true}'
    retained = root / "old.json" if replacement == "leaf" else tmp_path / "old-root" / "adapter_config.json"
    assert retained.read_bytes() == original
