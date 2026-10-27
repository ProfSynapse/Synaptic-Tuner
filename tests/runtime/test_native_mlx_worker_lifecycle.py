"""Real worker dispatch/callback control flow on the existing NumPy ML doubles."""

import json
import os

from tests.runtime import test_native_mlx_grpo_diagnostics as fixtures
from tuner.runtime import native_mlx_child as child
from tuner.runtime.native_lifecycle_journal import NativeLifecycleLaunchV1
from tuner.runtime.native_lifecycle_protocol import NativeLifecyclePolicyV1
from tuner.runtime.native_mlx_material import canonical, strict_json

publication = fixtures.publication
pipeline = fixtures.pipeline


def test_grpo_v2_child_emits_updates_and_separate_saved_reload(pipeline):
    selected = NativeLifecycleLaunchV1(NativeLifecyclePolicyV1(32, 65536), "a" * 64)
    request = strict_json(pipeline.state.request)
    request.update(schema="synaptic-native-child-request/v2", lifecycle=selected.to_dict())
    path = pipeline.state.root / "request.json"
    path.write_bytes(canonical(request))
    read, write = os.pipe()
    child.sys.argv.extend(("--progress-fd", str(write)))
    try:
        assert child.main() == 0
        events = [json.loads(raw) for raw in os.read(read, 8192).splitlines()]
        result = strict_json((pipeline.state.root / "output" / "native_result.json").read_bytes())
        training = [event for event in events if event["phase"] == "training"]
        assert [event["step"] for event in training] == [value["step"] for value in result["metrics"]]
        assert [event["metrics"]["loss"] for event in training] == [value["loss"] for value in result["metrics"]]
        assert events[0] == {"kind": "progress", "phase": "preparation", "step": 0, "metrics": {}}
        assert events[-2]["phase"] == "reload" and events[-2]["metrics"] == {"loss": result["final_validation"]["loss"]}
        assert events[-1]["phase"] == "publication"
        assert all(event["kind"] == "progress" for event in events)
    finally:
        os.close(read)
