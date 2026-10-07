"""Contract proofs for the cold provider-neutral training-input loader."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import synaptic_tuner.api.v1.training_input_loader as loader
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_input_loader import (
    LoadedTrainingInputContractV1,
    TrainingInputContractCodeV1,
    TrainingInputContractErrorV1,
    TrainingInputContractIdentityV1,
    load_training_input_contract_v1,
)


ROOT = Path(__file__).resolve().parents[2]
MODULE_IDS = (
    "synaptic_tuner.api.v1._contract",
    "synaptic_tuner.api.v1.training_input",
    "synaptic_tuner.api.v1.training_input_loader",
)
PUBLIC_NAMES = [
    "LoadedTrainingInputContractV1",
    "TrainingInputContractCodeV1",
    "TrainingInputContractErrorV1",
    "TrainingInputContractIdentityV1",
    "load_training_input_contract_v1",
]


def _document() -> dict[str, object]:
    return {
        "schema_version": "synaptic-training-input/v1",
        "method": "sft",
        "model": {
            "ref": "organization/model",
            "revision": "revision-1",
            "tokenizer_revision": "tokenizer-1",
        },
        "dataset": {"ref": "dataset://organization/corpus"},
        "hyperparameters": {
            "schema_version": "synaptic-sft-hyperparameters/v1",
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 0.0002,
            "duration": {"max_steps": 100, "num_epochs": None},
            "max_seq_length": 2048,
            "seed": 42,
            "save_steps": 25,
            "save_total_limit": 2,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ["k_proj", "q_proj", "v_proj"],
            "use_dora": False,
            "use_rslora": True,
            "init_lora_weights": True,
            "split_dataset": False,
        },
        "artifacts": {
            "required_kinds": ["final_model", "training_lineage"],
            "retain_checkpoints": True,
        },
    }


def _canonical(value: dict[str, object]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _fixture_finder(paths: dict[str, Path]):
    return lambda module_id: SimpleNamespace(origin=str(paths[module_id]))


def _ready_record() -> loader._ReadyTrainingInputContractV1:
    bundle = load_training_input_contract_v1()
    module = sys.modules["synaptic_tuner.api.v1.training_input"]
    return loader._ReadyTrainingInputContractV1(
        module, Path(module.__file__).resolve(), loader._TRAINING_INPUT_PARSER, bundle
    )


def test_public_load_is_exact_cached_and_immutable() -> None:
    first = load_training_input_contract_v1()
    second = load_training_input_contract_v1()
    assert type(first) is LoadedTrainingInputContractV1
    assert first is second
    assert first.identity is second.identity
    assert type(first.identity) is TrainingInputContractIdentityV1
    assert first.input_type is TrainingInputV1
    assert not hasattr(first, "parser")
    assert set(LoadedTrainingInputContractV1.__annotations__) == {
        "identity", "input_type",
    }
    assert not hasattr(first, "__dict__")
    assert not hasattr(first.identity, "__dict__")
    with pytest.raises(AttributeError):
        first.input_type = object  # type: ignore[misc,assignment]


def test_identity_and_implementation_digests_are_exact() -> None:
    closure: dict[str, object] = {}
    for module_id in MODULE_IDS:
        spec = importlib.util.find_spec(module_id)
        assert spec is not None and spec.origin is not None
        closure[module_id] = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    implementation = hashlib.sha256(
        b"synaptic-training-input-implementation/v1\0" + _canonical(closure)
    ).hexdigest()
    body: dict[str, object] = {
        "schema_version": "synaptic-training-input-contract-identity/v1",
        "contract_schema": "synaptic-training-input/v1",
        "module_name": "synaptic_tuner.api.v1.training_input",
        "type_name": "TrainingInputV1",
        "parser_name": "from_json",
        "implementation_digest": implementation,
    }
    expected_identity = hashlib.sha256(
        b"synaptic-training-input-contract-identity/v1\0" + _canonical(body)
    ).hexdigest()
    identity = load_training_input_contract_v1().identity
    assert dataclasses.asdict(identity) == {**body, "identity_digest": expected_identity}


def test_parse_json_uses_captured_canonical_parser_and_exact_result(monkeypatch) -> None:
    contract = load_training_input_contract_v1()
    monkeypatch.setattr(
        TrainingInputV1,
        "from_json",
        classmethod(lambda _cls, _value: pytest.fail("live parser was consulted")),
    )
    parsed = contract.parse_json(json.dumps(_document()))
    assert type(parsed) is TrainingInputV1
    assert parsed.to_dict() == _document()


def test_parse_json_is_exact_and_errors_are_closed() -> None:
    contract = load_training_input_contract_v1()
    with pytest.raises(TrainingInputContractErrorV1) as structural:
        contract.parse_json(b"{}")  # type: ignore[arg-type]
    assert structural.value.code is TrainingInputContractCodeV1.INPUT_INVALID
    secret = "private-value-must-not-leak"
    with pytest.raises(TrainingInputContractErrorV1) as captured:
        contract.parse_json(json.dumps({"secret": secret}))
    assert captured.value.code is TrainingInputContractCodeV1.INPUT_INVALID
    assert captured.value.args == ("input_invalid",)
    assert secret not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_fixed_closure_digest_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    paths = {
        module_id: tmp_path / f"member-{index}.py"
        for index, module_id in enumerate(MODULE_IDS)
    }
    for index, path in enumerate(paths.values()):
        path.write_bytes(f"member = {index}\n".encode("ascii"))
    finder = _fixture_finder(paths)
    first = loader._implementation_digest_v1(_find_spec=finder)
    assert loader._implementation_digest_v1(_find_spec=finder) == first
    paths[MODULE_IDS[1]].write_bytes(b"member = 'changed'\n")
    assert loader._implementation_digest_v1(_find_spec=finder) != first


def test_closure_rejects_missing_oversize_and_nonregular_members(tmp_path: Path) -> None:
    paths = {
        module_id: tmp_path / f"member-{index}.py"
        for index, module_id in enumerate(MODULE_IDS)
    }
    for path in paths.values():
        path.write_bytes(b"source\n")
    finder = _fixture_finder(paths)
    paths[MODULE_IDS[0]].unlink()
    with pytest.raises(TrainingInputContractErrorV1) as missing:
        loader._implementation_digest_v1(_find_spec=finder)
    assert missing.value.code is TrainingInputContractCodeV1.CONTRACT_UNAVAILABLE
    paths[MODULE_IDS[0]].write_bytes(b"x" * (256 * 1024 + 1))
    with pytest.raises(TrainingInputContractErrorV1):
        loader._implementation_digest_v1(_find_spec=finder)
    paths[MODULE_IDS[0]].unlink()
    paths[MODULE_IDS[0]].mkdir()
    with pytest.raises(TrainingInputContractErrorV1):
        loader._implementation_digest_v1(_find_spec=finder)


def test_closure_enforces_aggregate_bound_independently() -> None:
    finder = lambda module_id: SimpleNamespace(origin=f"{module_id}.py")
    with pytest.raises(RuntimeError, match="unavailable"):
        loader._implementation_digest_v1(
            _find_spec=finder,
            _source_reader=lambda _path: b"x" * 400_000,
        )


def test_source_reader_rejects_symlink_and_mid_read_identity_change(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"source\n")
    observed = source.stat()
    with pytest.raises(TrainingInputContractErrorV1):
        loader._read_stable_source(
            source,
            _after_read=lambda: os.utime(
                source,
                ns=(observed.st_atime_ns, observed.st_mtime_ns + 1_000_000),
            ),
        )
    target = tmp_path / "target.py"
    target.write_bytes(b"target\n")
    link = tmp_path / "link.py"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(TrainingInputContractErrorV1):
        loader._read_stable_source(link)


def test_public_loader_totalizes_private_failures() -> None:
    secret = "C:/private/contract-source.py"
    local = loader._install_contract_loader_v1(
        lambda: (_ for _ in ()).throw(RuntimeError(secret))
    )
    with pytest.raises(TrainingInputContractErrorV1) as captured:
        local()
    assert captured.value.code is TrainingInputContractCodeV1.CONTRACT_UNAVAILABLE
    assert secret not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_condition_loader_builds_once_and_converges_for_32_callers() -> None:
    started = threading.Event()
    release = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def build() -> loader._ReadyTrainingInputContractV1:
        nonlocal calls
        with calls_lock:
            calls += 1
        started.set()
        assert release.wait(5)
        return _ready_record()

    local = loader._install_contract_loader_v1(build)
    results: list[LoadedTrainingInputContractV1] = []
    failures: list[BaseException] = []

    def call() -> None:
        try:
            results.append(local())
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=call) for _ in range(32)]
    for thread in threads:
        thread.start()
    assert started.wait(5)
    release.set()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert failures == []
    assert calls == 1
    assert len(results) == 32
    assert all(result is results[0] for result in results)
    assert local() is results[0]


def test_failed_loader_converges_with_fresh_context_free_errors() -> None:
    started = threading.Event()
    release = threading.Event()
    calls = 0
    secret = "private-builder-trace"

    def build() -> loader._ReadyTrainingInputContractV1:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(5)
        raise RuntimeError(secret)

    local = loader._install_contract_loader_v1(build)
    failures: list[TrainingInputContractErrorV1] = []

    def call() -> None:
        try:
            local()
        except TrainingInputContractErrorV1 as error:
            failures.append(error)

    threads = [threading.Thread(target=call) for _ in range(32)]
    for thread in threads:
        thread.start()
    assert started.wait(5)
    release.set()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert calls == 1
    assert len(failures) == 32
    assert len({id(error) for error in failures}) == 32
    assert all(
        error.code is TrainingInputContractCodeV1.CONTRACT_UNAVAILABLE
        and error.args == ("contract_unavailable",)
        and error.__cause__ is None
        and error.__context__ is None
        and secret not in str(error)
        for error in failures
    )
    with pytest.raises(TrainingInputContractErrorV1) as later:
        local()
    assert later.value is not failures[0]
    assert later.value.code is TrainingInputContractCodeV1.CONTRACT_UNAVAILABLE


def test_same_thread_reentry_can_be_handled_or_terminal() -> None:
    handled_codes: list[TrainingInputContractCodeV1] = []
    handled_loader = None

    def handled_build() -> loader._ReadyTrainingInputContractV1:
        assert handled_loader is not None
        try:
            handled_loader()
        except TrainingInputContractErrorV1 as error:
            handled_codes.append(error.code)
            assert error.__cause__ is error.__context__ is None
        return _ready_record()

    handled_loader = loader._install_contract_loader_v1(handled_build)
    assert handled_loader() is load_training_input_contract_v1()
    assert handled_codes == [TrainingInputContractCodeV1.LOAD_REENTRANT]

    terminal_loader = None

    def terminal_build() -> loader._ReadyTrainingInputContractV1:
        assert terminal_loader is not None
        terminal_loader()
        return _ready_record()  # pragma: no cover

    terminal_loader = loader._install_contract_loader_v1(terminal_build)
    observed = []
    for _ in range(2):
        with pytest.raises(TrainingInputContractErrorV1) as captured:
            terminal_loader()
        observed.append(captured.value)
    assert observed[0] is not observed[1]
    assert all(
        error.code is TrainingInputContractCodeV1.LOAD_REENTRANT
        and error.__cause__ is None
        and error.__context__ is None
        for error in observed
    )


def test_interrupted_waiter_does_not_poison_authoritative_build() -> None:
    class InterruptingCondition(threading.Condition):
        def wait(self, timeout=None):
            if threading.current_thread().name == "interrupted-waiter":
                raise KeyboardInterrupt("waiter-secret")
            return super().wait(timeout)

    started = threading.Event()
    release = threading.Event()

    def build() -> loader._ReadyTrainingInputContractV1:
        started.set()
        assert release.wait(5)
        return _ready_record()

    local = loader._install_contract_loader_v1(
        build, _condition_factory=InterruptingCondition
    )
    owner_result: list[LoadedTrainingInputContractV1] = []
    owner = threading.Thread(target=lambda: owner_result.append(local()))
    owner.start()
    assert started.wait(5)
    interrupted: list[TrainingInputContractErrorV1] = []

    def wait() -> None:
        try:
            local()
        except TrainingInputContractErrorV1 as error:
            interrupted.append(error)

    waiter = threading.Thread(target=wait, name="interrupted-waiter")
    waiter.start()
    waiter.join(5)
    assert not waiter.is_alive()
    assert len(interrupted) == 1
    assert interrupted[0].code is TrainingInputContractCodeV1.LOAD_INTERRUPTED
    assert interrupted[0].__cause__ is interrupted[0].__context__ is None
    assert "waiter-secret" not in str(interrupted[0])
    release.set()
    owner.join(5)
    assert not owner.is_alive()
    assert owner_result == [load_training_input_contract_v1()]
    assert local() is owner_result[0]


def test_parser_throw_and_wrong_result_are_fresh_context_free_errors() -> None:
    canonical = load_training_input_contract_v1()
    for parser in (
        lambda _value: (_ for _ in ()).throw(SystemExit("parser-secret")),
        lambda _value: object(),
    ):
        candidate = LoadedTrainingInputContractV1(
            canonical.identity, canonical.input_type
        )
        object.__setattr__(
            candidate, "_LoadedTrainingInputContractV1__parser", parser
        )
        with pytest.raises(TrainingInputContractErrorV1) as captured:
            candidate.parse_json("{}")
        assert captured.value.code is TrainingInputContractCodeV1.INPUT_INVALID
        assert captured.value.__cause__ is captured.value.__context__ is None
        assert "parser-secret" not in str(captured.value)


def _contains_text(value: object, needle: str, seen: set[int]) -> bool:
    if type(value) is str:
        return needle in value
    if type(value) is bytes:
        return needle.encode("utf-8") in value
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if type(value) is dict:
        return any(
            _contains_text(item, needle, seen)
            for pair in value.items()
            for item in pair
        )
    if type(value) in (tuple, list, set, frozenset):
        return any(_contains_text(item, needle, seen) for item in value)
    for value_type in type(value).__mro__:
        slots = value_type.__dict__.get("__slots__", ())
        if type(slots) is str:
            slots = (slots,)
        for slot in slots:
            if slot in {"__dict__", "__weakref__"}:
                continue
            try:
                nested = object.__getattribute__(value, slot)
            except BaseException:
                continue
            if _contains_text(nested, needle, seen):
                return True
    try:
        namespace = object.__getattribute__(value, "__dict__")
    except BaseException:
        return False
    return type(namespace) is dict and _contains_text(namespace, needle, seen)


def _capture_parse_error(
    contract: LoadedTrainingInputContractV1, raw: str,
) -> TrainingInputContractErrorV1:
    observed: TrainingInputContractErrorV1 | None = None
    try:
        contract.parse_json(raw)
    except TrainingInputContractErrorV1 as error:
        observed = error
    del raw
    del contract
    assert type(observed) is TrainingInputContractErrorV1
    return observed


def test_public_parse_traceback_graph_retains_no_raw_input_or_wrong_result() -> None:
    canonical = load_training_input_contract_v1()
    probes = (
        (
            canonical,
            json.dumps({"path": "C:/private/raw-input", "secret": "raw-secret"}),
            ("C:/private/raw-input", "raw-secret"),
        ),
        (
            LoadedTrainingInputContractV1(
                canonical.identity, canonical.input_type
            ),
            "{}",
            ("C:/private/wrong-result", "wrong-result-secret"),
        ),
    )
    wrong_result = {
        "path": "C:/private/wrong-result", "secret": "wrong-result-secret"
    }
    object.__setattr__(
        probes[1][0],
        "_LoadedTrainingInputContractV1__parser",
        lambda _value: wrong_result,
    )
    for contract, raw, needles in probes:
        captured = _capture_parse_error(contract, raw)
        assert captured.code is TrainingInputContractCodeV1.INPUT_INVALID
        assert captured.__cause__ is captured.__context__ is None
        traceback = captured.__traceback__
        frames = []
        while traceback is not None:
            frames.append(traceback.tb_frame)
            traceback = traceback.tb_next
        assert [frame.f_code.co_name for frame in frames] == [
            "_capture_parse_error", "parse_json", "_raise_input_invalid",
        ]
        for frame in frames:
            for needle in needles:
                assert not _contains_text(frame.f_locals, needle, set())


def test_error_contract_is_final_closed_and_immutable() -> None:
    assert TrainingInputContractCodeV1.__final__ is True
    assert TrainingInputContractErrorV1.__final__ is True
    error = TrainingInputContractErrorV1(
        TrainingInputContractCodeV1.CONTRACT_UNAVAILABLE
    )
    assert error.args == ("contract_unavailable",)
    with pytest.raises(AttributeError):
        error.code = TrainingInputContractCodeV1.INPUT_INVALID
    with pytest.raises(AttributeError):
        error.args = ("secret",)
    with pytest.raises(TypeError):
        TrainingInputContractErrorV1("contract_unavailable")  # type: ignore[arg-type]


def test_host_authenticity_is_exact_cached_bundle_identity() -> None:
    authentic = load_training_input_contract_v1()
    equal_looking = LoadedTrainingInputContractV1(
        authentic.identity, authentic.input_type
    )
    assert equal_looking == authentic
    assert equal_looking is not load_training_input_contract_v1()
    assert authentic is load_training_input_contract_v1()


def test_public_identity_discloses_no_closure_members_or_paths() -> None:
    rendered = repr(load_training_input_contract_v1().identity)
    assert "_contract" not in rendered
    assert "training_input_loader" not in rendered
    assert str(ROOT) not in rendered
    assert set(dataclasses.asdict(load_training_input_contract_v1().identity)) == {
        "schema_version", "contract_schema", "module_name", "type_name",
        "parser_name", "implementation_digest", "identity_digest",
    }


def test_lazy_root_exports_are_exact_identities_and_cold() -> None:
    script = f"""
import json, sys
sys.path.insert(0, {str(ROOT)!r})
import synaptic_tuner.api.v1 as api
before = sorted(name for name in sys.modules if name.endswith(('training_input', 'training_input_loader')))
values = [api.LoadedTrainingInputContractV1, api.TrainingInputContractCodeV1, api.TrainingInputContractErrorV1, api.TrainingInputContractIdentityV1, api.load_training_input_contract_v1]
after = sorted(name for name in sys.modules if name.endswith(('training_input', 'training_input_loader')))
print(json.dumps({{'before': before, 'after': after, 'modules': [value.__module__ for value in values]}}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    observed = json.loads(completed.stdout)
    assert observed["before"] == []
    assert observed["after"] == [
        "synaptic_tuner.api.v1.training_input",
        "synaptic_tuner.api.v1.training_input_loader",
    ]
    assert observed["modules"] == [
        "synaptic_tuner.api.v1.training_input_loader",
        "synaptic_tuner.api.v1.training_input_loader",
        "synaptic_tuner.api.v1.training_input_loader",
        "synaptic_tuner.api.v1.training_input_loader",
        "synaptic_tuner.api.v1.training_input_loader",
    ]


def test_exports_and_fixed_closure_imports_are_exact() -> None:
    import synaptic_tuner.api.v1 as api

    assert loader.__all__ == PUBLIC_NAMES
    for name in PUBLIC_NAMES:
        assert getattr(api, name) is getattr(loader, name)
    local_imports: dict[str, set[str]] = {}
    for module_id in MODULE_IDS:
        spec = importlib.util.find_spec(module_id)
        assert spec is not None and spec.origin is not None
        tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
        local_imports[module_id] = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level and node.module
        }
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module and not node.level
        )
        assert imports.isdisjoint(
            {"synaptic_host", "tuner", "modal", "huggingface_hub", "runpod",
             "sqlite3", "subprocess", "socket", "requests"}
        )
    assert local_imports == {
        "synaptic_tuner.api.v1._contract": set(),
        "synaptic_tuner.api.v1.training_input": {"_contract"},
        "synaptic_tuner.api.v1.training_input_loader": {"training_input"},
    }
