"""Provider-free controls and independent objective references; no MLX on Linux."""
from dataclasses import asdict, replace
import hashlib
import math
from types import SimpleNamespace

import pytest

from tests.runtime.test_native_mlx_child import inputs as sft_inputs
from tests.runtime import test_native_mlx_publication as publication_fixtures
from tests.runtime import test_native_mlx_family as family
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.native_mlx import NativeMLXGRPOProfile, compile_native_grpo, NativePlanningV1
from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest, RunLogsRequest, RunOperationError
from tuner.runtime.native_mlx_grpo import prepared_prompts, response_reward, base_reference, generate_completion
from tuner.runtime.native_mlx_grpo_math import group_advantages, grpo_objective, completion_logprobs
from tuner.runtime.native_mlx_grpo_publication import verify_rollout_evidence
from tuner.runtime.native_mlx_material import canonical, strict_json
from tuner.runtime.native_mlx_publication import verify_native_publication
from tuner.training.grpo_compilation import grpo_prepared_semantic_digest
from tuner.training.native_mlx_compilation import parse_native_workload

publication = publication_fixtures.publication


def inputs(raw=None):
    raw = raw or (canonical({"prompt": "Pick red or blue", "answer": "SECRET-GOLD"}) + b"\n" +
                  canonical({"prompt": [{"role": "user", "content": "Choose a color"}]}) + b"\n")
    digest = hashlib.sha256(raw).hexdigest()
    semantic = grpo_prepared_semantic_digest(content_digest=digest, size_bytes=len(raw),
        row_count=len(raw.splitlines()), prompt_column="prompt")
    source = PreparedTrainingInputIdentity("prepared://sha256/" + semantic, semantic, digest,
        len(raw), "synaptic-grpo-prompts/v1")
    old_training, _, profile = sft_inputs()
    document = old_training.to_dict()
    document.update(method="grpo", artifacts={"required_kinds": sorted(("final_model", "tokenizer",
        "training_metrics", "training_lineage", "workload_record")), "retain_checkpoints": False})
    document["hyperparameters"] = dict(schema_version="synaptic-grpo-hyperparameters/v1",
        batch_size=4, gradient_accumulation_steps=1, num_generations=2, max_steps=3,
        max_seq_length=128, max_prompt_length=64, max_completion_length=8, learning_rate=.0001,
        temperature=1., beta=.1, weight_decay=0., warmup_ratio=0., max_grad_norm=1., seed=17,
        logging_steps=2, save_steps=4, save_total_limit=1, lora_rank=4, lora_alpha=32,
        lora_dropout=0., lora_target_modules=["self_attn.q_proj", "self_attn.v_proj"],
        use_dora=False, use_rslora=False, load_in_4bit=True, bf16=False, fp16=False, prompt_column="prompt",
        chat_template_kwargs={"enable_thinking": False})
    document["dataset"]["ref"] = source.ref
    document["rewards"] = {"items": [{"name": "blue", "weight": 1.}], "documents": {"blue": {
        "name": "blue", "scope": "response", "validation": [{"type": "regex", "match": r"\bblue\b"}],
        "scoring": {"strategy": "binary", "on_pass": 1., "on_fail": 0.}}}, "schema": {}}
    profile = NativeMLXGRPOProfile(**{**asdict(profile), "profile_ref": "candidate-grpo", "maximum_batch_tokens": 512},
        objective="mlxrl-token-mean-on-policy/v1", upstream_commit="5ee8a6bd7764b0585094d78c530ac87d26364034",
        maximum_rollout_tokens=96, maximum_reward_calls=12, maximum_evidence_bytes=65536, quantization_bits=4,
        reward_engine="regex-v0/v1", reward_timeout_ms=10)
    return TrainingInputV1.from_dict(document), source, profile, raw


def test_public_compilation_preserves_sft_and_recompiles_grpo():
    from tuner.training.native_mlx_compilation import compile_native_sft
    legacy = compile_native_sft(*sft_inputs())
    assert parse_native_workload(legacy.workload.canonical_bytes) == legacy
    training, source, profile, _ = inputs()
    projection = compile_native_grpo(training, source, profile)
    assert parse_native_workload(projection.workload.canonical_bytes) == projection
    assert projection.controls.batch_size == 4 and projection.optimizer.schedule == "constant"
    changed = projection.workload.document
    changed["controls"]["beta"] = .2
    with pytest.raises(ValueError, match="compilation"):
        parse_native_workload(canonical(changed))
    with pytest.raises(ValueError):
        compile_native_grpo(training, source, sft_inputs()[2])
    assert legacy.workload.document["schema_version"] == "synaptic-native-mlx-workload/v1"


@pytest.mark.parametrize("changes", [
    {"gradient_accumulation_steps": 2}, {"temperature": .7}, {"warmup_ratio": .1},
    {"use_dora": True}, {"use_rslora": True}, {"lora_dropout": .1}, {"bf16": True},
    {"fp16": True}, {"save_steps": 3}, {"save_total_limit": 2}, {"load_in_4bit": False},
    {"weight_decay": .1}, {"max_steps": 4},
])
def test_unsupported_controls_cannot_silently_change_objective(changes):
    training, source, profile, _ = inputs()
    value = training.to_dict()
    value["hyperparameters"].update(changes)
    with pytest.raises(ValueError):
        compile_native_grpo(TrainingInputV1.from_dict(value), source, profile)


@pytest.mark.parametrize("changes", [{"maximum_rollout_tokens": 95}, {"maximum_reward_calls": 11},
                                      {"maximum_batch_tokens": 511}, {"model_revision": "f" * 40},
                                      {"reward_timeout_ms": 1000}])
def test_profile_bounds_are_checked_before_imports(changes):
    training, source, profile, _ = inputs()
    with pytest.raises(ValueError):
        compile_native_grpo(training, source, replace(profile, **changes))


def test_only_selected_prompt_reaches_generation_boundary():
    training, source, profile, raw = inputs()
    projection = compile_native_grpo(training, source, profile)
    rows = prepared_prompts(raw, projection)
    assert rows[0] == [{"role": "user", "content": "Pick red or blue"}]
    assert "SECRET-GOLD" not in str(rows)
    assert response_reward("blue", training.rewards, timeout_ms=10) == 1
    assert response_reward("red", training.rewards, timeout_ms=10) == 0
    with pytest.raises(ValueError, match="bytes"):
        prepared_prompts(raw.replace(b"blue", b"BLUE"), projection)


@pytest.mark.parametrize("prompt", [[{"role": "assistant", "content": "gold answer"}],
    [{"role": "tool", "content": "external"}], ["bad-message"], ""])
def test_invalid_or_target_ended_prompts_reject(prompt):
    training, source, profile, raw = inputs(canonical({"prompt": prompt}) + b"\n")
    with pytest.raises(ValueError, match="boundary"):
        prepared_prompts(raw, compile_native_grpo(training, source, profile))


class NumpyBackend:
    def __init__(self):
        self.np = pytest.importorskip("numpy")
    def __getattr__(self, name):
        return getattr(self.np, name)
    def stop_gradient(self, value):
        return self.np.array(value, copy=True)
    def logsumexp(self, value, axis):
        maximum = self.np.max(value, axis=axis, keepdims=True)
        return self.np.squeeze(maximum + self.np.log(self.np.exp(value - maximum).sum(axis=axis, keepdims=True)), axis=axis)


def test_objective_matches_independent_probability_loss_and_analytic_gradient():
    mx = NumpyBackend()
    rewards = mx.array([0., 2., 7., 7.])
    advantages = group_advantages(rewards, 2, mx=mx)
    # Independent scalar normalization: groups have population std 1 and 0.
    expected_advantages = [-1 / math.sqrt(1 + 1e-8), 1 / math.sqrt(1 + 1e-8), 0, 0]
    assert advantages.tolist() == pytest.approx(expected_advantages)
    probabilities = [[.2, .3, .4], [.7, .2, .1], [.3, .4, .2], [.6, .5, .1]]
    old_probabilities = [[.3, .2, .4], [.5, .3, .2], [.2, .4, .1], [.5, .4, .2]]
    references = [[.1, .2, .3], [.6, .4, .2], [.1, .3, .2], [.5, .6, .2]]
    mask = mx.array([[1., 1., 0.], [1., 0., 0.], [1., 1., 1.], [1., 0., 0.]])
    policy, old, ref = (mx.log(mx.array(x)) for x in (probabilities, old_probabilities, references))
    count = 7
    loss, gradients = 0., []
    for i in range(4):
        row = []
        for j in range(3):
            p, q, r = probabilities[i][j], old_probabilities[i][j], references[i][j]
            if mask[i, j]:
                loss += (-p / q * expected_advantages[i] + .2 * (r / p - math.log(r / p) - 1)) / count
                row.append((-p / q * expected_advantages[i] + .2 * (1 - r / p)) / count)
            else:
                row.append(0.)
        gradients.append(row)
    assert float(grpo_objective(policy, old, ref, advantages, mask, .2, mx=mx)) == pytest.approx(loss, abs=2e-8)
    for i in range(4):
        for j in range(3):
            hi, lo = policy.copy(), policy.copy()
            hi[i, j] += 1e-5
            lo[i, j] -= 1e-5
            numerical = (grpo_objective(hi, old, ref, advantages, mask, .2, mx=mx)
                         - grpo_objective(lo, old, ref, advantages, mask, .2, mx=mx)) / 2e-5
            assert numerical == pytest.approx(gradients[i][j], abs=2e-8)
    policy[mask == 0] = 1e30
    assert math.isfinite(float(grpo_objective(policy, old, ref, advantages, mask, .2, mx=mx)))


def test_completion_logprobs_gather_causal_generated_targets_only():
    mx = NumpyBackend()
    seen = []
    logits = mx.array([[[3., 1., 0.], [0., 2., 1.], [1., 0., 4.]]])
    def model(tokens):
        seen.append(tokens.tolist())
        return logits[:, :tokens.shape[1]]
    values, mask = completion_logprobs(model, [(2, 1), (0,)], [(2, 1), (1,)], mx=mx)
    # First target is at causal position prompt length minus one, not token 0.
    expected = [1 - math.log(math.exp(0) + math.exp(2) + math.exp(1)),
                0 - math.log(math.exp(1) + math.exp(0) + math.exp(4))]
    assert values.tolist()[0] == pytest.approx(expected, abs=2e-6)
    assert mask.tolist() == [[1., 1.], [1., 0.]]
    assert values[1, 1] == 0 and seen == [[[2, 1, 2]], [[0]]]


def test_zero_beta_has_no_kl_overflow_or_reference_dependency():
    mx = NumpyBackend()
    policy = mx.array([[-1000.], [-1.]])
    old = policy.copy()
    # Both are valid log-probabilities; exp(reference-policy) would overflow.
    reference = mx.array([[-1.], [-1000.]])
    value = grpo_objective(policy, old, reference, mx.array([-1., 1.]), mx.ones((2, 1)), 0., mx=mx)
    assert float(value) == 0.


def test_reference_scale_is_restored_on_failure():
    class LoRA:
        scale = 8.
    module = LoRA()
    model = SimpleNamespace(named_modules=lambda: [("layer", module)])
    with pytest.raises(RuntimeError):
        with base_reference(model, lora_type=LoRA):
            assert module.scale == 0
            raise RuntimeError("fixture")
    assert module.scale == 8


def test_pathological_reward_search_is_time_bounded():
    training, _, _, _ = inputs()
    rewards = training.to_dict()["rewards"]
    rewards["documents"]["blue"]["validation"][0]["match"] = "(a|aa)+$"
    with pytest.raises(TimeoutError):
        response_reward("a" * 10000 + "!", rewards, timeout_ms=1)


def test_generation_stops_at_eos_or_budget_and_never_reuses_group_cache():
    mx = NumpyBackend()
    samples = iter([1, 2, 0, 0, 0])
    mx.random = SimpleNamespace(categorical=lambda _: mx.array([next(samples)]))
    mx.eval = lambda *args: None
    caches, calls = [], []
    def make_cache(model):
        cache = object()
        caches.append(cache)
        return cache
    def model(tokens, *, cache):
        calls.append((tokens.tolist(), cache))
        return mx.zeros((1, tokens.shape[1], 3))
    assert generate_completion(model, (0, 1), {2}, 8, mx=mx, make_cache=make_cache) == (1, 2)
    assert generate_completion(model, (1,), {2}, 3, mx=mx, make_cache=make_cache) == (0, 0, 0)
    assert len(calls) == 5 and len(caches) == 2 and caches[0] is not caches[1]
    assert calls[0][0] == [[0, 1]] and calls[1][0] == [[1]]


def fixture_grpo_publication(publication):
    from pathlib import Path
    from tuner.runtime.native_files import NativeTree
    model_root = Path(publication["model"].root)
    (model_root / "config.json").write_bytes(canonical({"num_hidden_layers": 2, "quantization": {"bits": 4}}))
    with NativeTree(model_root, private=True) as tree:
        publication["model"] = replace(publication["model"], files=tree.inventory())
    state = family.prepare(publication)
    training, source, profile, raw = inputs()
    profile = replace(profile, runtime_digest=hashlib.sha256(state.kwargs["runtime_capture"]).hexdigest())
    projection = compile_native_grpo(training, source, profile)
    out = publication["output"]
    initial, final = (out / "initial-adapters.safetensors").read_bytes(), (out / "adapters.safetensors").read_bytes()
    sft_result = strict_json((out / "native_result.json").read_bytes())
    binding = {"schema": "synaptic-native-grpo-input-binding/v1", "source": source.to_dict(),
        "tokenizer_snapshot_digest": publication["tokenizer"].digest, "tokenization": projection.tokenization.to_dict(),
        "prompt_tokens": [[1, 2], [2, 1]], "workload_digest": projection.workload.fingerprint,
        "runtime_digest": profile.runtime_digest}
    evidence = [{"step": step, "prompt_indices": [0, 1], "completions": [[1], [2], [1], [2]],
        "texts": ["red", "blue", "red", "blue"], "rewards": [0., 1., 0., 1.],
        "old": [[-1.], [-1.], [-1.], [-1.]], "reference": [[-1.], [-1.], [-1.], [-1.]]} for step in (1, 2, 3)]
    result = {"schema": "synaptic-native-mlx-grpo-result/v1", "status": "CANDIDATE_WORKER_COMPLETE",
        "input_binding_digest": hashlib.sha256(canonical(binding)).hexdigest(), "workload_digest": projection.workload.fingerprint,
        "runtime_digest": profile.runtime_digest, "model_snapshot_digest": publication["model"].digest,
        "completed_updates": 3, "adapter_format": "mlx-lm-lora-safetensors", "parameter_delta": sft_result["parameter_delta"],
        "final_validation": {"phase": "reloaded_saved_final_adapter_rollout_replay",
            "adapter_sha256": hashlib.sha256(final).hexdigest(), "loss": -.1},
        "metrics": [{"step": s, "loss": -.1, "gradient_norm": 1.} for s in (2, 3)],
        "evidence_sha256": hashlib.sha256(canonical(evidence)).hexdigest()}
    (out / "0000003_adapters.safetensors").unlink()  # fixture owns this file
    for name, value in (("input_binding.json", binding), ("rollout_evidence.json", evidence), ("native_result.json", result)):
        (out / name).write_bytes(canonical(value))
    (out / "workload.json").write_bytes(projection.workload.canonical_bytes)
    (state.root / "dataset.jsonl").write_bytes(raw)
    request = strict_json(state.request)
    request["workload"] = projection.workload.document
    state.request = canonical(request)
    (state.root / "request.json").write_bytes(state.request)
    state.training = training
    state.resolved = replace(state.resolved, workload_digest=projection.workload.fingerprint,
        resolved_config_digest=hashlib.sha256(training.canonical_json().encode()).hexdigest(),
        artifact_policy_digest=family._domain("synaptic-coordinator-artifact-policy/v1", training.to_dict()["artifacts"]))
    state.kwargs.update(workload=projection.workload.canonical_bytes, resolved=state.resolved)
    publication["workload"] = projection.workload.canonical_bytes
    assert initial != final
    return state, projection, binding, evidence


def test_public_grpo_facades_verify_stream_and_retain_no_replay(publication, monkeypatch):
    state, projection, _, _ = fixture_grpo_publication(publication)
    candidate = NativePlanningV1(**state.kwargs)
    assert not candidate.preflight(candidate.plan).ready
    composition, recompose, planning, calls, _ = family.compose(state, publication, monkeypatch)
    api = composition.api()
    plan = api.training.plan(api.training.resolve(api.training.load(state.training.canonical_json())), planning.provider)
    ready = api.training.preflight(plan)
    started = api.training.start(plan, ready)
    assert started.accepted and api.training.start(plan, ready) == started and len(calls) == 1
    assert api.runs.outcome(started.run).state.value == "succeeded"
    assert api.runs.verify(started.run).verified
    for role in training_roles(projection):
        assert b"".join(api.runs.artifacts(RunArtifactRequest(started.run, role, 1024 * 1024)).iter_bytes())
    assert recompose().api().runs.reverify(started.run).verified and len(calls) == 1
    with pytest.raises(RunOperationError):
        api.runs.logs(RunLogsRequest(started.run, limit=1, maximum_bytes=4096))


def test_uncertain_grpo_claim_cannot_be_replayed(publication, monkeypatch):
    state, _, _, _ = fixture_grpo_publication(publication)
    composition, _, planning, calls, _ = family.compose(state, publication, monkeypatch, fail=True)
    api = composition.api()
    plan = api.training.plan(api.training.resolve(api.training.load(state.training.canonical_json())), planning.provider)
    ready = api.training.preflight(plan)
    for _ in range(2):
        try:
            api.training.start(plan, ready)
        except (RunOperationError, ValueError, RuntimeError):
            pass
    assert len(calls) == 1


def test_grpo_child_recompiles_and_revalidates_before_runtime(publication, monkeypatch):
    from tuner.runtime import native_mlx_child, native_mlx_material, native_mlx_grpo, native_mlx_publication
    state, projection, _, _ = fixture_grpo_publication(publication)
    events = []
    monkeypatch.setattr(native_mlx_material, "verify_native_material", lambda *a, **k: events.append("capture"))
    monkeypatch.setattr(native_mlx_publication, "tokenizer_publication_bytes", lambda *a, **k: b"fixture")
    monkeypatch.setattr(native_mlx_child, "_load_published_tokenizer", lambda *a, **k: events.append("tokenizer"))
    def execute(**kwargs):
        assert kwargs["projection"] == projection
        assert hashlib.sha256(kwargs["dataset"]).hexdigest() == projection.source.content_digest
        events.append("worker")
        return b"cpu-double-only"
    monkeypatch.setattr(native_mlx_grpo, "execute_native_grpo", execute)
    request = strict_json(state.request)
    assert native_mlx_child.run_request(request, request_root=state.root) == b"cpu-double-only"
    assert events == ["capture", "tokenizer", "worker"]
    request["workload"]["controls"]["beta"] = .9
    with pytest.raises(ValueError, match="compilation"):
        native_mlx_child.run_request(request, request_root=state.root)
    assert len(events) == 3


def training_roles(projection):
    return projection.workload.document["training_input"]["artifacts"]["required_kinds"]


@pytest.mark.parametrize("mutation", ["constant", "missing_group", "bad_reward", "bad_mask", "overlong", "nan", "unchanged", "bool_index"])
def test_artifacts_refuse_inconclusive_or_modified_evidence(publication, mutation):
    _, projection, binding, evidence = fixture_grpo_publication(publication)
    if mutation == "unchanged":
        (publication["output"] / "adapters.safetensors").write_bytes(
            (publication["output"] / "initial-adapters.safetensors").read_bytes())
    else:
        if mutation == "constant":
            for row in evidence:
                row["texts"], row["rewards"] = ["red"] * 4, [0.] * 4
        elif mutation == "missing_group":
            evidence[0]["completions"].pop()
        elif mutation == "bad_reward":
            evidence[0]["rewards"][0] = 1.
        elif mutation == "bad_mask":
            evidence[0]["old"][0].append(-1.)
        elif mutation == "overlong":
            evidence[0]["completions"][0] = [1] * 9
        elif mutation == "nan":
            evidence[0]["old"][0][0] = float("inf")
        elif mutation == "bool_index":
            evidence[0]["prompt_indices"][1] = True
        with pytest.raises(ValueError):
            verify_rollout_evidence(evidence, binding, projection)
        if mutation == "nan":
            return
        (publication["output"] / "rollout_evidence.json").write_bytes(canonical(evidence))
    with pytest.raises(ValueError):
        verify_native_publication(**publication)


def test_reward_variation_must_survive_actual_float32_objective(publication):
    _, projection, binding, evidence = fixture_grpo_publication(publication)
    document = projection.workload.document["training_input"]
    document["rewards"]["documents"]["blue"]["scoring"].update(on_pass=1e-100)
    projection = compile_native_grpo(TrainingInputV1.from_dict(document), projection.source, projection.profile)
    for row in evidence:
        row["rewards"] = [0., 1e-100, 0., 1e-100]
    with pytest.raises(ValueError, match="inconclusive"):
        verify_rollout_evidence(evidence, binding, projection)
