from shared.experiment_tracking.runtime_autotune import AdaptiveTokenBudget, recommend_eval_max_workers


def test_adaptive_token_budget_grows_with_headroom_and_low_latency():
    controller = AdaptiveTokenBudget(initial_tokens=4096, max_tokens=16384)

    updated = controller.observe_success(
        padded_batch_tokens=4096,
        elapsed_seconds=1.0,
        headroom_bytes=3 * 1024 * 1024 * 1024,
    )

    assert updated > 4096


def test_adaptive_token_budget_shrinks_when_headroom_is_low():
    controller = AdaptiveTokenBudget(initial_tokens=8192, min_tokens=2048)

    updated = controller.observe_success(
        padded_batch_tokens=8192,
        elapsed_seconds=1.0,
        headroom_bytes=128 * 1024 * 1024,
    )

    assert updated < 8192


def test_recommend_eval_max_workers_caps_by_backend_shape():
    workers = recommend_eval_max_workers(backend="vllm", requested_max_workers=32, cpu_count=8)

    assert workers == 4
