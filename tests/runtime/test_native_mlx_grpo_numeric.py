"""Opt-in real MLX math checks for a separately admitted Apple runtime.

Never import MLX on Linux or in ordinary Mac CI. Root must review source/runtime
and authorize this exact bounded command before enabling the environment switch.
No model, network, training, filesystem outputs or provider operations.
"""
import math
import os
import platform

import pytest

from tuner.runtime.native_mlx_grpo_math import grpo_objective, group_advantages


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64"
    or os.environ.get("SYNAPTIC_TEST_NATIVE_GRPO_MLX") != "1",
    reason="requires separately admitted real Apple MLX numerical check",
)
def test_real_mlx_objective_gradients_and_detached_fixed_inputs():
    import mlx.core as mx

    policy = mx.log(mx.array([[.2, .5, .3], [.4, .25, .1]], dtype=mx.float32))
    old = mx.log(mx.array([[.25, .4, .3], [.5, .2, .1]], dtype=mx.float32))
    reference = mx.log(mx.array([[.15, .6, .3], [.35, .3, .1]], dtype=mx.float32))
    advantages = group_advantages(mx.array([1., 3.]), 2, mx=mx)
    mask = mx.array([[1., 1., 0.], [1., 0., 0.]])
    def objective(p, o, r, a):
        return grpo_objective(p, o, r, a, mask, .2, mx=mx)
    observed = objective(policy, old, reference, advantages)
    gradients = mx.grad(objective, argnums=(0, 1, 2, 3))(policy, old, reference, advantages)
    mx.eval(observed, gradients)
    expected_loss, expected_gradient = 0., [[0.] * 3 for _ in range(2)]
    for i, j, p, q, r, advantage in ((0, 0, .2, .25, .15, -1), (0, 1, .5, .4, .6, -1),
                                     (1, 0, .4, .5, .35, 1)):
        advantage /= math.sqrt(1 + 1e-8)
        expected_loss += (-p / q * advantage + .2 * (r / p - math.log(r / p) - 1)) / 3
        expected_gradient[i][j] = (-p / q * advantage + .2 * (1 - r / p)) / 3
    assert float(observed.item()) == pytest.approx(expected_loss, abs=2e-6)
    for actual, expected in zip(gradients[0].tolist(), expected_gradient):
        assert actual == pytest.approx(expected, abs=2e-6)
    for frozen in gradients[1:]:
        assert bool(mx.all(frozen == 0).item())
    reward_grad = mx.grad(lambda r: mx.sum(group_advantages(r, 2, mx=mx)))(mx.array([1., 3.]))
    assert bool(mx.all(reward_grad == 0).item())
