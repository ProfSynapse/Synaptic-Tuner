"""Attributed mlxrl objective adaptation; no ML imports at module import time.

Source: inductiveML/mlxrl@5ee8a6bd7764b0585094d78c530ac87d26364034,
mlxrl/algo/grpo.py (group_normalize_rewards, GRPOAlgorithm.compute_loss,
token_importance_ratio, approximate_kl, _token_loss_metrics).
Changes: explicit array backend; one scalar result; detach all fixed inputs;
masked positions are cleared before exponentiation; no other algorithms.

MIT License
Copyright (c) 2026 mlxrl contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def group_advantages(rewards, group_size, *, mx):
    if len(rewards.shape) != 1 or group_size < 2 or rewards.size % group_size:
        raise ValueError("native GRPO reward grouping differs")
    grouped = mx.stop_gradient(rewards).reshape((-1, group_size)).astype(mx.float32)
    mean = mx.mean(grouped, axis=1, keepdims=True)
    variance = mx.mean(mx.square(grouped - mean), axis=1, keepdims=True)
    return mx.stop_gradient(((grouped - mean) / mx.sqrt(variance + 1e-8)).reshape((-1,)))


def grpo_objective(policy, old, reference, advantages, mask, beta, *, mx):
    if (len(policy.shape) != 2 or policy.shape != old.shape or policy.shape != reference.shape
            or policy.shape != mask.shape or advantages.shape != (policy.shape[0],)):
        raise ValueError("native GRPO objective shapes differ")
    # Zero ignored values before exp: even huge ignored padding must not poison
    # a finite loss/gradient through inf*0. Boundary admission requires a token.
    policy = mx.where(mask != 0, policy, 0)
    old = mx.where(mask != 0, mx.stop_gradient(old), 0)
    reference = mx.where(mask != 0, mx.stop_gradient(reference), 0)
    ratio = mx.exp(policy - old)
    terms = -ratio * mx.stop_gradient(advantages)[:, None]
    if beta != 0:
        log_ratio = reference - policy
        kl = mx.exp(log_ratio) - log_ratio - 1.0
        terms = terms + beta * kl
    return mx.sum(terms * mask) / mx.sum(mask)


def completion_logprobs(model, prompts, completions, *, mx):
    """Serial full causal forwards; gathered targets are sampled IDs, never text.

    The first generated target is predicted at len(prompt)-1. Right padding is
    added only to already gathered logprobs and contributes no loss/attention.
    This avoids stock-model fast paths and prefix-cache differentiation.
    """
    if len(prompts) != len(completions) or not prompts or any(
        not p or not c for p, c in zip(prompts, completions)
    ):
        raise ValueError("native GRPO token sequences are empty or misaligned")
    width = max(map(len, completions))
    rows, masks = [], []
    for prompt, completion in zip(prompts, completions):
        sequence = tuple(prompt) + tuple(completion)
        logits = model(mx.array([sequence[:-1]], dtype=mx.int32))[0].astype(mx.float32)
        logits = logits[len(prompt) - 1:len(prompt) - 1 + len(completion)]
        targets = mx.array(completion, dtype=mx.int32)
        selected = mx.take_along_axis(logits, targets[:, None], axis=-1)[:, 0]
        logprobs = selected - mx.logsumexp(logits, axis=-1)
        rows.append(mx.concatenate((logprobs, mx.zeros((width - len(completion),)))))
        masks.append([1.0] * len(completion) + [0.0] * (width - len(completion)))
    return mx.stack(rows), mx.array(masks, dtype=mx.float32)
