"""Real MLX loss/autograd checks on the separately qualified Apple host."""

import math
import platform

import pytest

from shared.sft_preprocessing import PreparedSFTExample
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from tuner.runtime.native_mlx_sft import (
    MLXSFTConfig, make_mlx_sft_hooks, prepare_mlx_sft,
)


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="real MLX numeric qualification requires Apple silicon",
)
@pytest.mark.parametrize("labels", [
    (1, 2, 3, 4),                    # Full-sequence supervision.
    (-100, -100, 3, 4),             # Assistant target after masked prompt.
    (-100, 2, -100, 4),             # Noncontiguous supervision.
])
def test_real_mlx_loss_and_gradient_match_independent_scalar_reference(labels):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    row = PreparedSFTExample([1, 2, 3, 4], [1] * 4, list(labels),
                             "messages", "assistant_only", False)
    data = prepare_mlx_sft(
        source=PreparedTrainingInputIdentity(
            "prepared://sha256/" + "a" * 64, "a" * 64, "b" * 64, 10,
            "synaptic-conversation-sft-row/v1",
        ),
        tokenizer_revision="c" * 40,
        train=(("train", row),), validation=(("heldout", row),),
        config=MLXSFTConfig.from_dict(dict(
            batch_size=1, max_seq_length=8, max_batch_tokens=8, iterations=1,
            seed=17, steps_per_report=1, steps_per_eval=1, steps_per_save=1,
        )),
    )
    iterate, loss = make_mlx_sft_hooks(data, mx=mx, nn=nn)
    tokens, targets = next(iterate(data.validation, 1, 8))
    values = [[0.2, -0.7, 1.1, 0.4, -0.2],
              [-0.3, 0.8, 0.1, -0.4, 0.9],
              [1.0, -0.5, 0.6, -0.1, 0.3]]
    count = sum(label != -100 for label in labels[1:])
    reference_loss, reference_gradient = 0.0, []
    for vector, label in zip(values, labels[1:]):
        maximum = max(vector)
        partition = sum(math.exp(value - maximum) for value in vector)
        probabilities = [math.exp(value - maximum) / partition for value in vector]
        if label == -100:
            reference_gradient.append([0.0] * len(vector))
        else:
            reference_loss += maximum + math.log(partition) - vector[label]
            reference_gradient.append([
                (probability - (index == label)) / count
                for index, probability in enumerate(probabilities)
            ])

    def objective(logits):
        return loss(lambda inputs: logits, tokens, targets)[0]

    logits = mx.array([values], dtype=mx.float32)
    actual, gradient = mx.value_and_grad(objective)(logits)
    _, supervised = loss(lambda inputs: logits, tokens, targets)
    mx.eval(actual, gradient, supervised)
    assert int(supervised.item()) == count
    assert float(actual.item()) == pytest.approx(reference_loss / count, abs=2e-6)
    for observed, expected in zip(gradient.tolist()[0], reference_gradient):
        assert all(math.isfinite(value) for value in observed)
        assert observed == pytest.approx(expected, abs=2e-6)
