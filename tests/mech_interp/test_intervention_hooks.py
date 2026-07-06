"""CPU tests for the intervention hook math."""

import numpy as np
import pytest
import torch

from MechInterp.intervention.hooks import (
    InterventionHook,
    additive_push,
    erase_and_write,
    resolve_final_positions,
)


def _unit(v):
    v = torch.tensor(v, dtype=torch.float32)
    return v / v.norm()


def test_resolve_final_positions_right_padding():
    # rows padded on the RIGHT: real tokens first, then zeros.
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
    pos = resolve_final_positions(3, 5, attention_mask=mask)
    assert pos.tolist() == [2, 4, 0]


def test_resolve_final_positions_left_padding():
    # rows padded on the LEFT: zeros first, then real tokens.
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1]])
    pos = resolve_final_positions(3, 5, attention_mask=mask)
    assert pos.tolist() == [4, 4, 4]


def test_resolve_final_positions_explicit_wraps():
    pos = resolve_final_positions(2, 5, explicit=torch.tensor([-1, 2]))
    assert pos.tolist() == [4, 2]


def test_resolve_final_positions_fallback_last_column():
    pos = resolve_final_positions(3, 7)
    assert pos.tolist() == [6, 6, 6]


def test_additive_push_final_position_per_row():
    d = _unit([1.0, 0.0, 0.0])
    hidden = torch.zeros(2, 4, 3)
    # row 0 last real token = col 1 (right padded), row 1 = col 3
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    final_pos = resolve_final_positions(2, 4, attention_mask=mask)
    alpha = torch.tensor([2.0, -3.0])
    out = additive_push(hidden.clone(), d, alpha, torch.zeros(4, dtype=torch.bool), True, final_pos)
    assert out[0, 1, 0].item() == pytest.approx(2.0)
    assert out[1, 3, 0].item() == pytest.approx(-3.0)
    # untouched columns stay zero
    assert out[0, 0, 0].item() == 0.0
    assert out[1, 0, 0].item() == 0.0


def test_additive_push_zero_alpha_is_noop():
    d = _unit([0.0, 1.0, 0.0])
    hidden = torch.randn(2, 3, 3)
    final_pos = torch.tensor([2, 2])
    alpha = torch.tensor([0.0, 0.0])
    out = additive_push(hidden.clone(), d, alpha, torch.zeros(3, dtype=torch.bool), True, final_pos)
    assert torch.allclose(out, hidden)


def test_erase_and_write_lands_at_setpoint_exactly():
    # direction along a non-axis unit vector; post-write projection == setpoint.
    d = _unit([1.0, 2.0, -1.0, 0.5])
    hidden = torch.randn(3, 5, 4).double().float()
    final_pos = torch.tensor([4, 4, 4])
    gain = torch.tensor([2.0, 2.0, 2.0])
    sigma = 3.0
    setpoint = gain * sigma
    out = erase_and_write(
        hidden.clone(), d, setpoint, gain, torch.zeros(5, dtype=torch.bool), True, final_pos
    )
    for b in range(3):
        proj = float(out[b, 4, :].double() @ d.double())
        assert proj == pytest.approx(6.0, abs=1e-4)


def test_erase_and_write_preserves_orthogonal_complement():
    d = _unit([1.0, 0.0, 0.0, 0.0])
    hidden = torch.randn(2, 3, 4)
    final_pos = torch.tensor([2, 2])
    gain = torch.tensor([1.0, 1.0])
    sigma = 5.0
    before = hidden.clone()
    out = erase_and_write(
        hidden.clone(), d, gain * sigma, gain, torch.zeros(3, dtype=torch.bool), True, final_pos
    )
    # the orthogonal complement (dims 1..3) at the edited token is unchanged
    for b in range(2):
        assert torch.allclose(out[b, 2, 1:], before[b, 2, 1:], atol=1e-5)
        # the on-direction coordinate (dim 0) equals the setpoint
        assert out[b, 2, 0].item() == pytest.approx(5.0, abs=1e-4)


def test_erase_and_write_inactive_rows_untouched():
    d = _unit([1.0, 1.0, 0.0])
    hidden = torch.randn(3, 2, 3)
    final_pos = torch.tensor([1, 1, 1])
    gain = torch.tensor([2.0, 0.0, 2.0])  # row 1 inactive
    before = hidden.clone()
    out = erase_and_write(
        hidden.clone(), d, gain * 4.0, gain, torch.zeros(2, dtype=torch.bool), True, final_pos
    )
    assert torch.allclose(out[1], before[1])
    assert not torch.allclose(out[0], before[0])


def test_hook_tuple_output_reattaches_rest():
    d = _unit([1.0, 0.0])
    hook = InterventionHook("additive", d, strength=1.0, position="final")
    hidden = torch.zeros(1, 2, 2)
    extra = torch.tensor([7.0])
    out = hook(None, None, (hidden, extra))
    assert isinstance(out, tuple)
    assert torch.equal(out[1], extra)


def test_hook_readback_reports_setpoint_for_erase_write():
    d = _unit([1.0, 0.0, 0.0])
    hook = InterventionHook(
        "erase_write", d, strength=2.0, sigma=3.0, position="final", measure_readback=True
    )
    hidden = torch.randn(2, 3, 3)
    mask = torch.tensor([[1, 1, 1], [1, 1, 1]])
    hook.attention_mask = mask
    hook(None, None, hidden)
    rb = hook.last_readback
    assert rb["commanded"] == pytest.approx([6.0, 6.0])
    assert rb["measured"] == pytest.approx([6.0, 6.0], abs=1e-3)


def test_hook_anchor_position_edits_single_shared_column():
    d = _unit([1.0, 0.0])
    hook = InterventionHook("additive", d, strength=1.0, position="anchor")
    hidden = torch.zeros(2, 4, 2)
    out = hook(None, None, hidden.clone())
    # anchor default = last column
    assert out[0, 3, 0].item() == pytest.approx(1.0)
    assert out[0, 0, 0].item() == 0.0


def test_additive_push_multi_column_shared_mask_per_row():
    # regression: a shared-column additive edit over MORE than one column (the
    # anchor_onward / answer_window case) must broadcast the per-row strength
    # across every masked column without a shape mismatch.
    d = _unit([1.0, 0.0, 0.0])
    hidden = torch.zeros(2, 4, 3)
    cols = torch.zeros(4, dtype=torch.bool)
    cols[1:] = True  # columns 1, 2, 3
    alpha = torch.tensor([2.0, -1.0])
    out = additive_push(hidden.clone(), d, alpha, cols, False, None)
    assert out[0, 0, 0].item() == 0.0
    assert all(out[0, c, 0].item() == pytest.approx(2.0) for c in (1, 2, 3))
    assert all(out[1, c, 0].item() == pytest.approx(-1.0) for c in (1, 2, 3))


def test_hook_anchor_onward_additive_over_full_sequence():
    d = _unit([1.0, 0.0])
    hook = InterventionHook("additive", d, strength=1.5, position="anchor_onward")
    hidden = torch.zeros(1, 3, 2)
    out = hook(None, None, hidden.clone())
    assert all(out[0, c, 0].item() == pytest.approx(1.5) for c in range(3))


def test_answer_window_excludes_prompt():
    # window_start = first generated token, so prompt columns are not steered.
    d = _unit([1.0, 0.0])
    hook = InterventionHook("additive", d, strength=2.0, position="answer_window")
    hook.window_start = 3  # prompt is columns 0..2, generated tokens start at 3
    hidden = torch.zeros(1, 5, 2)
    out = hook(None, None, hidden.clone())
    # prompt columns untouched
    assert all(out[0, c, 0].item() == 0.0 for c in (0, 1, 2))
    # answer columns steered
    assert all(out[0, c, 0].item() == pytest.approx(2.0) for c in (3, 4))


def test_answer_window_requires_window_start():
    d = _unit([1.0, 0.0])
    hook = InterventionHook("additive", d, strength=1.0, position="answer_window")
    hidden = torch.zeros(1, 4, 2)
    # window_start defaults to None -> must raise, never silently steer the prompt
    with pytest.raises(ValueError):
        hook(None, None, hidden)


def test_erase_and_write_active_override_forces_zero_gain_row():
    # row 1 has gain 0.0 (would be skipped by default) but active_override
    # forces it active: the ablate case, erase the projection and write zero.
    d = _unit([1.0, 0.0, 0.0])
    hidden = torch.tensor(
        [[10.0, 1.0, 1.0], [10.0, 1.0, 1.0], [10.0, 1.0, 1.0]]
    ).unsqueeze(1)  # (3, 1, 3)
    final_pos = torch.tensor([0, 0, 0])
    gain = torch.tensor([0.0, 0.0, 0.0])
    override = torch.tensor([False, True, False])
    out = erase_and_write(
        hidden.clone(), d, gain, gain, torch.zeros(1, dtype=torch.bool), True,
        final_pos, active_override=override,
    )
    # row 0 (not overridden, gain 0): untouched
    assert out[0, 0, 0].item() == pytest.approx(10.0)
    # row 1 (overridden, gain 0): projection erased to exactly 0, not skipped
    assert out[1, 0, 0].item() == pytest.approx(0.0, abs=1e-6)
    # row 2 (not overridden): untouched
    assert out[2, 0, 0].item() == pytest.approx(10.0)


def test_erase_and_write_active_override_can_deactivate_nonzero_gain():
    # active_override is a full replacement, not just an "also active" union:
    # a nonzero-gain row can be explicitly excluded too.
    d = _unit([1.0, 0.0])
    hidden = torch.tensor([[5.0, 1.0]]).unsqueeze(1)
    final_pos = torch.tensor([0])
    gain = torch.tensor([3.0])
    override = torch.tensor([False])
    out = erase_and_write(
        hidden.clone(), d, gain * 2.0, gain, torch.zeros(1, dtype=torch.bool), True,
        final_pos, active_override=override,
    )
    assert out[0, 0, 0].item() == pytest.approx(5.0)


def test_erase_and_write_active_override_never_writes_nan_gain():
    d = _unit([1.0, 0.0])
    hidden = torch.tensor([[5.0, 1.0]]).unsqueeze(1)
    final_pos = torch.tensor([0])
    gain = torch.tensor([float("nan")])
    override = torch.tensor([True])  # force active, but gain is NaN
    out = erase_and_write(
        hidden.clone(), d, gain, gain, torch.zeros(1, dtype=torch.bool), True,
        final_pos, active_override=override,
    )
    # NaN is excluded even under an override: row stays untouched, not NaN-corrupted
    assert out[0, 0, 0].item() == pytest.approx(5.0)


def test_hook_force_active_writes_zero_setpoint_for_erase_write():
    # the ablate case at the hook level: strength 0, force_active True.
    d = _unit([1.0, 0.0, 0.0])
    hook = InterventionHook(
        "erase_write", d, strength=0.0, sigma=3.0, position="final",
        measure_readback=True, force_active=True,
    )
    hidden = torch.tensor([[8.0, 2.0, 2.0], [8.0, 2.0, 2.0]]).unsqueeze(1)
    mask = torch.tensor([[1], [1]])
    hook.attention_mask = mask
    out = hook(None, None, hidden.clone())
    # projection along d is erased to 0 (the setpoint), not left at 8.0
    assert out[0, 0, 0].item() == pytest.approx(0.0, abs=1e-4)
    rb = hook.last_readback
    assert rb["commanded"] == pytest.approx([0.0, 0.0])
    assert rb["measured"] == pytest.approx([0.0, 0.0], abs=1e-3)


def test_hook_without_force_active_strength_zero_is_true_noop():
    # regression: default behavior (baseline) is unaffected by the new knob.
    d = _unit([1.0, 0.0, 0.0])
    hook = InterventionHook(
        "erase_write", d, strength=0.0, sigma=3.0, position="final",
        measure_readback=True,
    )
    hidden = torch.tensor([[8.0, 2.0, 2.0]]).unsqueeze(1)
    hook.attention_mask = torch.tensor([[1]])
    out = hook(None, None, hidden.clone())
    assert out[0, 0, 0].item() == pytest.approx(8.0)
    assert hook.last_readback["active_rows"] == []


def test_strength_length_mismatch_raises():
    d = _unit([1.0, 0.0])
    hook = InterventionHook("additive", d, strength=[1.0, 2.0, 3.0], position="final")
    hidden = torch.zeros(2, 2, 2)
    hook.attention_mask = torch.ones(2, 2, dtype=torch.long)
    with pytest.raises(ValueError):
        hook(None, None, hidden)
