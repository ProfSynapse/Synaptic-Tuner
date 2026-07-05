"""Tests for the remote-container env defaults in Trainers/cloud/train_modal.py.

Focus: apply_hf_xet_mitigation, which sets the hf_xet-hang workaround defaults
on the remote training container while still honoring an explicit override
forwarded from the local launch env.
"""

import pytest

# train_modal defines a Modal @app.function, so importing it needs the modal
# package; skip cleanly where modal is unavailable rather than erroring.
modal = pytest.importorskip("modal")

from Trainers.cloud.train_modal import (  # noqa: E402
    HF_XET_MITIGATION,
    apply_hf_xet_mitigation,
)


def test_mitigation_applied_when_unset():
    env = {}
    apply_hf_xet_mitigation(env)
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "0"


def test_mitigation_applied_when_empty_string():
    # The @app.function secrets dict forwards these as "" when they are absent
    # from the local env; an empty value must still fall through to the default.
    env = {"HF_HUB_DISABLE_XET": "", "HF_HUB_ENABLE_HF_TRANSFER": ""}
    apply_hf_xet_mitigation(env)
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "0"


def test_explicit_local_value_overrides_default():
    env = {"HF_HUB_DISABLE_XET": "0"}
    apply_hf_xet_mitigation(env)
    assert env["HF_HUB_DISABLE_XET"] == "0"  # explicit local override wins
    assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "0"  # unset -> default


def test_mitigation_keys_are_the_two_expected():
    assert set(HF_XET_MITIGATION) == {"HF_HUB_DISABLE_XET", "HF_HUB_ENABLE_HF_TRANSFER"}
