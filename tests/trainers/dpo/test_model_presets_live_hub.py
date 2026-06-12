"""Live HF-hub existence check for the pinned Qwen3 model presets.

TEST-phase finding (phase1-pipeline, focus item 9 + arch §5.3 VERIFY item).
The DPO/KTO model presets pin the hybrid Qwen3 bnb-4bit repos:

    qwen3_4b -> unsloth/Qwen3-4B-bnb-4bit
    qwen3_8b -> unsloth/Qwen3-8B-bnb-4bit

The original pins (`unsloth/Qwen3-{4B,8B}-Instruct-bnb-4bit`) were 404 on the
hub: a live huggingface_hub.HfApi().model_info() query (2026-06-11, real backend
Request IDs) returned RepositoryNotFoundError for both, so every real run would
have failed at model-download time. The user-ratified repoint moved both presets
to the hybrid models above (enable_thinking=False matches PROTOCOL v0.3's
'thinking mode OFF' literally), both verified 200.

This test pins those repos against the live hub so a future preset edit cannot
silently reintroduce a 404. It is NETWORK-GATED and SKIPPED by default (no
network in the normal suite, per the TEST-phase network boundary: item 9 is the
only network-permitted check). Run it explicitly when verifying the pins:

    RUN_LIVE_HUB=1 pytest tests/trainers/dpo/test_model_presets_live_hub.py

It must PASS now (the hybrid names resolve 200). Do not "fix" a future failure
by relaxing the existence check — the check is the point; repoint the presets to
a repo that actually exists.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_HUB") != "1",
    reason="network-gated; set RUN_LIVE_HUB=1 to verify pinned repo names exist",
)

PINNED_PRESET_REPOS = [
    "unsloth/Qwen3-4B-bnb-4bit",
    "unsloth/Qwen3-8B-bnb-4bit",
]


@pytest.mark.parametrize("repo_id", PINNED_PRESET_REPOS)
def test_pinned_preset_repo_exists_on_hub(repo_id):
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        info = api.model_info(repo_id)
    except Exception as exc:  # RepositoryNotFoundError / HTTP 404 / auth-wrapped 404
        pytest.fail(
            f"[{type(exc).__name__}] "
            f"Pinned model preset '{repo_id}' does NOT exist on the HF hub (404). "
            f"Phase-1 runs would fail at model download. Repoint the DPO/KTO "
            f"presets + the 7 recipe model.name fields to an existing repo "
            f"(e.g. unsloth/Qwen3-4B-Instruct-2507-bnb-4bit or "
            f"unsloth/Qwen3-4B-bnb-4bit), then update the sibling "
            f"test_model_presets.py assertions."
        )
    assert info.id == repo_id
