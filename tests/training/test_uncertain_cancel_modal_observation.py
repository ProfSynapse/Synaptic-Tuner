"""Actual native GRPO completion can settle run truth after a lost cancel reply."""

from tests.training import test_modal_grpo_process_restart as process


def test_cold_modal_pending_and_native_completion_preserve_uncertain_cancel(request):
    process.test_grpo_killed_producer_then_three_fresh_public_readers(
        request, uncertain_cancel=True
    )
