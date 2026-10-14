"""Original submit read authority survives a lost cancellation receipt."""

from tests.training import test_modal_grpo_process_restart as process


def test_post_ack_orphaned_cancel_first_cold_reads_then_native_five(request):
    process.test_grpo_killed_producer_then_three_fresh_public_readers(
        request, cold_cancellation=True, receipt_gap=True, orphaned_reads=True
    )
