"""Actual configured producer death, cold cancellation, and native completion."""

from tests.training import test_modal_grpo_process_restart as process


def test_killed_producer_first_cold_cancel_then_lookup_and_native_reads(request):
    process.test_grpo_killed_producer_then_three_fresh_public_readers(
        request, cold_cancellation=True
    )


def test_post_ack_receipt_gap_never_replays_after_process_loss(request):
    process.test_grpo_killed_producer_then_three_fresh_public_readers(
        request, cold_cancellation=True, receipt_gap=True
    )
