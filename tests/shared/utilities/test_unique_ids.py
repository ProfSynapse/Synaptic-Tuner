"""Tests for shared.utilities.unique_ids — pure timestamp+nonce helpers."""

import re
from datetime import datetime, timezone

from shared.utilities.unique_ids import unique_prefixed_id, unique_utc_timestamp


class TestUniqueUtcTimestamp:
    """Verify unique_utc_timestamp produces correctly formatted strings."""

    def test_default_format_matches_pattern(self):
        result = unique_utc_timestamp()
        # Expected: YYYYMMDD_HHMMSS_XXXX (4-char hex nonce)
        assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{4}", result), f"Unexpected format: {result}"

    def test_deterministic_with_fixed_now(self):
        fixed = datetime(2026, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        result = unique_utc_timestamp(now=fixed)
        assert result.startswith("20260115_103045_")
        assert len(result) == len("20260115_103045_") + 4

    def test_custom_format(self):
        fixed = datetime(2026, 3, 20, 8, 0, 0, tzinfo=timezone.utc)
        result = unique_utc_timestamp(now=fixed, fmt="%Y-%m-%d")
        assert result.startswith("2026-03-20_")

    def test_nonce_varies_across_calls(self):
        """Two calls at the same instant should (almost always) differ due to nonce."""
        fixed = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        results = {unique_utc_timestamp(now=fixed) for _ in range(20)}
        # With 16-bit nonce space and 20 draws, collisions are possible but >=2 unique is near-certain
        assert len(results) >= 2


class TestUniquePrefixedId:
    """Verify unique_prefixed_id prepends the prefix correctly."""

    def test_prefix_prepended(self):
        fixed = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = unique_prefixed_id("run_", now=fixed)
        assert result.startswith("run_20260601_120000_")

    def test_empty_prefix(self):
        result = unique_prefixed_id("")
        # Should just be the timestamp+nonce with no prefix
        assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{4}", result)

    def test_custom_format_forwarded(self):
        fixed = datetime(2026, 3, 20, 0, 0, 0, tzinfo=timezone.utc)
        result = unique_prefixed_id("exp-", now=fixed, fmt="%Y%m%d")
        assert result.startswith("exp-20260320_")
