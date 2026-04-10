"""Shared fixtures for inference plugin tests."""

from __future__ import annotations

import pytest

from shared.inference.config import (
    DoLaConfig,
    InferencePluginConfig,
    MinPConfig,
    RepetitionPenaltyConfig,
)


@pytest.fixture
def default_config() -> InferencePluginConfig:
    """All plugins disabled (defaults)."""
    return InferencePluginConfig()


@pytest.fixture
def dola_config() -> DoLaConfig:
    """DoLa config with sane defaults for testing."""
    return DoLaConfig(
        enabled=True,
        premature_layers="high",
        relative_top=0.1,
        jsd_threshold=0.0,
    )


@pytest.fixture
def min_p_config() -> MinPConfig:
    """Min-P config for testing."""
    return MinPConfig(enabled=True, threshold=0.05)


@pytest.fixture
def repetition_config() -> RepetitionPenaltyConfig:
    """Repetition penalty config for testing."""
    return RepetitionPenaltyConfig(enabled=True, penalty=1.2, window=32)


@pytest.fixture
def factual_config() -> InferencePluginConfig:
    """Config with DoLa enabled (factual profile)."""
    return InferencePluginConfig(
        dola=DoLaConfig(enabled=True, premature_layers="high", relative_top=0.1),
    )


@pytest.fixture
def creative_config() -> InferencePluginConfig:
    """Config with Min-P enabled, DoLa disabled (creative profile)."""
    return InferencePluginConfig(
        min_p=MinPConfig(enabled=True, threshold=0.02),
    )
