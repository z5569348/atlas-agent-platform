import pytest
from pydantic import ValidationError

from atlas_agent_platform.llm.capabilities import (
    ModelCapabilities,
    ModelProfile,
    get_model_profile,
)


def test_get_registered_model_profile() -> None:
    profile = get_model_profile("openai", "gpt-5.6-luna")

    assert isinstance(profile, ModelProfile)
    assert profile.provider == "openai"
    assert profile.model_name == "gpt-5.6-luna"

def test_luna_model_capabilities() -> None:
    capabilities = get_model_profile(
        "openai",
        "gpt-5.6-luna",
    ).capabilities

    assert capabilities.supports_temperature is False
    assert capabilities.supports_tool_calling is True
    assert capabilities.supports_structured_output is True
    assert capabilities.supports_vision is True
    assert capabilities.supports_streaming is True
    assert capabilities.supports_reasoning is True
    assert capabilities.max_output_tokens == 128_000

def test_get_unregistered_model_profile_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Model profile is not registered: openai/unknown-model",
    ):
        get_model_profile("openai", "unknown-model")

def test_model_capabilities_are_immutable() -> None:
    capabilities = ModelCapabilities(
        max_output_tokens=128_000,
    )

    with pytest.raises(ValidationError, match="Instance is frozen"):
        capabilities.supports_streaming = False