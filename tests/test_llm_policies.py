from atlas_agent_platform.llm.capabilities import (
    ModelCapabilities,
    ModelProfile,
)
from atlas_agent_platform.llm.policies import apply_model_request_policy
from atlas_agent_platform.llm.schemas import ChatMessage, LLMRequest


def create_test_profile(
    max_output_tokens: int,
) -> ModelProfile:
    return ModelProfile(
        provider="test-provider",
        model_name="test-model",
        capabilities=ModelCapabilities(
            max_output_tokens=max_output_tokens,
        ),
    )

def test_policy_keeps_request_below_model_limit() -> None:
    request = LLMRequest(
        messages=[
            ChatMessage(role="user", content="Hello"),
        ],
        max_output_tokens=100,
    )
    profile = create_test_profile(max_output_tokens=200)

    result = apply_model_request_policy(request, profile)

    assert result.request is request
    assert result.request.max_output_tokens == 100
    assert result.warnings == ()

def test_policy_keeps_request_equal_to_model_limit() -> None:
    request = LLMRequest(
        messages=[
            ChatMessage(role="user", content="Hello"),
        ],
        max_output_tokens=200,
    )
    profile = create_test_profile(max_output_tokens=200)

    result = apply_model_request_policy(request, profile)

    assert result.request is request
    assert result.request.max_output_tokens == 200
    assert result.warnings == ()

def test_policy_reduces_request_above_model_limit() -> None:
    request = LLMRequest(
        messages=[
            ChatMessage(role="user", content="Hello"),
        ],
        max_output_tokens=300,
    )
    profile = create_test_profile(max_output_tokens=200)

    result = apply_model_request_policy(request, profile)

    assert result.request is not request
    assert request.max_output_tokens == 300
    assert result.request.max_output_tokens == 200
    assert result.warnings == (
        (
            "max_output_tokens was reduced from 300 to 200 "
            "for test-provider/test-model."
        ),
    )