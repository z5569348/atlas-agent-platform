import pytest

from atlas_agent_platform.llm.capabilities import (
    ModelCapabilities,
    ModelProfile,
)
from atlas_agent_platform.llm.exceptions import LLMInvalidRequestError
from atlas_agent_platform.llm.policies import apply_model_request_policy
from atlas_agent_platform.llm.schemas import ChatMessage, LLMRequest
from atlas_agent_platform.tools.schemas import ToolDefinition


def create_test_profile(
    max_output_tokens: int,
    *,
    supports_tool_calling: bool = False,
) -> ModelProfile:
    return ModelProfile(
        provider="test-provider",
        model_name="test-model",
        capabilities=ModelCapabilities(
            supports_tool_calling=supports_tool_calling,
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

def test_policy_rejects_tools_for_unsupported_model() -> None:
    request = LLMRequest(
        messages=[
            ChatMessage(role="user", content="Calculate 6 times 7."),
        ],
        tools=[
            ToolDefinition(
                name="calculator",
                description="Perform arithmetic calculations.",
                input_schema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ],
    )
    profile = create_test_profile(
        max_output_tokens=200,
        supports_tool_calling=False,
    )

    with pytest.raises(
        LLMInvalidRequestError,
        match="does not support tool calling",
    ) as error_info:
        apply_model_request_policy(request, profile)

    assert error_info.value.provider == "test-provider"
    assert error_info.value.code == "llm_invalid_request"
    assert error_info.value.retryable is False

def test_policy_keeps_tools_for_supported_model() -> None:
    tool = ToolDefinition(
        name="calculator",
        description="Perform arithmetic calculations.",
        input_schema={
            "type": "object",
            "properties": {},
        },
    )
    request = LLMRequest(
        messages=[
            ChatMessage(role="user", content="Calculate 6 times 7."),
        ],
        tools=[tool],
        max_output_tokens=100,
    )
    profile = create_test_profile(
        max_output_tokens=200,
        supports_tool_calling=True,
    )

    result = apply_model_request_policy(request, profile)

    assert result.request is request
    assert result.request.tools == [tool]
    assert result.warnings == ()