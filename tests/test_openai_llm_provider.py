from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import RateLimitError
from openai.types.responses import ResponseFunctionToolCall
from pydantic import SecretStr

from atlas_agent_platform.llm.capabilities import ModelCapabilities
from atlas_agent_platform.llm.exceptions import (
    LLMQuotaExceededError,
)
from atlas_agent_platform.llm.providers.openai import (
    OpenAILLMProvider,
)
from atlas_agent_platform.llm.schemas import (
    ChatMessage,
    LLMRequest,
)
from atlas_agent_platform.tools.builtin.calculator import (
    CalculatorTool,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"

@pytest.fixture
def provider() -> OpenAILLMProvider:
    return OpenAILLMProvider(
        api_key=SecretStr("test-api-key"),
        model_name="test-model",
        capabilities=ModelCapabilities(
            supports_tool_calling=True,
            max_output_tokens=128_000,
        ),
    )

@pytest.mark.anyio
async def test_openai_provider_generates_response(
    provider: OpenAILLMProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_response = SimpleNamespace(
        model="test-model",
        output_text="Generated answer",
        output=[],
        incomplete_details=None,
        usage=SimpleNamespace(
            input_tokens=8,
            output_tokens=3,
        ),
    )
    create_response = AsyncMock(return_value=sdk_response)

    monkeypatch.setattr(
        provider._client.responses,
        "create",
        create_response,
    )

    request = LLMRequest(
        messages=[
            ChatMessage(role="user", content="Hello"),
        ],
        temperature=0.2,
        max_output_tokens=128,
    )

    result = await provider.generate(request)

    create_response.assert_awaited_once_with(
        model="test-model",
        input=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        max_output_tokens=128,
        store=False,
        stream=False,
    )

    assert result.provider == "openai"
    assert result.model == "test-model"
    assert result.content == "Generated answer"
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens == 8
    assert result.usage.output_tokens == 3
    assert result.latency_ms >= 0
    assert result.tool_calls == []

@pytest.mark.anyio
async def test_openai_provider_sends_supported_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAILLMProvider(
        api_key=SecretStr("test-api-key"),
        model_name="temperature-model",
        capabilities=ModelCapabilities(
            supports_temperature=True,
            max_output_tokens=128_000,
        ),
    )
    sdk_response = SimpleNamespace(
        model="temperature-model",
        output_text="Generated answer",
        output=[],
        incomplete_details=None,
        usage=SimpleNamespace(
            input_tokens=8,
            output_tokens=3,
        ),
    )
    create_response = AsyncMock(return_value=sdk_response)

    monkeypatch.setattr(
        provider._client.responses,
        "create",
        create_response,
    )

    request = LLMRequest(
        messages=[
            ChatMessage(role="user", content="Hello"),
        ],
        temperature=0.7,
        max_output_tokens=128,
    )

    await provider.generate(request)

    create_response.assert_awaited_once_with(
        model="temperature-model",
        input=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        max_output_tokens=128,
        temperature=0.7,
        store=False,
        stream=False,
    )

@pytest.mark.anyio
async def test_openai_provider_maps_quota_error(
    provider: OpenAILLMProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx2.Request(
        "POST",
        "https://api.openai.com/v1/responses",
    )
    response = httpx2.Response(
        status_code=429,
        request=request,
    )
    sdk_error = RateLimitError(
        "Quota exhausted.",
        response=response,
        body={
            "type": "insufficient_quota",
            "code": "credit_balance_exhausted",
        },
    )

    monkeypatch.setattr(
        provider._client.responses,
        "create",
        AsyncMock(side_effect=sdk_error),
    )

    llm_request = LLMRequest(
        messages=[
            ChatMessage(role="user", content="Hello"),
        ]
    )

    with pytest.raises(LLMQuotaExceededError) as error_info:
        await provider.generate(llm_request)

    error = error_info.value
    assert error.provider == "openai"
    assert error.code == "llm_quota_exceeded"
    assert error.retryable is False
    assert error.__cause__ is sdk_error

@pytest.mark.anyio
async def test_openai_provider_returns_tool_calls(
    provider: OpenAILLMProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = CalculatorTool().definition
    sdk_tool_call = ResponseFunctionToolCall(
        arguments='{"operation":"multiply","left":6,"right":7}',
        call_id="call_123",
        name="calculator",
        type="function_call",
    )
    sdk_response = SimpleNamespace(
        model="test-model",
        output_text="",
        output=[sdk_tool_call],
        incomplete_details=None,
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=8,
        ),
    )
    create_response = AsyncMock(return_value=sdk_response)

    monkeypatch.setattr(
        provider._client.responses,
        "create",
        create_response,
    )

    request = LLMRequest(
        messages=[
            ChatMessage(
                role="user",
                content="Calculate 6 times 7.",
            ),
        ],
        tools=[definition],
        max_output_tokens=128,
    )

    result = await provider.generate(request)

    create_response.assert_awaited_once_with(
        model="test-model",
        input=[
            {
                "role": "user",
                "content": "Calculate 6 times 7.",
            },
        ],
        max_output_tokens=128,
        store=False,
        stream=False,
        tools=[
            {
                "type": "function",
                "name": "calculator",
                "description": definition.description,
                "parameters": definition.input_schema,
                "strict": True,
                "async": False,
            },
        ],
    )

    assert result.content == ""
    assert result.finish_reason == "tool_call"
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert tool_call.id == "call_123"
    assert tool_call.name == "calculator"
    assert tool_call.arguments == {
        "operation": "multiply",
        "left": 6,
        "right": 7,
    }