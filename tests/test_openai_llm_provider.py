from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import RateLimitError
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


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"

@pytest.fixture
def provider() -> OpenAILLMProvider:
    return OpenAILLMProvider(
        api_key=SecretStr("test-api-key"),
        model_name="test-model",
        capabilities=ModelCapabilities(
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
    )

    assert result.provider == "openai"
    assert result.model == "test-model"
    assert result.content == "Generated answer"
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens == 8
    assert result.usage.output_tokens == 3
    assert result.latency_ms >= 0

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