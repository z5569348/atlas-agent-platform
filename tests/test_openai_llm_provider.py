from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

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
        ## temperature=0.2,
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