from unittest.mock import AsyncMock

import pytest

from atlas_agent_platform.llm.capabilities import ModelCapabilities
from atlas_agent_platform.llm.providers.mock import MockLLMProvider
from atlas_agent_platform.llm.schemas import ChatMessage, LLMRequest
from atlas_agent_platform.llm.service import LLMGatewayService


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"

@pytest.mark.anyio
async def test_gateway_applies_model_request_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MockLLMProvider(
        model_name="limited-model",
        capabilities=ModelCapabilities(
            max_output_tokens=100,
        ),
    )
    generate = AsyncMock(wraps=provider.generate)
    monkeypatch.setattr(provider, "generate", generate)

    service = LLMGatewayService(provider)
    request = LLMRequest(
        messages=[
            ChatMessage(role="user", content="Hello"),
        ],
        max_output_tokens=300,
    )

    response = await service.generate(request)

    expected_request = request.model_copy(
        update={"max_output_tokens": 100}
    )
    generate.assert_awaited_once_with(expected_request)

    assert request.max_output_tokens == 300
    assert response.content == "Mock response: Hello"
    assert response.warnings == [
        (
            "max_output_tokens was reduced from 300 to 100 "
            "for mock/limited-model."
        )
    ]