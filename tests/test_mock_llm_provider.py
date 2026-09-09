import pytest

from atlas_agent_platform.llm.providers.base import LLMProvider
from atlas_agent_platform.llm.providers.mock import MockLLMProvider
from atlas_agent_platform.llm.schemas import ChatMessage, LLMRequest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"

def test_mock_provider_identity() -> None:
    provider: LLMProvider = MockLLMProvider(model_name="mock-model-v1")

    assert provider.provider_name == "mock"
    assert provider.model_name == "mock-model-v1"



@pytest.mark.anyio
async def test_mock_provider_generates_standard_response() -> None:
    provider = MockLLMProvider(model_name="mock-model-v1")
    request = LLMRequest(
        messages=[
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="What is RAG?"),
        ]
    )

    response = await provider.generate(request)

    assert response.provider == "mock"
    assert response.model == "mock-model-v1"
    assert response.content == "Mock response: What is RAG?"
    assert response.finish_reason == "stop"
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.latency_ms >= 0