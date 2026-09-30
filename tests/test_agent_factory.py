import pytest
from pydantic import SecretStr

from atlas_agent_platform.agents import factory as agent_factory
from atlas_agent_platform.agents.service import AgentService
from atlas_agent_platform.llm.capabilities import ModelCapabilities
from atlas_agent_platform.llm.providers.mock import MockLLMProvider
from atlas_agent_platform.llm.providers.openai import OpenAILLMProvider


def test_get_agent_service_creates_and_caches_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAILLMProvider(
        api_key=SecretStr("test-api-key"),
        model_name="test-model",
        capabilities=ModelCapabilities(
            max_output_tokens=32_768,
            supports_tool_calling=True,
        ),
    )

    monkeypatch.setattr(
        agent_factory,
        "get_llm_provider",
        lambda: provider,
    )
    agent_factory.get_agent_service.cache_clear()

    try:
        first_service = agent_factory.get_agent_service()
        second_service = agent_factory.get_agent_service()

        assert isinstance(first_service, AgentService)
        assert second_service is first_service
    finally:
        agent_factory.get_agent_service.cache_clear()


def test_get_agent_service_rejects_single_turn_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MockLLMProvider(
        model_name="single-turn-model",
    )

    monkeypatch.setattr(
        agent_factory,
        "get_llm_provider",
        lambda: provider,
    )
    agent_factory.get_agent_service.cache_clear()

    try:
        with pytest.raises(
            RuntimeError,
            match="does not support agent turns",
        ):
            agent_factory.get_agent_service()
    finally:
        agent_factory.get_agent_service.cache_clear()
