import pytest
from pydantic import SecretStr

from atlas_agent_platform.core.config import Settings
from atlas_agent_platform.llm.factory import create_llm_provider
from atlas_agent_platform.llm.providers.mock import MockLLMProvider
from atlas_agent_platform.llm.providers.openai import (
    OpenAILLMProvider,
)


def test_factory_creates_mock_provider() -> None:
    settings = Settings(
        llm_provider="mock",
        llm_model="factory-test-model",
    )

    provider = create_llm_provider(settings)

    assert isinstance(provider, MockLLMProvider)
    assert provider.provider_name == "mock"
    assert provider.model_name == "factory-test-model"

def test_factory_creates_openai_provider() -> None:
    settings = Settings(
        llm_provider="openai",
        llm_model="factory-openai-model",
        llm_api_key=SecretStr("test-api-key"),
        llm_base_url=None,
    )

    provider = create_llm_provider(settings)

    assert isinstance(provider, OpenAILLMProvider)
    assert provider.provider_name == "openai"
    assert provider.model_name == "factory-openai-model"

def test_factory_rejects_unimplemented_provider() -> None:
    settings = Settings(llm_provider="qwen")

    with pytest.raises(NotImplementedError, match="qwen"):
        create_llm_provider(settings)

def test_factory_requires_openai_api_key() -> None:
    settings = Settings(
        llm_provider="openai",
        llm_api_key=None,
    )

    with pytest.raises(
        ValueError,
        match="ATLAS_LLM_API_KEY",
    ):
        create_llm_provider(settings)