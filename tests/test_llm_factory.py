import pytest

from atlas_agent_platform.core.config import Settings
from atlas_agent_platform.llm.factory import create_llm_provider
from atlas_agent_platform.llm.providers.mock import MockLLMProvider


def test_factory_creates_mock_provider() -> None:
    settings = Settings(
        llm_provider="mock",
        llm_model="factory-test-model",
    )

    provider = create_llm_provider(settings)

    assert isinstance(provider, MockLLMProvider)
    assert provider.provider_name == "mock"
    assert provider.model_name == "factory-test-model"


def test_factory_rejects_unimplemented_provider() -> None:
    settings = Settings(llm_provider="openai")

    with pytest.raises(NotImplementedError, match="openai"):
        create_llm_provider(settings)