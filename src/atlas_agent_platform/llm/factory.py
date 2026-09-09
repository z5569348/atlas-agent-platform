from functools import lru_cache

from atlas_agent_platform.core.config import Settings, get_settings
from atlas_agent_platform.llm.providers.base import LLMProvider
from atlas_agent_platform.llm.providers.mock import MockLLMProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "mock":
        return MockLLMProvider(model_name=settings.llm_model)

    raise NotImplementedError(
        f"LLM provider {settings.llm_provider!r} is not implemented yet."
    )


@lru_cache
def get_llm_provider() -> LLMProvider:
    return create_llm_provider(get_settings())