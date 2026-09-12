from functools import lru_cache

from atlas_agent_platform.core.config import Settings, get_settings
from atlas_agent_platform.llm.providers.base import LLMProvider
from atlas_agent_platform.llm.providers.mock import MockLLMProvider
from atlas_agent_platform.llm.providers.openai import (
    OpenAILLMProvider,
)


def create_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "mock":
        return MockLLMProvider(model_name=settings.llm_model)

    if settings.llm_provider == "openai":
        if settings.llm_api_key is None:
            raise ValueError(
                "ATLAS_LLM_API_KEY is required for OpenAI."
            )

        base_url = (
            str(settings.llm_base_url)
            if settings.llm_base_url is not None
            else None
        )

        return OpenAILLMProvider(
            api_key=settings.llm_api_key,
            model_name=settings.llm_model,
            base_url=base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    raise NotImplementedError(
        f"LLM provider {settings.llm_provider!r} is not implemented yet."
    )


@lru_cache
def get_llm_provider() -> LLMProvider:
    return create_llm_provider(get_settings())