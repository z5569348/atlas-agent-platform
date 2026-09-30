from functools import lru_cache

from atlas_agent_platform.agents.service import AgentService
from atlas_agent_platform.llm.factory import get_llm_provider
from atlas_agent_platform.llm.providers.base import AgentLLMProvider
from atlas_agent_platform.tools.builtin.calculator import CalculatorTool
from atlas_agent_platform.tools.registry import ToolRegistry


def create_agent_service(
    provider: AgentLLMProvider,
) -> AgentService:
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    return AgentService(
        provider=provider,
        registry=registry,
    )

@lru_cache
def get_agent_service() -> AgentService:
    provider = get_llm_provider()

    if not isinstance(provider, AgentLLMProvider):
        raise RuntimeError(
            f"LLM provider {provider.provider_name!r} "
            "does not support agent turns."
        )

    return create_agent_service(provider)