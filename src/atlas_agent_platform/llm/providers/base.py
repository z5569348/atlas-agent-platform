from typing import Protocol

from atlas_agent_platform.llm.capabilities import ModelCapabilities
from atlas_agent_platform.llm.schemas import LLMRequest, LLMResponse


class LLMProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def capabilities(self) -> ModelCapabilities: ...

    async def generate(self, request: LLMRequest) -> LLMResponse: ...
