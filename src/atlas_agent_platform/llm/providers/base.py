from typing import Protocol

from atlas_agent_platform.llm.schemas import LLMRequest, LLMResponse


class LLMProvider(Protocol):
    @property
    def provider_name(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...

    async def generate(self, request: LLMRequest) -> LLMResponse:
        ...