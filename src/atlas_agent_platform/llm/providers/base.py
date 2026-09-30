from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from atlas_agent_platform.llm.capabilities import ModelCapabilities
from atlas_agent_platform.llm.schemas import LLMRequest, LLMResponse
from atlas_agent_platform.tools.schemas import ToolResult


@dataclass(
    frozen=True,
    slots=True,
)
class LLMProviderTurn:
    response: LLMResponse
    state: object


class LLMProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def capabilities(self) -> ModelCapabilities: ...

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse: ...


@runtime_checkable
class AgentLLMProvider(LLMProvider, Protocol):
    async def start_turn(
        self,
        request: LLMRequest,
    ) -> LLMProviderTurn: ...

    async def continue_turn(
        self,
        turn: LLMProviderTurn,
        tool_results: Sequence[ToolResult],
    ) -> LLMProviderTurn: ...