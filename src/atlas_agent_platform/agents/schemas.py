from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from atlas_agent_platform.llm.schemas import (
    ChatMessage,
    LLMResponse,
    TokenUsage,
)
from atlas_agent_platform.tools.schemas import ToolResult


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
    )
    max_output_tokens: int = Field(
        default=1024,
        ge=1,
        le=128_000,
    )
    max_iterations: int = Field(
        default=8,
        ge=1,
        le=32,
    )

AgentRunStatus = Literal[
    "completed",
    "max_iterations_reached",
]


class AgentStep(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    iteration: int = Field(ge=1)
    llm_response: LLMResponse
    tool_results: tuple[ToolResult, ...] = ()


class AgentRunResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    run_id: UUID = Field(default_factory=uuid4)
    status: AgentRunStatus
    content: str
    steps: tuple[AgentStep, ...] = Field(min_length=1)
    usage: TokenUsage
    warnings: tuple[str, ...] = ()