from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from atlas_agent_platform.tools.schemas import (
    ToolCall,
    ToolDefinition,
)

MessageRole = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal["stop", "length", "tool_call", "content_filter"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    role: MessageRole
    content: str = Field(min_length=1)


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1)
    tools: list[ToolDefinition] = Field(
        default_factory=list
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=1, le=32768)


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    content: str
    tool_calls: list[ToolCall] = Field(
        default_factory=list
    )
    finish_reason: FinishReason
    usage: TokenUsage
    latency_ms: float = Field(ge=0.0)
    warnings: list[str] = Field(default_factory=list)