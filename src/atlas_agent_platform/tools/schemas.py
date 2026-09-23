from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)


class ToolDefinition(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$",
    )
    description: str = Field(
        min_length=1,
        max_length=1024,
    )
    input_schema: dict[str, JsonValue]

class ToolCall(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    id: str = Field(min_length=1)
    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$",
    )
    arguments: dict[str, JsonValue] = Field(
        default_factory=dict
    )

class ToolCallBatch(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    calls: list[ToolCall] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_call_ids(self) -> Self:
        call_ids = [call.id for call in self.calls]

        if len(call_ids) != len(set(call_ids)):
            raise ValueError(
                "Tool call IDs must be unique within a batch."
            )

        return self

class ToolResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    tool_call_id: str = Field(min_length=1)
    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$",
    )
    output: JsonValue
    is_error: bool = False