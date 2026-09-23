import pytest
from pydantic import ValidationError

from atlas_agent_platform.tools.schemas import (
    ToolCall,
    ToolCallBatch,
    ToolDefinition,
    ToolResult,
)


def test_tool_definition_accepts_json_schema() -> None:
    definition = ToolDefinition(
        name="get_weather",
        description="Get the current weather for a city.",
        input_schema={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                }
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    )

    assert definition.name == "get_weather"
    assert definition.input_schema["type"] == "object"

@pytest.mark.parametrize(
    "name",
    [
        "",
        "123tool",
        "tool name",
        "tool.name",
    ],
)
def test_tool_definition_rejects_invalid_name(
    name: str,
) -> None:
    with pytest.raises(ValidationError):
        ToolDefinition(
            name=name,
            description="Test tool.",
            input_schema={
                "type": "object",
                "properties": {},
            },
        )

def test_tool_call_batch_rejects_duplicate_ids() -> None:
    with pytest.raises(
        ValidationError,
        match="Tool call IDs must be unique within a batch.",
    ):
        ToolCallBatch(
            calls=[
                ToolCall(
                    id="call_1",
                    name="tool_a",
                    arguments={},
                ),
                ToolCall(
                    id="call_1",
                    name="tool_b",
                    arguments={},
                ),
            ]
        )

def test_tool_result_accepts_structured_output() -> None:
    result = ToolResult(
        tool_call_id="call_1",
        name="get_weather",
        output={
            "temperature": 22,
            "condition": "sunny",
            "alerts": [],
        },
    )

    assert result.tool_call_id == "call_1"
    assert result.output == {
        "temperature": 22,
        "condition": "sunny",
        "alerts": [],
    }
    assert result.is_error is False