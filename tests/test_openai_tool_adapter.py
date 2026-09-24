import json

import pytest
from openai.types.responses import ResponseFunctionToolCall
from pydantic import ValidationError

from atlas_agent_platform.llm.providers.openai_tools import (
    from_openai_function_call,
    to_openai_function_tool,
)
from atlas_agent_platform.tools.builtin.calculator import (
    CalculatorTool,
)


def test_converts_tool_definition_to_openai_function() -> None:
    definition = CalculatorTool().definition

    openai_tool = to_openai_function_tool(definition)

    assert openai_tool == {
        "type": "function",
        "name": "calculator",
        "description": definition.description,
        "parameters": definition.input_schema,
        "strict": True,
        "async": False,
    }

def test_converts_openai_function_call_to_tool_call() -> None:
    openai_call = ResponseFunctionToolCall(
        arguments=(
            '{"operation":"multiply","left":6,"right":7}'
        ),
        call_id="call_123",
        name="calculator",
        type="function_call",
    )

    call = from_openai_function_call(openai_call)

    assert call.id == "call_123"
    assert call.name == "calculator"
    assert call.arguments == {
        "operation": "multiply",
        "left": 6,
        "right": 7,
    }

def test_rejects_invalid_openai_arguments_json() -> None:
    openai_call = ResponseFunctionToolCall(
        arguments="{invalid-json}",
        call_id="call_123",
        name="calculator",
        type="function_call",
    )

    with pytest.raises(json.JSONDecodeError):
        from_openai_function_call(openai_call)

def test_rejects_non_object_openai_arguments() -> None:
    openai_call = ResponseFunctionToolCall(
        arguments="[1, 2, 3]",
        call_id="call_123",
        name="calculator",
        type="function_call",
    )

    with pytest.raises(ValidationError):
        from_openai_function_call(openai_call)