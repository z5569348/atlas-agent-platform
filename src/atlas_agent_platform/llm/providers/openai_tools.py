import json
from typing import cast

from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCall,
)
from pydantic import JsonValue, TypeAdapter

from atlas_agent_platform.tools.schemas import (
    ToolCall,
    ToolDefinition,
)

_ARGUMENTS_ADAPTER = TypeAdapter(
    dict[str, JsonValue]
)

def to_openai_function_tool(
    definition: ToolDefinition,
) -> FunctionToolParam:
    parameters = cast(
        dict[str, object],
        definition.input_schema,
    )

    return {
        "type": "function",
        "name": definition.name,
        "description": definition.description,
        "parameters": parameters,
        "strict": True,
        "async": False,
    }

def from_openai_function_call(
    call: ResponseFunctionToolCall,
) -> ToolCall:
    raw_arguments = json.loads(call.arguments)
    arguments = _ARGUMENTS_ADAPTER.validate_python(
        raw_arguments
    )

    return ToolCall(
        id=call.call_id,
        name=call.name,
        arguments=arguments,
    )