import pytest
from pydantic import JsonValue, ValidationError

from atlas_agent_platform.tools.builtin.calculator import (
    CalculatorTool,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"

def test_calculator_tool_builds_definition() -> None:
    tool = CalculatorTool()
    definition = tool.definition

    assert definition.name == "calculator"
    assert definition.input_schema["type"] == "object"
    assert (
        definition.input_schema["additionalProperties"]
        is False
    )

    properties = definition.input_schema["properties"]
    assert isinstance(properties, dict)

    operation_schema = properties["operation"]
    assert isinstance(operation_schema, dict)
    assert operation_schema["enum"] == [
        "add",
        "subtract",
        "multiply",
        "divide",
    ]

@pytest.mark.parametrize(
    (
        "operation",
        "left",
        "right",
        "expected",
    ),
    [
        ("add", 2.0, 3.0, 5.0),
        ("subtract", 7.0, 2.0, 5.0),
        ("multiply", 6.0, 7.0, 42.0),
        ("divide", 8.0, 2.0, 4.0),
    ],
)
@pytest.mark.anyio
async def test_calculator_tool_executes_operation(
    operation: str,
    left: float,
    right: float,
    expected: float,
) -> None:
    tool = CalculatorTool()
    arguments: dict[str, JsonValue] = {
        "operation": operation,
        "left": left,
        "right": right,
    }

    result = await tool.invoke(arguments)

    assert result == {"result": expected}


@pytest.mark.anyio
async def test_calculator_tool_rejects_invalid_operation() -> None:
    tool = CalculatorTool()
    arguments: dict[str, JsonValue] = {
        "operation": "power",
        "left": 2,
        "right": 3,
    }

    with pytest.raises(ValidationError):
        await tool.invoke(arguments)

@pytest.mark.anyio
async def test_calculator_tool_rejects_division_by_zero() -> None:
    tool = CalculatorTool()
    arguments: dict[str, JsonValue] = {
        "operation": "divide",
        "left": 10,
        "right": 0,
    }

    with pytest.raises(
        ValueError,
        match="Cannot divide by zero.",
    ):
        await tool.invoke(arguments)