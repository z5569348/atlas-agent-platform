import pytest

from atlas_agent_platform.tools.builtin.calculator import (
    CalculatorTool,
)
from atlas_agent_platform.tools.executor import ToolExecutor
from atlas_agent_platform.tools.registry import ToolRegistry
from atlas_agent_platform.tools.schemas import ToolCall


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return ToolExecutor(registry)

@pytest.mark.anyio
async def test_executor_returns_success_result(
    executor: ToolExecutor,
) -> None:
    call = ToolCall(
        id="call_1",
        name="calculator",
        arguments={
            "operation": "multiply",
            "left": 6,
            "right": 7,
        },
    )

    result = await executor.execute(call)

    assert result.tool_call_id == "call_1"
    assert result.name == "calculator"
    assert result.output == {"result": 42.0}
    assert result.is_error is False

@pytest.mark.anyio
async def test_executor_returns_unknown_tool_error(
    executor: ToolExecutor,
) -> None:
    call = ToolCall(
        id="call_2",
        name="unknown_tool",
        arguments={},
    )

    result = await executor.execute(call)

    assert result.tool_call_id == "call_2"
    assert result.name == "unknown_tool"
    assert result.output == {
        "code": "tool_not_found",
        "message": "Tool is not available: unknown_tool",
    }
    assert result.is_error is True

@pytest.mark.anyio
async def test_executor_returns_argument_validation_error(
    executor: ToolExecutor,
) -> None:
    call = ToolCall(
        id="call_3",
        name="calculator",
        arguments={
            "operation": "power",
            "left": 2,
            "right": 3,
        },
    )

    result = await executor.execute(call)

    assert result.output == {
        "code": "invalid_tool_arguments",
        "message": "Tool arguments failed validation.",
    }
    assert result.is_error is True

@pytest.mark.anyio
async def test_executor_hides_tool_execution_error(
    executor: ToolExecutor,
) -> None:
    call = ToolCall(
        id="call_4",
        name="calculator",
        arguments={
            "operation": "divide",
            "left": 10,
            "right": 0,
        },
    )

    result = await executor.execute(call)

    assert result.output == {
        "code": "tool_execution_failed",
        "message": "Tool execution failed.",
    }
    assert result.is_error is True