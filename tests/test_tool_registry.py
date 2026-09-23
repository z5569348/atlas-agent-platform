import pytest

from atlas_agent_platform.tools.builtin.calculator import (
    CalculatorTool,
)
from atlas_agent_platform.tools.registry import ToolRegistry


def test_registry_registers_and_gets_tool() -> None:
    registry = ToolRegistry()
    tool = CalculatorTool()

    registry.register(tool)

    assert len(registry) == 1
    assert registry.get("calculator") is tool

def test_registry_exposes_tool_definitions() -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    definitions = registry.definitions

    assert isinstance(definitions, tuple)
    assert len(definitions) == 1
    assert definitions[0].name == "calculator"

def test_registry_rejects_duplicate_tool_name() -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    with pytest.raises(
        ValueError,
        match="Tool is already registered: calculator",
    ):
        registry.register(CalculatorTool())

def test_registry_rejects_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        KeyError,
        match="Tool is not registered: unknown",
    ):
        registry.get("unknown")