from pydantic import ValidationError

from atlas_agent_platform.tools.registry import ToolRegistry
from atlas_agent_platform.tools.schemas import ToolCall, ToolResult


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(
        self,
        call: ToolCall,
    ) -> ToolResult:
        try:
            tool = self._registry.get(call.name)
        except KeyError:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                output={
                    "code": "tool_not_found",
                    "message": (
                        f"Tool is not available: {call.name}"
                    ),
                },
                is_error=True,
            )

        try:
            output = await tool.invoke(call.arguments)
        except ValidationError:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                output={
                    "code": "invalid_tool_arguments",
                    "message": (
                        "Tool arguments failed validation."
                    ),
                },
                is_error=True,
            )
        except Exception:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                output={
                    "code": "tool_execution_failed",
                    "message": "Tool execution failed.",
                },
                is_error=True,
            )

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            output=output,
        )