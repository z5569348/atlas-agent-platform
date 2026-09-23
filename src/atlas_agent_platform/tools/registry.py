from atlas_agent_platform.tools.base import Tool
from atlas_agent_platform.tools.schemas import ToolDefinition


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.definition.name

        if name in self._tools:
            raise ValueError(
                f"Tool is already registered: {name}"
            )

        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise KeyError(
                f"Tool is not registered: {name}"
            ) from error

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(
            tool.definition
            for tool in self._tools.values()
        )

    def __len__(self) -> int:
        return len(self._tools)