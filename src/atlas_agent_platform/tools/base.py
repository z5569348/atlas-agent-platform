from abc import ABC, abstractmethod
from typing import Protocol

from pydantic import BaseModel, JsonValue

from atlas_agent_platform.tools.schemas import ToolDefinition


class Tool(Protocol):
    @property
    def definition(self) -> ToolDefinition: ...

    async def invoke(
        self,
        arguments: dict[str, JsonValue],
    ) -> JsonValue: ...

class BaseTool[ToolInput: BaseModel](ABC):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_model: type[ToolInput],
    ) -> None:
        self._input_model = input_model
        self._definition = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_model.model_json_schema(),
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def invoke(
        self,
        arguments: dict[str, JsonValue],
    ) -> JsonValue:
        validated_arguments = self._input_model.model_validate(
            arguments
        )
        return await self._execute(validated_arguments)

    @abstractmethod
    async def _execute(
        self,
        arguments: ToolInput,
    ) -> JsonValue:
        raise NotImplementedError