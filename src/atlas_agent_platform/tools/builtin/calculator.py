from typing import Literal, assert_never

from pydantic import BaseModel, ConfigDict, FiniteFloat, JsonValue

from atlas_agent_platform.tools.base import BaseTool


class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal[
        "add",
        "subtract",
        "multiply",
        "divide",
    ]
    left: FiniteFloat
    right: FiniteFloat

class CalculatorTool(BaseTool[CalculatorInput]):
    def __init__(self) -> None:
        super().__init__(
            name="calculator",
            description=(
                "Perform basic arithmetic using add, subtract, "
                "multiply, or divide."
            ),
            input_model=CalculatorInput,
        )

    async def _execute(
        self,
        arguments: CalculatorInput,
    ) -> JsonValue:
        match arguments.operation:
            case "add":
                result = arguments.left + arguments.right
            case "subtract":
                result = arguments.left - arguments.right
            case "multiply":
                result = arguments.left * arguments.right
            case "divide":
                if arguments.right == 0:
                    raise ValueError("Cannot divide by zero.")

                result = arguments.left / arguments.right
            case _:
                assert_never(arguments.operation)

        return {"result": result}