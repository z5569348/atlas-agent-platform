from collections.abc import Sequence

import pytest

from atlas_agent_platform.agents.schemas import AgentRunRequest
from atlas_agent_platform.agents.service import AgentService
from atlas_agent_platform.llm.capabilities import ModelCapabilities
from atlas_agent_platform.llm.providers.base import LLMProviderTurn
from atlas_agent_platform.llm.schemas import (
    ChatMessage,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)
from atlas_agent_platform.tools.builtin.calculator import CalculatorTool
from atlas_agent_platform.tools.registry import ToolRegistry
from atlas_agent_platform.tools.schemas import ToolCall, ToolResult


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ScriptedAgentProvider:
    def __init__(
        self,
        responses: Sequence[LLMResponse],
    ) -> None:
        self._responses = tuple(responses)
        self.start_request: LLMRequest | None = None
        self.continued_results: list[
            tuple[ToolResult, ...]
        ] = []

    @property
    def provider_name(self) -> str:
        return "scripted"

    @property
    def model_name(self) -> str:
        return "scripted-model"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_tool_calling=True,
            max_output_tokens=128_000,
        )

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        turn = await self.start_turn(request)
        return turn.response

    async def start_turn(
        self,
        request: LLMRequest,
    ) -> LLMProviderTurn:
        self.start_request = request
        return LLMProviderTurn(
            response=self._responses[0],
            state=0,
        )

    async def continue_turn(
        self,
        turn: LLMProviderTurn,
        tool_results: Sequence[ToolResult],
    ) -> LLMProviderTurn:
        if not isinstance(turn.state, int):
            raise TypeError("Expected integer scripted state.")

        self.continued_results.append(
            tuple(tool_results)
        )
        next_index = turn.state + 1

        return LLMProviderTurn(
            response=self._responses[next_index],
            state=next_index,
        )

@pytest.mark.anyio
async def test_agent_service_executes_tool_and_returns_final_response() -> None:
    tool_call = ToolCall(
        id="call_123",
        name="calculator",
        arguments={
            "operation": "multiply",
            "left": 6,
            "right": 7,
        },
    )
    provider = ScriptedAgentProvider(
        responses=[
            LLMResponse(
                provider="scripted",
                model="scripted-model",
                content="",
                finish_reason="tool_call",
                tool_calls=[tool_call],
                usage=TokenUsage(
                    input_tokens=10,
                    output_tokens=5,
                ),
                latency_ms=10.0,
            ),
            LLMResponse(
                provider="scripted",
                model="scripted-model",
                content="6 times 7 equals 42.",
                finish_reason="stop",
                usage=TokenUsage(
                    input_tokens=8,
                    output_tokens=4,
                ),
                latency_ms=8.0,
            ),
        ],
    )

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    service = AgentService(
        provider=provider,
        registry=registry,
    )

    request = AgentRunRequest(
        messages=[
            ChatMessage(
                role="user",
                content="Calculate 6 times 7.",
            ),
        ],
    )

    result = await service.run(request)

    assert result.status == "completed"
    assert result.content == "6 times 7 equals 42."
    assert len(result.steps) == 2
    assert result.usage == TokenUsage(
        input_tokens=18,
        output_tokens=9,
    )
    assert result.warnings == ()

    assert provider.start_request is not None
    assert [
        definition.name
        for definition in provider.start_request.tools
    ] == ["calculator"]

    assert len(provider.continued_results) == 1
    tool_results = provider.continued_results[0]
    assert len(tool_results) == 1

    tool_result = tool_results[0]
    assert tool_result.tool_call_id == "call_123"
    assert tool_result.name == "calculator"
    assert tool_result.output == {"result": 42.0}
    assert tool_result.is_error is False

    assert result.steps[0].tool_results == tool_results
    assert result.steps[1].tool_results == ()

@pytest.mark.anyio
async def test_agent_service_stops_before_tool_execution_at_limit() -> None:
    provider = ScriptedAgentProvider(
        responses=[
            LLMResponse(
                provider="scripted",
                model="scripted-model",
                content="",
                finish_reason="tool_call",
                tool_calls=[
                    ToolCall(
                        id="call_limit",
                        name="calculator",
                        arguments={
                            "operation": "add",
                            "left": 1,
                            "right": 2,
                        },
                    ),
                ],
                usage=TokenUsage(
                    input_tokens=6,
                    output_tokens=3,
                ),
                latency_ms=5.0,
            ),
        ],
    )

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    service = AgentService(
        provider=provider,
        registry=registry,
    )

    result = await service.run(
        AgentRunRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="Calculate 1 plus 2.",
                ),
            ],
            max_iterations=1,
        )
    )

    assert result.status == "max_iterations_reached"
    assert result.content == ""
    assert len(result.steps) == 1
    assert result.steps[0].tool_results == ()
    assert provider.continued_results == []
    assert result.usage == TokenUsage(
        input_tokens=6,
        output_tokens=3,
    )
    assert result.warnings == (
        "Agent stopped after reaching max_iterations=1.",
    )