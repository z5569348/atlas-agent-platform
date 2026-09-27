import pytest
from pydantic import ValidationError

from atlas_agent_platform.agents.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    AgentStep,
)
from atlas_agent_platform.llm.schemas import (
    ChatMessage,
    LLMResponse,
    TokenUsage,
)
from atlas_agent_platform.tools.schemas import (
    ToolCall,
    ToolResult,
)


def test_agent_run_request_applies_defaults() -> None:
    request = AgentRunRequest(
        messages=[
            ChatMessage(
                role="user",
                content="Calculate 6 times 7.",
            ),
        ],
    )

    assert request.temperature == 0.2
    assert request.max_output_tokens == 1024
    assert request.max_iterations == 8

@pytest.mark.parametrize(
    "max_iterations",
    [0, 33],
)
def test_agent_run_request_rejects_invalid_max_iterations(
    max_iterations: int,
) -> None:
    with pytest.raises(ValidationError):
        AgentRunRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="Hello",
                ),
            ],
            max_iterations=max_iterations,
        )

def test_agent_run_response_records_tool_execution() -> None:
    tool_call = ToolCall(
        id="call_123",
        name="calculator",
        arguments={
            "operation": "multiply",
            "left": 6,
            "right": 7,
        },
    )
    tool_result = ToolResult(
        tool_call_id="call_123",
        name="calculator",
        output={"result": 42.0},
    )

    tool_step = AgentStep(
        iteration=1,
        llm_response=LLMResponse(
            provider="openai",
            model="test-model",
            content="",
            finish_reason="tool_call",
            tool_calls=[tool_call],
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=5,
            ),
            latency_ms=12.5,
        ),
        tool_results=(tool_result,),
    )

    final_step = AgentStep(
        iteration=2,
        llm_response=LLMResponse(
            provider="openai",
            model="test-model",
            content="6 times 7 equals 42.",
            finish_reason="stop",
            usage=TokenUsage(
                input_tokens=8,
                output_tokens=4,
            ),
            latency_ms=10.0,
        ),
    )

    response = AgentRunResponse(
        status="completed",
        content="6 times 7 equals 42.",
        steps=(tool_step, final_step),
        usage=TokenUsage(
            input_tokens=18,
            output_tokens=9,
        ),
    )

    assert response.status == "completed"
    assert response.content == "6 times 7 equals 42."
    assert len(response.steps) == 2
    assert response.steps[0].tool_results == (tool_result,)
    assert response.usage.input_tokens == 18
    assert response.usage.output_tokens == 9