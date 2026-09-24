import pytest
from pydantic import ValidationError

from atlas_agent_platform.llm.schemas import LLMRequest, LLMResponse


def test_llm_request_applies_defaults_and_normalizes_message() -> None:
    request = LLMRequest.model_validate(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "  Explain RAG  ",
                }
            ]
        }
    )

    assert request.messages[0].content == "Explain RAG"
    assert request.temperature == 0.2
    assert request.max_output_tokens == 1024


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": []},
        {"messages": [{"role": "user", "content": "   "}]},
        {"messages": [{"role": "developer", "content": "Hello"}]},
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 2.1,
        },
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "unknown_field": True,
        },
    ],
)
def test_llm_request_rejects_invalid_payload(payload: object) -> None:
    with pytest.raises(ValidationError):
        LLMRequest.model_validate(payload)


def test_llm_response_generates_unique_request_ids() -> None:
    payload = {
        "provider": "mock",
        "model": "mock-model",
        "content": "Mock response",
        "finish_reason": "stop",
        "usage": {
            "input_tokens": 2,
            "output_tokens": 2,
        },
        "latency_ms": 1.5,
    }

    first_response = LLMResponse.model_validate(payload)
    second_response = LLMResponse.model_validate(payload)

    assert first_response.request_id != second_response.request_id
    assert first_response.usage.input_tokens == 2
    assert first_response.usage.output_tokens == 2


def test_llm_response_rejects_negative_token_usage() -> None:
    with pytest.raises(ValidationError):
        LLMResponse.model_validate(
            {
                "provider": "mock",
                "model": "mock-model",
                "content": "Mock response",
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": -1,
                    "output_tokens": 2,
                },
                "latency_ms": 1.5,
            }
        )

def test_llm_request_accepts_tool_definitions() -> None:
    request = LLMRequest.model_validate(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Calculate 6 times 7.",
                }
            ],
            "tools": [
                {
                    "name": "calculator",
                    "description": "Perform arithmetic.",
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                }
            ],
        }
    )

    assert len(request.tools) == 1
    assert request.tools[0].name == "calculator"

def test_llm_response_accepts_tool_calls() -> None:
    response = LLMResponse.model_validate(
        {
            "provider": "openai",
            "model": "test-model",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "name": "calculator",
                    "arguments": {
                        "operation": "multiply",
                        "left": 6,
                        "right": 7,
                    },
                }
            ],
            "finish_reason": "tool_call",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
            },
            "latency_ms": 25.0,
        }
    )

    assert response.content == ""
    assert response.finish_reason == "tool_call"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_123"