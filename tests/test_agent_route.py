from collections.abc import Iterator
from uuid import UUID

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from atlas_agent_platform.agents.factory import get_agent_service
from atlas_agent_platform.agents.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    AgentStep,
)
from atlas_agent_platform.llm.schemas import LLMResponse, TokenUsage
from atlas_agent_platform.main import app


class FakeAgentService:
    async def run(
        self,
        request: AgentRunRequest,
    ) -> AgentRunResponse:
        llm_response = LLMResponse(
            provider="test-provider",
            model="test-model",
            content=f"Agent response: {request.messages[-1].content}",
            finish_reason="stop",
            usage=TokenUsage(
                input_tokens=4,
                output_tokens=3,
            ),
            latency_ms=1.0,
        )

        return AgentRunResponse(
            status="completed",
            content=llm_response.content,
            steps=(
                AgentStep(
                    iteration=1,
                    llm_response=llm_response,
                    tool_results=(),
                ),
            ),
            usage=llm_response.usage,
        )

def get_fake_agent_service() -> FakeAgentService:
    return FakeAgentService()


@pytest.fixture
def agent_client(
    client: TestClient,
) -> Iterator[TestClient]:
    app.dependency_overrides[get_agent_service] = (
        get_fake_agent_service
    )

    try:
        yield client
    finally:
        app.dependency_overrides.pop(
            get_agent_service,
            None,
        )

def test_run_agent_returns_response(
    agent_client: TestClient,
) -> None:
    response = agent_client.post(
        "/api/v1/agents/run",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Calculate 12 times 8",
                }
            ]
        },
    )

    assert response.status_code == status.HTTP_200_OK

    payload = response.json()
    run_id = UUID(payload["run_id"])

    assert str(run_id) == payload["run_id"]
    assert payload["status"] == "completed"
    assert payload["content"] == (
        "Agent response: Calculate 12 times 8"
    )
    assert len(payload["steps"]) == 1
    assert payload["steps"][0]["iteration"] == 1
    assert payload["usage"] == {
        "input_tokens": 4,
        "output_tokens": 3,
    }

def test_run_agent_rejects_invalid_iteration_limit(
    agent_client: TestClient,
) -> None:
    response = agent_client.post(
        "/api/v1/agents/run",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            "max_iterations": 0,
        },
    )

    assert (
        response.status_code
        == status.HTTP_422_UNPROCESSABLE_CONTENT
    )