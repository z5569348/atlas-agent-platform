from uuid import UUID

from fastapi import status
from fastapi.testclient import TestClient

from atlas_agent_platform.llm.capabilities import ModelCapabilities
from atlas_agent_platform.llm.exceptions import (
    LLMQuotaExceededError,
)
from atlas_agent_platform.llm.factory import get_llm_provider
from atlas_agent_platform.llm.providers.base import LLMProvider
from atlas_agent_platform.llm.providers.mock import MockLLMProvider
from atlas_agent_platform.llm.schemas import LLMRequest, LLMResponse
from atlas_agent_platform.main import app


def test_generate_llm_response(client: TestClient) -> None:
    response = client.post(
        "/api/v1/llm/generate",
        json={
            "messages": [
                {"role": "user", "content": "Explain RAG"},
            ]
        },
    )

    assert response.status_code == 200

    payload = response.json()
    request_id = UUID(payload["request_id"])

    assert str(request_id) == payload["request_id"]
    assert payload["provider"] == "mock"
    assert payload["model"] == "mock-model"
    assert payload["content"] == "Mock response: Explain RAG"
    assert payload["finish_reason"] == "stop"
    assert payload["usage"]["input_tokens"] > 0
    assert payload["usage"]["output_tokens"] > 0
    assert payload["latency_ms"] >= 0

class QuotaExceededProvider:
    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return "test-model"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            max_output_tokens=32_768,
        )

    async def generate(
        self,
        _request: LLMRequest,
    ) -> LLMResponse:
        raise LLMQuotaExceededError(
            "Model provider quota is exhausted.",
            provider=self.provider_name,
        )

def get_quota_exceeded_provider() -> LLMProvider:
    return QuotaExceededProvider()

def get_limited_mock_provider() -> LLMProvider:
    return MockLLMProvider(
        model_name="limited-model",
        capabilities=ModelCapabilities(
            max_output_tokens=100,
        ),
    )

def test_generate_llm_rejects_empty_messages(client: TestClient) -> None:
    response = client.post(
        "/api/v1/llm/generate",
        json={"messages": []},
    )

    assert response.status_code == 422

def test_generate_llm_reports_adjusted_output_limit(
    client: TestClient,
) -> None:
    app.dependency_overrides[get_llm_provider] = (
        get_limited_mock_provider
    )

    response = client.post(
        "/api/v1/llm/generate",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            "max_output_tokens": 300,
        },
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["warnings"] == [
        (
            "max_output_tokens was reduced from 300 to 100 "
            "for mock/limited-model."
        )
    ]

def test_generate_llm_handles_quota_error(
    client: TestClient,
) -> None:
    app.dependency_overrides[get_llm_provider] = (
        get_quota_exceeded_provider
    )

    response = client.post(
        "/api/v1/llm/generate",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ]
        },
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json() == {
        "error": {
            "code": "llm_quota_exceeded",
            "message": "Model provider quota is exhausted.",
            "provider": "openai",
            "retryable": False,
        }
    }