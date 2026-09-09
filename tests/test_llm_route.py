from uuid import UUID

from fastapi.testclient import TestClient


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



def test_generate_llm_rejects_empty_messages(client: TestClient) -> None:
    response = client.post(
        "/api/v1/llm/generate",
        json={"messages": []},
    )

    assert response.status_code == 422