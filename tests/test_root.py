from fastapi.testclient import TestClient


def test_root_returns_service_message(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Atlas Agent Platform is running",
    }