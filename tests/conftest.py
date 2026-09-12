from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from atlas_agent_platform.llm.factory import get_llm_provider
from atlas_agent_platform.llm.providers.base import LLMProvider
from atlas_agent_platform.llm.providers.mock import MockLLMProvider
from atlas_agent_platform.main import app


def get_test_llm_provider() -> LLMProvider:
    return MockLLMProvider(model_name="mock-model")

@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_llm_provider] = (
        get_test_llm_provider
    )

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(
            get_llm_provider,
            None,
        )