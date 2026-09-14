import json
from unittest.mock import MagicMock

import pytest
from fastapi import Request, status

from atlas_agent_platform.core.exception_handlers import (
    get_llm_error_status,
    llm_provider_error_handler,
)
from atlas_agent_platform.llm.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMInvalidRequestError,
    LLMProviderError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamServiceError,
)


@pytest.mark.parametrize(
    ("exception_type", "expected_status"),
    [
        (LLMProviderError, status.HTTP_502_BAD_GATEWAY),
        (LLMAuthenticationError, status.HTTP_502_BAD_GATEWAY),
        (LLMConnectionError, status.HTTP_502_BAD_GATEWAY),
        (LLMInvalidRequestError, status.HTTP_400_BAD_REQUEST),
        (LLMQuotaExceededError, status.HTTP_503_SERVICE_UNAVAILABLE),
        (LLMRateLimitError, status.HTTP_429_TOO_MANY_REQUESTS),
        (LLMTimeoutError, status.HTTP_504_GATEWAY_TIMEOUT),
        (LLMUpstreamServiceError, status.HTTP_503_SERVICE_UNAVAILABLE),
    ],
)
def test_get_llm_error_status(
    exception_type: type[LLMProviderError],
    expected_status: int,
) -> None:
    error = exception_type(
        "Safe public message",
        provider="openai",
    )

    assert get_llm_error_status(error) == expected_status

@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_llm_provider_error_handler_response() -> None:
    request = MagicMock(spec=Request)
    error = LLMQuotaExceededError(
        "Model provider quota is exhausted.",
        provider="openai",
    )

    response = await llm_provider_error_handler(
        request,
        error,
    )
    payload = json.loads(bytes(response.body))

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert payload == {
        "error": {
            "code": "llm_quota_exceeded",
            "message": "Model provider quota is exhausted.",
            "provider": "openai",
            "retryable": False,
        }
    }