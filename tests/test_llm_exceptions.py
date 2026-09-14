import pytest

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
    (
        "exception_type",
        "expected_code",
        "expected_retryable",
    ),
    [
        (LLMProviderError, "llm_provider_error", False),
        (LLMAuthenticationError, "llm_authentication_error", False),
        (LLMQuotaExceededError, "llm_quota_exceeded", False),
        (LLMRateLimitError, "llm_rate_limit", True),
        (LLMInvalidRequestError, "llm_invalid_request", False),
        (LLMTimeoutError, "llm_timeout", True),
        (LLMConnectionError, "llm_connection_error", True),
        (LLMUpstreamServiceError, "llm_upstream_service_error", True),
    ],
)
def test_llm_provider_error_contract(
    exception_type: type[LLMProviderError],
    expected_code: str,
    expected_retryable: bool,
) -> None:
    error = exception_type(
        "Safe public message",
        provider="openai",
    )

    assert str(error) == "Safe public message"
    assert error.provider == "openai"
    assert error.code == expected_code
    assert error.retryable is expected_retryable