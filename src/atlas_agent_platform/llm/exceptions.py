from typing import ClassVar


class LLMProviderError(Exception):
    code: ClassVar[str] = "llm_provider_error"
    retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        provider: str,
    ) -> None:
        super().__init__(message)
        self.provider = provider

class LLMAuthenticationError(LLMProviderError):
    code: ClassVar[str] = "llm_authentication_error"


class LLMQuotaExceededError(LLMProviderError):
    code: ClassVar[str] = "llm_quota_exceeded"


class LLMRateLimitError(LLMProviderError):
    code: ClassVar[str] = "llm_rate_limit"
    retryable: ClassVar[bool] = True

class LLMInvalidRequestError(LLMProviderError):
    code: ClassVar[str] = "llm_invalid_request"


class LLMTimeoutError(LLMProviderError):
    code: ClassVar[str] = "llm_timeout"
    retryable: ClassVar[bool] = True


class LLMConnectionError(LLMProviderError):
    code: ClassVar[str] = "llm_connection_error"
    retryable: ClassVar[bool] = True


class LLMUpstreamServiceError(LLMProviderError):
    code: ClassVar[str] = "llm_upstream_service_error"
    retryable: ClassVar[bool] = True