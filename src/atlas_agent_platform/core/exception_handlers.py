from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

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


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    provider: str
    retryable: bool


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail

def get_llm_error_status(error: LLMProviderError) -> int:
    if isinstance(error, LLMRateLimitError):
        return status.HTTP_429_TOO_MANY_REQUESTS

    if isinstance(error, LLMInvalidRequestError):
        return status.HTTP_400_BAD_REQUEST

    if isinstance(error, LLMTimeoutError):
        return status.HTTP_504_GATEWAY_TIMEOUT

    if isinstance(
        error,
        (
            LLMAuthenticationError,
            LLMConnectionError,
        ),
    ):
        return status.HTTP_502_BAD_GATEWAY

    if isinstance(
        error,
        (
            LLMQuotaExceededError,
            LLMUpstreamServiceError,
        ),
    ):
        return status.HTTP_503_SERVICE_UNAVAILABLE

    return status.HTTP_502_BAD_GATEWAY

async def llm_provider_error_handler(
    _request: Request,
    error: Exception,
) -> JSONResponse:
    if not isinstance(error, LLMProviderError):
        raise error
    payload = ErrorResponse(
        error=ErrorDetail(
            code=error.code,
            message=str(error),
            provider=error.provider,
            retryable=error.retryable,
        )
    )

    return JSONResponse(
        status_code=get_llm_error_status(error),
        content=payload.model_dump(),
    )