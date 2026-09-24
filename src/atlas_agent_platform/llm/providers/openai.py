from json import JSONDecodeError
from time import perf_counter

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError,
)
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseFunctionToolCall,
    ResponseInputParam,
)
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming,
)
from pydantic import SecretStr, ValidationError

from atlas_agent_platform.llm.capabilities import ModelCapabilities
from atlas_agent_platform.llm.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMInvalidRequestError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamServiceError,
)
from atlas_agent_platform.llm.providers.openai_tools import (
    from_openai_function_call,
    to_openai_function_tool,
)
from atlas_agent_platform.llm.schemas import (
    FinishReason,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)


class OpenAILLMProvider:
    def __init__(
        self,
        api_key: SecretStr,
        model_name: str,
        capabilities: ModelCapabilities,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self._model_name = model_name
        self._capabilities = capabilities
        self._client = AsyncOpenAI(
            api_key=api_key.get_secret_value(),
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    @staticmethod
    def _build_input(request: LLMRequest) -> ResponseInputParam:
        input_messages: ResponseInputParam = []

        for message in request.messages:
            if message.role == "tool":
                raise ValueError("Tool messages require tool call metadata.")

            input_message: EasyInputMessageParam = {
                "role": message.role,
                "content": message.content,
            }
            input_messages.append(input_message)

        return input_messages

    @staticmethod
    def _get_finish_reason(response: Response) -> FinishReason:
        details = response.incomplete_details

        if details is None:
            return "stop"

        if details.reason == "content_filter":
            return "content_filter"

        return "length"

    @staticmethod
    def _is_quota_error(error: RateLimitError) -> bool:
        quota_codes = {
            "insufficient_quota",
            "credit_balance_exhausted",
        }

        return error.code in quota_codes or error.type == "insufficient_quota"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        started_at = perf_counter()

        try:
            input_messages = self._build_input(request)
            request_params: ResponseCreateParamsNonStreaming = {
                "model": self.model_name,
                "input": input_messages,
                "max_output_tokens": request.max_output_tokens,
                "store": False,
                "stream": False,
            }

            if self.capabilities.supports_temperature:
                request_params["temperature"] = request.temperature

            if request.tools:
                request_params["tools"] = [
                    to_openai_function_tool(definition)
                    for definition in request.tools
                ]

            response = await self._client.responses.create(
                **request_params
            )
        except (
            AuthenticationError,
            PermissionDeniedError,
        ) as error:
            raise LLMAuthenticationError(
                "Model provider authentication failed.",
                provider=self.provider_name,
            ) from error
        except RateLimitError as error:
            if self._is_quota_error(error):
                raise LLMQuotaExceededError(
                    "Model provider quota is exhausted.",
                    provider=self.provider_name,
                ) from error

            raise LLMRateLimitError(
                "Model provider rate limit was exceeded.",
                provider=self.provider_name,
            ) from error
        except BadRequestError as error:
            raise LLMInvalidRequestError(
                "Model provider rejected the request.",
                provider=self.provider_name,
            ) from error
        except APITimeoutError as error:
            raise LLMTimeoutError(
                "Model provider request timed out.",
                provider=self.provider_name,
            ) from error
        except APIConnectionError as error:
            raise LLMConnectionError(
                "Could not connect to model provider.",
                provider=self.provider_name,
            ) from error
        except APIStatusError as error:
            if error.status_code >= 500:
                raise LLMUpstreamServiceError(
                    "Model provider service is unavailable.",
                    provider=self.provider_name,
                ) from error

            raise LLMInvalidRequestError(
                "Model provider returned an API error.",
                provider=self.provider_name,
            ) from error

        try:
            tool_calls = [
                from_openai_function_call(item)
                for item in response.output
                if isinstance(item, ResponseFunctionToolCall)
            ]
        except (
            JSONDecodeError,
            ValidationError,
        ) as error:
            raise LLMUpstreamServiceError(
                "Model provider returned invalid tool arguments.",
                provider=self.provider_name,
            ) from error

        finish_reason: FinishReason = (
            "tool_call"
            if tool_calls
            else self._get_finish_reason(response)
        )
        latency_ms = (perf_counter() - started_at) * 1000
        usage = response.usage

        return LLMResponse(
            provider=self.provider_name,
            model=response.model,
            content=response.output_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=TokenUsage(
                input_tokens=(usage.input_tokens if usage is not None else 0),
                output_tokens=(usage.output_tokens if usage is not None else 0),
            ),
            latency_ms=latency_ms,
        )
