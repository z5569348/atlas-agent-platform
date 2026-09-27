from collections.abc import Sequence
from dataclasses import dataclass
from json import JSONDecodeError, dumps
from time import perf_counter
from typing import cast

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
from atlas_agent_platform.llm.providers.base import LLMProviderTurn
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
from atlas_agent_platform.tools.schemas import ToolResult


@dataclass(
    frozen=True,
    slots=True,
)
class OpenAITurnState:
    request: LLMRequest
    input_items: ResponseInputParam


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

    def _build_request_params(
        self,
        request: LLMRequest,
        input_items: ResponseInputParam,
    ) -> ResponseCreateParamsNonStreaming:
        request_params: ResponseCreateParamsNonStreaming = {
            "model": self.model_name,
            "input": input_items,
            "max_output_tokens": request.max_output_tokens,
            "store": False,
            "stream": False,
        }

        if self.capabilities.supports_temperature:
            request_params["temperature"] = request.temperature

        if request.tools:
            request_params["tools"] = [
                to_openai_function_tool(definition) for definition in request.tools
            ]

        return request_params

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

    def _to_llm_response(
        self,
        response: Response,
        started_at: float,
    ) -> LLMResponse:
        try:
            tool_calls = [
                from_openai_function_call(item)
                for item in response.output
                if isinstance(
                    item,
                    ResponseFunctionToolCall,
                )
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
            "tool_call" if tool_calls else self._get_finish_reason(response)
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

    async def _request_response(
        self,
        request: LLMRequest,
        input_items: ResponseInputParam,
    ) -> Response:

        try:
            request_params = self._build_request_params(
                request,
                input_items,
            )

            response = await self._client.responses.create(**request_params)
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

        return response

    async def _create_turn(
        self,
        request: LLMRequest,
        input_items: ResponseInputParam,
    ) -> LLMProviderTurn:
        started_at = perf_counter()
        response = await self._request_response(
            request,
            input_items,
        )
        llm_response = self._to_llm_response(
            response,
            started_at,
        )
        replay_items = cast(
            ResponseInputParam,
            [
                *input_items,
                *response.output,
            ],
        )

        return LLMProviderTurn(
            response=llm_response,
            state=OpenAITurnState(
                request=request,
                input_items=replay_items,
            ),
        )

    async def start_turn(
        self,
        request: LLMRequest,
    ) -> LLMProviderTurn:
        input_items = self._build_input(request)
        return await self._create_turn(
            request,
            input_items,
        )

    async def continue_turn(
        self,
        turn: LLMProviderTurn,
        tool_results: Sequence[ToolResult],
    ) -> LLMProviderTurn:
        if not isinstance(turn.state, OpenAITurnState):
            raise TypeError("Expected OpenAITurnState.")

        expected_ids = [call.id for call in turn.response.tool_calls]
        result_ids = [result.tool_call_id for result in tool_results]

        if not expected_ids:
            raise ValueError("Cannot continue a turn without tool calls.")

        if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(expected_ids):
            raise ValueError("Tool results do not match the pending tool calls.")

        output_items = [
            {
                "type": "function_call_output",
                "call_id": result.tool_call_id,
                "output": dumps(
                    result.output,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
            for result in tool_results
        ]
        input_items = cast(
            ResponseInputParam,
            [
                *turn.state.input_items,
                *output_items,
            ],
        )

        return await self._create_turn(
            turn.state.request,
            input_items,
        )

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        turn = await self.start_turn(request)
        return turn.response
