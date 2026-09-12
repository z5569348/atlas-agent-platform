from time import perf_counter

from openai import AsyncOpenAI
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseInputParam,
)
from pydantic import SecretStr

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
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self._model_name = model_name
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

    @staticmethod
    def _build_input(request: LLMRequest) -> ResponseInputParam:
        input_messages: ResponseInputParam = []

        for message in request.messages:
            if message.role == "tool":
                raise ValueError(
                    "Tool messages require tool call metadata."
                )

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

    async def generate(self, request: LLMRequest) -> LLMResponse:
        started_at = perf_counter()

        response = await self._client.responses.create(
            model=self.model_name,
            input=self._build_input(request),
            ## temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            store=False,
        )

        latency_ms = (perf_counter() - started_at) * 1000
        usage = response.usage

        return LLMResponse(
            provider=self.provider_name,
            model=response.model,
            content=response.output_text,
            finish_reason=self._get_finish_reason(response),
            usage=TokenUsage(
                input_tokens=(
                    usage.input_tokens if usage is not None else 0
                ),
                output_tokens=(
                    usage.output_tokens if usage is not None else 0
                ),
            ),
            latency_ms=latency_ms,
        )