from time import perf_counter

from atlas_agent_platform.llm.capabilities import ModelCapabilities
from atlas_agent_platform.llm.schemas import LLMRequest, LLMResponse, TokenUsage


class MockLLMProvider:
    def __init__(
        self,
        model_name: str = "mock-model",
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self._model_name = model_name
        self._capabilities = capabilities or ModelCapabilities(
            max_output_tokens=32_768,
        )
    @property
    def provider_name(self) -> str:
        return "mock"
    @property
    def model_name(self) -> str:
        return self._model_name
    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, (len(text) + 3) // 4)
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        started_at = perf_counter()
        last_message = request.messages[-1]
        content = f"Mock response: {last_message.content}"

        input_tokens = sum(
            self._estimate_tokens(message.content)
            for message in request.messages
        )
        output_tokens = self._estimate_tokens(content)
        latency_ms = (perf_counter() - started_at) * 1000

        return LLMResponse(
            provider=self.provider_name,
            model=self.model_name,
            content=content,
            finish_reason="stop",
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            latency_ms=latency_ms,
        )