from atlas_agent_platform.agents.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    AgentStep,
)
from atlas_agent_platform.llm.capabilities import ModelProfile
from atlas_agent_platform.llm.policies import (
    RequestPolicyResult,
    apply_model_request_policy,
)
from atlas_agent_platform.llm.providers.base import AgentLLMProvider
from atlas_agent_platform.llm.schemas import (
    LLMRequest,
    TokenUsage,
)
from atlas_agent_platform.tools.executor import ToolExecutor
from atlas_agent_platform.tools.registry import ToolRegistry
from atlas_agent_platform.tools.schemas import ToolResult


class AgentService:
    def __init__(
        self,
        provider: AgentLLMProvider,
        registry: ToolRegistry,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._executor = ToolExecutor(registry)

    def _build_llm_request(
        self,
        request: AgentRunRequest,
    ) -> LLMRequest:
        return LLMRequest(
            messages=request.messages,
            tools=list(self._registry.definitions),
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
        )

    def _apply_request_policy(
        self,
        request: LLMRequest,
    ) -> RequestPolicyResult:
        profile = ModelProfile(
            provider=self._provider.provider_name,
            model_name=self._provider.model_name,
            capabilities=self._provider.capabilities,
        )
        return apply_model_request_policy(
            request,
            profile,
        )

    async def run(
        self,
        request: AgentRunRequest,
    ) -> AgentRunResponse:
        llm_request = self._build_llm_request(request)
        policy_result = self._apply_request_policy(llm_request)
        turn = await self._provider.start_turn(
            policy_result.request
        )

        steps: list[AgentStep] = []
        warnings = list(policy_result.warnings)
        input_tokens = 0
        output_tokens = 0

        for iteration in range(
            1,
            request.max_iterations + 1,
        ):
            response = turn.response
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
            warnings.extend(response.warnings)

            if not response.tool_calls:
                steps.append(
                    AgentStep(
                        iteration=iteration,
                        llm_response=response,
                    )
                )
                return AgentRunResponse(
                    status="completed",
                    content=response.content,
                    steps=tuple(steps),
                    usage=TokenUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    ),
                    warnings=tuple(warnings),
                )

            if iteration == request.max_iterations:
                steps.append(
                    AgentStep(
                        iteration=iteration,
                        llm_response=response,
                    )
                )
                warnings.append(
                    "Agent stopped after reaching "
                    f"max_iterations={request.max_iterations}."
                )
                return AgentRunResponse(
                    status="max_iterations_reached",
                    content=response.content,
                    steps=tuple(steps),
                    usage=TokenUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    ),
                    warnings=tuple(warnings),
                )

            tool_results_list: list[ToolResult] = []
            for call in response.tool_calls:
                tool_results_list.append(
                    await self._executor.execute(call)
                )

            tool_results = tuple(tool_results_list)
            steps.append(
                AgentStep(
                    iteration=iteration,
                    llm_response=response,
                    tool_results=tool_results,
                )
            )
            turn = await self._provider.continue_turn(
                turn,
                tool_results,
            )

        raise RuntimeError("Agent loop ended unexpectedly.")