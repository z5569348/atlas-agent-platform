from atlas_agent_platform.llm.capabilities import ModelProfile
from atlas_agent_platform.llm.policies import apply_model_request_policy
from atlas_agent_platform.llm.providers.base import LLMProvider
from atlas_agent_platform.llm.schemas import LLMRequest, LLMResponse


class LLMGatewayService:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self._profile = ModelProfile(
            provider=provider.provider_name,
            model_name=provider.model_name,
            capabilities=provider.capabilities,
        )
        
    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        policy_result = apply_model_request_policy(
            request,
            self._profile,
        )
        response = await self._provider.generate(
            policy_result.request
        )

        if not policy_result.warnings:
            return response

        warnings = [
            *response.warnings,
            *policy_result.warnings,
        ]

        return response.model_copy(
            update={"warnings": warnings}
        )