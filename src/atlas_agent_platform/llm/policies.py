from pydantic import BaseModel, ConfigDict

from atlas_agent_platform.llm.capabilities import ModelProfile
from atlas_agent_platform.llm.exceptions import LLMInvalidRequestError
from atlas_agent_platform.llm.schemas import LLMRequest


class RequestPolicyResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    request: LLMRequest
    warnings: tuple[str, ...] = ()


def apply_model_request_policy(
    request: LLMRequest,
    profile: ModelProfile,
) -> RequestPolicyResult:

    if request.tools and not profile.capabilities.supports_tool_calling:
        raise LLMInvalidRequestError(
            (f"Model {profile.provider}/{profile.model_name} does not support tool calling."),
            provider=profile.provider,
        )

    model_limit = profile.capabilities.max_output_tokens

    if request.max_output_tokens <= model_limit:
        return RequestPolicyResult(request=request)

    adjusted_request = request.model_copy(update={"max_output_tokens": model_limit})
    warning = (
        f"max_output_tokens was reduced from "
        f"{request.max_output_tokens} to {model_limit} "
        f"for {profile.provider}/{profile.model_name}."
    )

    return RequestPolicyResult(
        request=adjusted_request,
        warnings=(warning,),
    )
